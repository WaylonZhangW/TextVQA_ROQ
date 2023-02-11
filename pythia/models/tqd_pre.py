# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import math
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

from pythia.common.registry import registry
from pythia.models.base_model import BaseModel
from pythia.modules.layers import ClassifierLayer
from pythia.modules.embeddings import TextEmbedding
from pythia.modules.encoders import ImageEncoder
from pythia.modules.embeddings import ImageEmbedding
from pythia.utils.configuration import ConfigNode

from pythia.modules.attention import SpatialBertEncoder

import pdb

@registry.register_model("tqd_pretrain")
class TQD(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.mmt_config = BertConfig(**self.config.mmt)
        self.ocr_config = BertConfig(**self.config.ocr_attention)
        self._datasets = registry.get("config").datasets.split(",")

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        # self._init_text_embeddings()
        self._build_obj_encoding()
        self._build_ocr_encoding()
        self._build_mmt()
        self._build_output()
        self._build_query_obj()
        self._build_query_ocr()


    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768

        self.text_bert_config = BertConfig(**self.config.text_bert)
        if self.config.text_bert_init_from_bert_base:
            self.text_bert = TextBert.from_pretrained(
                'bert-base-uncased', config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append({
                'module': self.text_bert,
                'lr_scale': self.config.lr_scale_text_bert,
            })
        else:
            self.writer.write('NOT initializing text_bert from BERT_BASE')
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            self.writer.write(
                'Projecting text_bert output to {} dim'.format(
                    self.mmt_config.hidden_size
                )
            )
            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()

    def _build_obj_encoding(self):
        # object appearance feature: Faster R-CNN
        self.obj_faster_rcnn_fc7 = ImageEncoder(
            encoder_type='finetune_faster_rcnn_fpn_fc7',
            in_dim=2048,
            weights_file='detectron/fc6/fc7_w.pkl',
            bias_file='detectron/fc6/fc7_b.pkl',
            model_data_dir=self.config["model_data_dir"]
        )
        # apply smaller lr to pretrained Faster R-CNN fc7
        self.finetune_modules.append({
            'module': self.obj_faster_rcnn_fc7,
            'lr_scale': self.config.lr_scale_frcn,
        })
        self.linear_obj_feat_to_mmt_in = nn.Linear(
            self.config.obj.mmt_in_dim, self.mmt_config.hidden_size
        )

        # object location feature: relative bounding box coordinates (4-dim)
        self.linear_obj_bbox_to_mmt_in = nn.Linear(
            4, self.mmt_config.hidden_size
        )

        self.obj_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.obj_drop = nn.Dropout(self.config.obj.dropout_prob)

    def _build_ocr_encoding(self):
        ### OCR grouping

        self.linear_ocr_bbox = nn.Linear(8, 512)

        init_norm = self.config.scale
        self.bbox_normalizer =  NormalizeScale(dim=512,init_norm=init_norm)
        self.ocr_normalizer = NormalizeScale(dim=512, init_norm=init_norm)
        self.ocrs_attention = OCR_ATTENTION(self.ocr_config)

        self.finetune_modules.append({
            'module': self.ocrs_attention,
            'lr_scale': self.config.lr_scale_ocrs_attention,
        })

        ###########################################################################

        self.linear_ocr_feat_to_mmt_in = nn.Linear(
            self.config.ocr.mmt_in_dim, self.mmt_config.hidden_size
        )

        # OCR location feature: relative bounding box coordinates (4-dim)
        self.linear_ocr_bbox_to_mmt_in = nn.Linear(
            8, self.mmt_config.hidden_size
        )

        self.ocr_feat_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_bbox_layer_norm = BertLayerNorm(self.mmt_config.hidden_size)
        self.ocr_drop = nn.Dropout(self.config.ocr.dropout_prob)

    def _build_mmt(self):
        self.mmt = MMT(self.mmt_config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append({
            'module': self.mmt,
            'lr_scale': self.config.lr_scale_mmt,
        })

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.ocr_ptr_net = OcrPtrNet(**self.config.classifier.ocr_ptr_net)

        # fixed answer vocabulary scores
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        num_choices -= self.config.classifier.ocr_max_num
        self.classifier = ClassifierLayer(
            self.config["classifier"]["type"],
            in_dim=self.mmt_config.hidden_size,
            out_dim=num_choices,
            **self.config["classifier"]["params"]
        )

        self.answer_processor = registry.get(
            self._datasets[0] + "_answer_processor"
        )


#############################################################
    def _build_query_obj(self):

        self.query_obj = nn.ModuleList([
                    TransformerDecoderLayer(768, 8, 0.1) for _ in range(2)
                ])
        self.mlp = nn.Sequential(
            nn.Linear(self.mmt_config.hidden_size*2, 768),
            nn.Linear(768, 100),
            nn.Linear(100, 1),
        )

    def _build_query_ocr(self):
        # self.query_ocr = MultiHeadAttention(n_head =8, d_model=self.mmt_config.hidden_size, d_k=self.mmt_config.hidden_size, d_v=self.mmt_config.hidden_size)
        self.query_ocr = nn.ModuleList([
                    TransformerDecoderLayer(768, 8, 0.1) for _ in range(2)
                ])



################################################


    def forward(self, sample_list):
        # fwd_results holds intermediate forward pass results
        # TODO possibly replace it with another sample list
        fwd_results = {}
        self._forward_txt_encoding(sample_list, fwd_results)
        self._forward_obj_encoding(sample_list, fwd_results)
        self._forward_ocr_encoding(sample_list, fwd_results)
        # new added
        self._forward_query_obj(sample_list, fwd_results)
        self._forward_query_ocr(sample_list, fwd_results)
        self._forward_mmt_and_output(sample_list, fwd_results)

        obj_label = sample_list.obj_label
        results = {"scores": fwd_results["scores"],
                    "obj_region":fwd_results['obj_region'],
                    "obj_label": obj_label,
                }

        return results

    def _forward_txt_encoding(self, sample_list, fwd_results):

        fwd_results['txt_inds'] = sample_list.text
        # binary mask of valid text (question words) vs padding
        fwd_results['txt_mask'] = _get_mask(
            sample_list.text_len, sample_list.text.size(1)
        )
        ## ocr text
        fwd_results['ocr_txt_inds'] = sample_list.ocr_text
        fwd_results['ocr_txt_mask'] = _get_mask(
            sample_list.ocr_text_len, sample_list.ocr_text.size(1)
        )
        ## obj text
        fwd_results['obj_txt_inds'] = sample_list.obj_text
        fwd_results['obj_txt_mask'] = _get_mask(
            sample_list.obj_text_len, sample_list.obj_text.size(1)
        )
        all_text = torch.cat([fwd_results['txt_inds'],fwd_results['ocr_txt_inds'],fwd_results['obj_txt_inds']],dim=-1)

        all_text_mask = torch.cat([fwd_results['txt_mask'],fwd_results['ocr_txt_mask'],fwd_results['obj_txt_mask']],dim=-1)
        text_bert_out = self.text_bert(
            txt_inds=all_text,
            txt_mask=all_text_mask
        )

        text_bert_out = self.text_bert_out_linear(text_bert_out)
        q_len = sample_list.text.size(1)
        ocr_len = sample_list.ocr_text.size(1)
        fwd_results['txt_emb'] = text_bert_out[:,:q_len,]
        fwd_results['ocr_txt_emb'] = text_bert_out[:,q_len:q_len+ocr_len,]
        fwd_results['obj_txt_emb'] = text_bert_out[:,q_len+ocr_len:,]

        assert fwd_results['txt_emb'].size(1) == 20
        assert fwd_results['ocr_txt_emb'].size(1) == 50
        assert fwd_results['obj_txt_emb'].size(1) == 100



    def _forward_obj_encoding(self, sample_list, fwd_results):
        # object appearance feature: Faster R-CNN fc7
        obj_fc6 = sample_list.image_feature_0
        obj_fc7 = self.obj_faster_rcnn_fc7(obj_fc6)
        obj_fc7 = F.normalize(obj_fc7, dim=-1)

        obj_txt = F.normalize(fwd_results['obj_txt_emb'], dim=-1)
        obj_feat = torch.cat([obj_txt,obj_fc7],dim=-1)
        obj_bbox = sample_list.obj_bbox_coordinates

        obj_mmt_in = (
            self.obj_feat_layer_norm(
                self.linear_obj_feat_to_mmt_in(obj_feat)
            ) + self.obj_bbox_layer_norm(
                self.linear_obj_bbox_to_mmt_in(obj_bbox)
            )
        )
        obj_mmt_in = self.obj_drop(obj_mmt_in)
        fwd_results['obj_mmt_in'] = obj_mmt_in

        # binary mask of valid object vs padding
        obj_nums = sample_list.image_info_0.max_features
        fwd_results['obj_mask'] = _get_mask(obj_nums, obj_mmt_in.size(1))

    def _forward_ocr_encoding(self, sample_list, fwd_results):
        # OCR FastText feature (300-dim)
        ocr_fasttext = sample_list.context_feature_0
        ocr_fasttext = F.normalize(ocr_fasttext, dim=-1)
        assert ocr_fasttext.size(-1) == 300

        # OCR PHOC feature (604-dim)
        ocr_phoc = sample_list.context_feature_1
        ocr_phoc = F.normalize(ocr_phoc, dim=-1)
        assert ocr_phoc.size(-1) == 604

        #########################################################################################
        # OCR appearance feature: Faster R-CNN fc7
        # TODO
        ocr_roi = sample_list.image_feature_1[:, :ocr_fasttext.size(1), :]
        distance_matrix = sample_list.distance_matrix


        ocr_roi = ocr_roi.view(ocr_roi.size(0),ocr_roi.size(1),-1)

        # enrich ocr bbox feature
        ocr_bbox = sample_list.ocr_bbox_coordinates
        ocr_bbox_other_info = torch.zeros(ocr_bbox.size(0),ocr_bbox.size(1),4,dtype=torch.float32, device=ocr_bbox.device)
        # ocr bbox center point coordinate
        ocr_bbox_other_info[:, :, 0] = (ocr_bbox[:, :, 0] + ocr_bbox[:, :, 2]) / 2
        ocr_bbox_other_info[:, :, 1] = (ocr_bbox[:, :, 1] + ocr_bbox[:, :, 3]) / 2
        # ocr bbox height and width
        ocr_bbox_other_info[:, :, 2] = ocr_bbox[:, :, 2] - ocr_bbox[:, :, 0]
        ocr_bbox_other_info[:, :, 3] = ocr_bbox[:, :, 3] - ocr_bbox[:, :, 1]

        ocr_bbox_concat = torch.cat([ocr_bbox,ocr_bbox_other_info],dim=-1)
        ocr_bbox_concat_2 = ocr_bbox_concat
        ocr_bbox_concat = self.linear_ocr_bbox(ocr_bbox_concat)
        ocr_rois =  self.ocr_normalizer(ocr_roi)  + self.bbox_normalizer(ocr_bbox_concat)

        # binary mask of valid OCR vs padding
        ocr_nums = sample_list.context_info_0.max_features
        fwd_results['ocr_mask'] = _get_mask(ocr_nums,ocr_fasttext.size(1))

        ocr_roi_attention = self.ocrs_attention(ocr_visual = ocr_rois,
                                ocr_visual_mask = fwd_results['ocr_mask'],
                                distance_matrix = distance_matrix
        )
        ocr_fc7 = F.normalize(ocr_roi_attention, dim=-1)

        #########################################################################################
        #### tap feature
        ocr_txt = F.normalize(fwd_results['ocr_txt_emb'], dim=-1)
        ocr_tap = sample_list.tap_ocr
        ocr_tap = F.normalize(ocr_tap, dim=-1)
        ocr_feat = torch.cat(
            [ocr_txt,ocr_tap, ocr_fc7],
            dim=-1
        )

        ocr_mmt_in = (
            self.ocr_feat_layer_norm(
                self.linear_ocr_feat_to_mmt_in(ocr_feat)
            ) + self.ocr_bbox_layer_norm(
                self.linear_ocr_bbox_to_mmt_in(ocr_bbox_concat_2)
            )
        )
        ocr_mmt_in = self.ocr_drop(ocr_mmt_in)
        fwd_results['ocr_mmt_in'] = ocr_mmt_in


    def process_text_embedding(self, text_bert_out):
        # question self-attention
        embedding = self.text_embedding_model(text_bert_out)
        return embedding[0], embedding[1], embedding[2], embedding[3]

####################################################

    def _forward_query_obj(self, sample_list, fwd_results):
        """
        input:
        query: bert word embedding: ([32, 20, 768])
        obj: object embedding (feat+bbox) ([32, 100, 768])
        transformer encoder: 4-layer bert encoder
        output:
        question_obj
        obj
        """

        object_token,_ = fwd_results['txt_emb'].max(dim=1,keepdim=True)

        q_obj = torch.cat([object_token,fwd_results['txt_emb']],dim=1)
        txt_mask = fwd_results['txt_mask']  # (bs,20)
        obj_token_mask = torch.ones(q_obj.size(0),1,dtype=torch.float32,device=q_obj.device)
        mask = torch.cat([obj_token_mask,txt_mask],dim=1)

        # obj_mask = fwd_results['obj_mask']  # (bs,50)
        # txt_obj_mask = txt_mask.unsqueeze(-1) * obj_mask.unsqueeze(1)

        # q_obj = fwd_results['txt_emb']
        # mask = fwd_results['txt_mask']
        for layer in self.query_obj:
            q_obj = layer(q_obj,fwd_results['obj_mmt_in'],mask=mask)

        fwd_results['mmt1_s_o_output'] = q_obj

        obj = fwd_results['obj_mmt_in']
        obj_region = q_obj[:,0,:].unsqueeze(1).repeat(1,obj.size(1),1)
        obj_region = torch.cat([obj,obj_region],dim=-1)
        fwd_results['obj_region'] = self.mlp(obj_region).squeeze(2)



    def _forward_query_ocr(self, sample_list, fwd_results):
        """
        input:
        question_obj ([32, 20, 768])
        ocr vector (visual+semantic+all) ([32, 50, 768*3])  -> ([32, 50, 768])
        transformer decoder: 8-head multiheadattention
        output:
        q_ocr_obj

        """

        question_obj = fwd_results['mmt1_s_o_output']  # ([32, 20, 768])
        total_ocr = fwd_results['ocr_mmt_in']

        ocr_mask = fwd_results['ocr_mask']  # (bs,50)
        txt_mask = fwd_results['txt_mask']  # (bs,20)
        obj_token_mask = torch.ones(question_obj.size(0),1,dtype=torch.float32,device=question_obj.device)
        mask = torch.cat([obj_token_mask,txt_mask],dim=1)
        txt_ocr_mask = mask.unsqueeze(-1) * ocr_mask.unsqueeze(1)

        for layer in self.query_ocr:
            question_obj = layer(question_obj, total_ocr,mask=mask,target_mask=txt_ocr_mask)
        # q_ocr_obj, attens = self.query_ocr(question_obj, total_ocr, total_ocr)   # ([32, 20, 768])
        # attens: (bs,8,20,50)

        fwd_results["q_ocr_obj"] = question_obj



####################################################

    def _forward_mmt(self, sample_list, fwd_results):

        #use q
        q = fwd_results['txt_emb']
        q_mask = fwd_results['txt_mask']

        #use q-obj
        q_obj = fwd_results['mmt1_s_o_output'][:,0,:].unsqueeze(1)
        q_obj_mask = torch.ones(q_obj.size(0),q_obj.size(1),dtype=torch.float32,device=q_obj.device)

        ### q_obj_ocr
        q_ocr_obj = fwd_results['q_ocr_obj'][:,0,:].unsqueeze(1)
        q_ocr_obj_mask = torch.ones(q_ocr_obj.size(0),q_ocr_obj.size(1),dtype=torch.float32,device=q_ocr_obj.device)

        mmt_results = self.mmt(
            q = q,
            q_mask = q_mask,
            q_obj=q_obj,
            q_obj_mask=q_obj_mask,
            q_ocr_obj=q_ocr_obj,
            q_ocr_obj_mask=q_ocr_obj_mask,
            ocr_emb=fwd_results['ocr_mmt_in'],
            ocr_mask=fwd_results['ocr_mask'],
            fixed_ans_emb=self.classifier.module.weight,
            prev_inds=fwd_results['prev_inds'],
        )

        fwd_results.update(mmt_results)


    def _forward_output(self, sample_list, fwd_results):


        embedding3 = fwd_results['mmt2_q_ocr_obj_output']

        mmt_dec_output = fwd_results['mmt2_dec_output']
        score_feature = torch.cat([embedding3, mmt_dec_output[:,1:,:]], dim=-2)
        mmt_ocr_output = fwd_results['mmt2_ocr_output']
        ocr_mask = fwd_results['ocr_mask']

        fixed_scores = self.classifier(score_feature)
        dynamic_ocr_scores = self.ocr_ptr_net(score_feature, mmt_ocr_output, ocr_mask)
        scores = torch.cat([fixed_scores, dynamic_ocr_scores], dim=-1)
        fwd_results['scores'] = scores

    def _forward_mmt_and_output(self, sample_list, fwd_results):
        if self.training:
            fwd_results['prev_inds'] = sample_list.train_prev_inds.clone()
            self._forward_mmt(sample_list, fwd_results)
            self._forward_output(sample_list, fwd_results)
        else:
            dec_step_num = sample_list.train_prev_inds.size(1)
            # fill prev_inds with BOS_IDX at index 0, and zeros elsewhere
            fwd_results['prev_inds'] = torch.zeros_like(
                sample_list.train_prev_inds
            )
            fwd_results['prev_inds'][:, 0] = self.answer_processor.BOS_IDX

            # greedy decoding at test time
            for t in range(dec_step_num):
                self._forward_mmt(sample_list, fwd_results)
                self._forward_output(sample_list, fwd_results)

                # find the highest scoring output (either a fixed vocab
                # or an OCR), and add it to prev_inds for auto-regressive
                # decoding
                argmax_inds = fwd_results["scores"].argmax(dim=-1)
                fwd_results['prev_inds'][:, 1:] = argmax_inds[:, :-1]

    def get_optimizer_parameters(self, config):
        optimizer_param_groups = []

        base_lr = config.optimizer_attributes.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * m['lr_scale']
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
            p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups


class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output


class MMT(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.prev_pred_embeddings = PrevPredEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self,
                q,
                q_mask,
                q_obj,
                q_obj_mask,
                q_ocr_obj,
                q_ocr_obj_mask,
                ocr_emb,
                ocr_mask,
                fixed_ans_emb,
                prev_inds):

        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        dec_emb = self.prev_pred_embeddings(fixed_ans_emb, ocr_emb, prev_inds)

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        dec_mask = torch.zeros(
            dec_emb.size(0),
            dec_emb.size(1),
            dtype=torch.float32,
            device=dec_emb.device
        )
        encoder_inputs = torch.cat(
            [q, q_obj, q_ocr_obj, ocr_emb, dec_emb],
            dim=1
        )
        attention_mask = torch.cat(
            [q_mask, q_obj_mask, q_ocr_obj_mask, ocr_mask, dec_mask],
            dim=1
        )


        q_max_num = q_mask.size(-1)
        q_obj_max_num = q_obj_mask.size(-1)
        q_ocr_obj_max_num = q_ocr_obj_mask.size(-1)
        ocr_max_num = ocr_mask.size(-1)
        dec_max_num = dec_mask.size(-1)
        # offsets of each modality in the joint embedding space
        q_begin = 0
        q_end = q_begin + q_max_num
        q_obj_begin = q_max_num
        q_obj_end = q_obj_begin + q_obj_max_num
        q_ocr_obj_begin = q_max_num + q_obj_max_num
        q_ocr_obj_end = q_ocr_obj_begin + q_ocr_obj_max_num
        ocr_begin = q_max_num + q_obj_max_num +q_obj_max_num
        ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )
        # decoding step elements can attend to themselves in a causal manner
        extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = \
            _get_causal_mask(dec_max_num, encoder_inputs.device)

        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )

        mmt2_seq_output = encoder_outputs[0]
        mmt2_q_output = mmt2_seq_output[:, q_begin:q_end]
        mmt2_q_obj_output = mmt2_seq_output[:, q_obj_begin:q_obj_end]
        mmt2_q_ocr_obj_output = mmt2_seq_output[:, q_ocr_obj_begin:q_ocr_obj_end]
        mmt2_ocr_output = mmt2_seq_output[:, ocr_begin:ocr_end]
        mmt2_dec_output = mmt2_seq_output[:, -dec_max_num:]

        results = {
            'mmt2_q_output': mmt2_q_output,
            'mmt2_q_obj_output': mmt2_q_obj_output,
            'mmt2_q_ocr_obj_output': mmt2_q_ocr_obj_output,
            'mmt2_ocr_output':mmt2_ocr_output,
            'mmt2_dec_output':mmt2_dec_output
        }
        return results


class OCR_ATTENTION(BertPreTrainedModel):
    ''' attention between visual part of OCR tokens '''
    def __init__(self,config):
        super().__init__(config)
        self.encoder = SpatialBertEncoder(config)
        self.init_weights()

    def forward(self, ocr_visual,ocr_visual_mask,distance_matrix):
        # TODO
        encoder_inputs = ocr_visual
        attention_mask = ocr_visual_mask
        to_seq_length = attention_mask.size(1)
        from_seq_length = to_seq_length
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.repeat(
            1, 1, from_seq_length, 1
        )

        extended_attention_mask = (1.0 - extended_attention_mask) * -100000.0

        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            distance_matrix,
            head_mask=head_mask
        )

        seq_output = encoder_outputs[0]

        return seq_output

class OcrPtrNet(nn.Module):
    def __init__(self, hidden_size, query_key_size=None):
        super().__init__()

        if query_key_size is None:
            query_key_size = hidden_size
        self.hidden_size = hidden_size
        self.query_key_size = query_key_size

        self.query = nn.Linear(hidden_size, query_key_size)
        self.key = nn.Linear(hidden_size, query_key_size)

    def forward(self, query_inputs, key_inputs, attention_mask):
        extended_attention_mask = (1.0 - attention_mask) * -10000.0
        assert extended_attention_mask.dim() == 2
        extended_attention_mask = extended_attention_mask.unsqueeze(1)

        query_layer = self.query(query_inputs)
        if query_layer.dim() == 2:
            query_layer = query_layer.unsqueeze(1)
            squeeze_result = True
        else:
            squeeze_result = False
        key_layer = self.key(key_inputs)

        scores = torch.matmul(
            query_layer,
            key_layer.transpose(-1, -2)
        )
        scores = scores / math.sqrt(self.query_key_size)
        scores = scores + extended_attention_mask
        if squeeze_result:
            scores = scores.squeeze(1)

        return scores


class PrevPredEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        MAX_DEC_LENGTH = 100
        MAX_TYPE_NUM = 5
        hidden_size = config.hidden_size
        ln_eps = config.layer_norm_eps

        self.position_embeddings = nn.Embedding(MAX_DEC_LENGTH, hidden_size)
        self.token_type_embeddings = nn.Embedding(MAX_TYPE_NUM, hidden_size)

        self.ans_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.ocr_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_layer_norm = BertLayerNorm(hidden_size, eps=ln_eps)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, ans_emb, ocr_emb, prev_inds):
        assert prev_inds.dim() == 2 and prev_inds.dtype == torch.long
        assert ans_emb.dim() == 2

        batch_size = prev_inds.size(0)
        seq_length = prev_inds.size(1)
        ans_num = ans_emb.size(0)

        # apply layer normalization to both answer embedding and OCR embedding
        # before concatenation, so that they have the same scale
        ans_emb = self.ans_layer_norm(ans_emb)
        ocr_emb = self.ocr_layer_norm(ocr_emb)
        assert ans_emb.size(-1) == ocr_emb.size(-1)
        ans_emb = ans_emb.unsqueeze(0).expand(batch_size, -1, -1)
        ans_ocr_emb_cat = torch.cat([ans_emb, ocr_emb], dim=1)
        raw_dec_emb = _batch_gather(ans_ocr_emb_cat, prev_inds)

        # Add position and type embedding for previous predictions
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=ocr_emb.device
        )
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        # Token type ids: 0 -- vocab; 1 -- OCR
        token_type_ids = prev_inds.ge(ans_num).long()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = position_embeddings + token_type_embeddings
        embeddings = self.emb_layer_norm(embeddings)
        embeddings = self.emb_dropout(embeddings)
        dec_emb = raw_dec_emb + embeddings

        return dec_emb


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i+1):
            mask[i, j] = 1.
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size*length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results


class NormalizeScale(nn.Module):
    def __init__(self, dim, init_norm=1):
        super(NormalizeScale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        bottom_normalized = F.normalize(bottom, p=2, dim=-1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        #q (sz_b,n_head,N=len_q,d_k)
        #k (sz_b,n_head,N=len_k,d_k)
        #v (sz_b,n_head,N=len_v,d_v)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)

        #q (sz_b,len_q,n_head,N * d_k)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)#(1,N,d)

    def forward(self, x):
        # x(B,N,d)
        return x + self.pos_table[:, :x.size(1)].clone().detach()



##################
# Query object

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v,mask=None,target_mask=None):
        """
        q: (B,N,C)
        k,v: (B,M,C)
        mask:  the self-attention mask, (B,N)
        target_mask: the cross-attention mask, (B,N,M)
        return: (B,N,C)
        """
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale  # (bs,num_head,N,M)

        if mask is not None:
            from_seq_length = mask.size(1)
            extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.repeat(
                1, 1, from_seq_length, 1
            )
            attn = attn.masked_fill(extended_attention_mask == 0, -1e9)

        if target_mask is not None:
            extend_target_mask = target_mask.unsqueeze(1) # (bs,num_head,20,50)
            attn = attn.masked_fill(extend_target_mask == 0, -1e9)

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1,
    ):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = Attention(d_model, nhead, proj_drop=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, mem,mask=None,target_mask=None):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v,mask=mask)
        q = self.norm2(x)

        x = x + self.cross_attn(q, mem, mem,target_mask=target_mask)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x



