# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: Update kwargs with defaults
import os
import pickle
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm


from pythia.modules.attention import AttentionLayer
from pythia.modules.layers import Identity
from pythia.utils.vocab import Vocab
from torch.nn.utils.weight_norm import weight_norm

class TextEmbedding(nn.Module):
    def __init__(self, emb_type, **kwargs):
        super(TextEmbedding, self).__init__()
        self.model_data_dir = kwargs.get("model_data_dir", None)
        self.embedding_dim = kwargs.get("embedding_dim", None)

        # Update kwargs here
        if emb_type == "identity":
            self.module = Identity()
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "vocab":
            self.module = VocabEmbedding(**kwargs)
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "preextracted":
            self.module = PreExtractedEmbedding(**kwargs)
        elif emb_type == "bilstm":
            self.module = BiLSTMTextEmbedding(**kwargs)
        elif emb_type == "attention":
            self.module = AttentionTextEmbedding(**kwargs)
        elif emb_type == "attention_two":
            self.module = AttentionTwoTextEmbedding(**kwargs)
        elif emb_type == "attention_three":
            self.module = AttentionTwoTextEmbedding_2(**kwargs)

        elif emb_type == "torch":
            vocab_size = kwargs["vocab_size"]
            embedding_dim = kwargs["embedding_dim"]
            self.module = nn.Embedding(vocab_size, embedding_dim)
            self.module.text_out_dim = self.embedding_dim
        else:
            raise NotImplementedError("Unknown question embedding '%s'" % emb_type)

        self.text_out_dim = self.module.text_out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class VocabEmbedding(nn.Module):
    def __init__(self, embedding_dim, vocab_params):
        self.vocab = Vocab(**vocab_params)
        self.module = self.vocab.get_embedding(nn.Embedding, embedding_dim)

    def forward(self, x):
        return self.module(x)


class BiLSTMTextEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        num_layers,
        dropout,
        bidirectional=False,
        rnn_type="GRU",
    ):
        super(BiLSTMTextEmbedding, self).__init__()
        self.text_out_dim = hidden_dim
        self.bidirectional = bidirectional

        if rnn_type == "LSTM":
            rnn_cls = nn.LSTM
        elif rnn_type == "GRU":
            rnn_cls = nn.GRU

        self.recurrent_encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.recurrent_encoder(x)
        # Return last state
        if self.bidirectional:
            return out[:, -1]

        forward_ = out[:, -1, : self.num_hid]
        backward = out[:, 0, self.num_hid :]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        output, _ = self.recurrent_encoder(x)
        return output


class PreExtractedEmbedding(nn.Module):
    def __init__(self, out_dim, base_path):
        super(PreExtractedEmbedding, self).__init__()
        self.text_out_dim = out_dim
        self.base_path = base_path
        self.cache = {}

    def forward(self, qids):
        embeddings = []
        for qid in qids:
            embeddings.append(self.get_item(qid))
        return torch.stack(embeddings, dim=0)

    @lru_cache(maxsize=5000)
    def get_item(self, qid):
        return np.load(os.path.join(self.base_path, str(qid.item()) + ".npy"))


class AttentionTextEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super(AttentionTextEmbedding, self).__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]

        bidirectional = kwargs.get("bidirectional", False)

        self.recurrent_unit = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        conv1_out = kwargs["conv1_out"]
        conv2_out = kwargs["conv2_out"]
        kernel_size = kwargs["kernel_size"]
        padding = kwargs["padding"]

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        self.recurrent_unit.flatten_parameters()
        # self.recurrent_unit.flatten_parameters()
        lstm_out, _ = self.recurrent_unit(x)  # N * T * hidden_dim
        lstm_drop = self.dropout(lstm_out)  # N * T * hidden_dim
        lstm_reshape = lstm_drop.permute(0, 2, 1)  # N * hidden_dim * T

        qatt_conv1 = self.conv1(lstm_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        return qtt_feature_concat

class AttentionTwoTextEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super(AttentionTwoTextEmbedding, self).__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]

        conv1_out = kwargs["conv1_out"]
        conv2_out = kwargs["conv2_out"]
        kernel_size = kwargs["kernel_size"]
        padding = kwargs["padding"]

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu = nn.ReLU()

        self.conv1_semantic = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2_semantic = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu_semantic = nn.ReLU()

        self.conv1_o = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2_o = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu_o = nn.ReLU()


    def forward(self, x):

        ### x: (bs,20,768)
        batch_size = x.size(0)

        bert_reshape = x.permute(0,2,1) # N * hidden_dim * T

        qatt_conv1 = self.conv1(bert_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)

        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, x)  #
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        # semantic part self-attention
        qatt_conv1_semantic = self.conv1_semantic(bert_reshape)  # N x conv1_out x T
        qatt_relu_semantic = self.relu_semantic(qatt_conv1_semantic)
        qatt_conv2_semantic = self.conv2_semantic(qatt_relu_semantic)  # N x conv2_out x T

        # Over last dim
        qtt_softmax_semantic = nn.functional.softmax(qatt_conv2_semantic, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature_semantic = torch.bmm(qtt_softmax_semantic, x)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat_semantic = qtt_feature_semantic.view(batch_size, -1)

        qatt_conv1_o = self.conv1_o(bert_reshape)  # N x conv1_out x T
        qatt_relu_o = self.relu_o(qatt_conv1_o)
        qatt_conv2_o = self.conv2_o(qatt_relu_o)  # N x conv2_out x T

        # Over last dim
        qtt_softmax_o = nn.functional.softmax(qatt_conv2_o, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature_o = torch.bmm(qtt_softmax_o, x)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat_o = qtt_feature_o.view(batch_size, -1)

        return x, qtt_feature_concat, qtt_feature_concat_semantic, qtt_feature_concat_o,


class AttentionTwoTextEmbedding_2(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super(AttentionTwoTextEmbedding_2, self).__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]

        conv1_out = kwargs["conv1_out"]
        conv2_out = kwargs["conv2_out"]
        kernel_size = kwargs["kernel_size"]
        padding = kwargs["padding"]

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # x.size (bs,20,768)
        batch_size = x.size(0)

        bert_reshape = x.permute(0,2,1) # N * hidden_dim * T

        qatt_conv1 = self.conv1(bert_reshape)  # N x conv1_out x T
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)  # N x conv2_out x T

        # Over last dim
        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        # N * conv2_out * hidden_dim
        qtt_feature = torch.bmm(qtt_softmax, x)
        # N * (conv2_out * hidden_dim)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        return  qtt_feature_concat


class ImageEmbedding(nn.Module):
    """
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    question_embedding:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]


    """

    def __init__(self, img_dim, question_dim, **kwargs):
        super(ImageEmbedding, self).__init__()

        self.image_attention_model = AttentionLayer(img_dim, question_dim, **kwargs)
        self.out_dim = self.image_attention_model.out_dim

    def forward(self, image_feat_variable, question_embedding, image_dims, extra={}):

        attention = self.image_attention_model(
            image_feat_variable, question_embedding, image_dims
        )   #  (bs,100,1)
        att_reshape = attention.permute(0, 2, 1)
        order_vectors = getattr(extra, "order_vectors", None)

        if order_vectors is not None:
            image_feat_variable = torch.cat(
                [image_feat_variable, order_vectors], dim=-1
            )
        tmp_embedding = torch.bmm(
            att_reshape, image_feat_variable
        )  # N x n_att x image_dim
        batch_size = att_reshape.size(0)
        image_embedding = tmp_embedding.view(batch_size, -1)

        return image_embedding, att_reshape


class ImageFinetune(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file):
        super(ImageFinetune, self).__init__()
        with open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3


class ReLUWithWeightNormFC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ReLUWithWeightNormFC, self).__init__()

        layers = []
        layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ImageEmbedding_2(nn.Module):
    """
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    question_embedding:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]


    """

    EPS = 1.0e-08

    def __init__(self, img_dim, question_dim, **kwargs):
        super(ImageEmbedding_2, self).__init__()

        combine_type = kwargs["modal_combine"]["type"]
        modal_params = kwargs["modal_combine"]["params"]
        normalization = kwargs["normalization"]
        transform_type = kwargs["transform"]["type"]
        transform_params = kwargs["transform"]["params"]

        self.fa_image = ReLUWithWeightNormFC(img_dim, modal_params["hidden_dim"])
        self.fa_txt = ReLUWithWeightNormFC(question_dim, modal_params["hidden_dim"])
        self.fa_txt2 = ReLUWithWeightNormFC(768,modal_params["hidden_dim"])
        self.dropout = nn.Dropout(modal_params["dropout"])
        self.dropout2 = nn.Dropout(modal_params["dropout"])
        self.lc = weight_norm(
            nn.Linear(in_features=modal_params["hidden_dim"], out_features=transform_params['out_dim']), dim=None
        )

        self.lc2 = weight_norm(
            nn.Linear(in_features=modal_params["hidden_dim"], out_features=transform_params['out_dim']), dim=None
        )

    @staticmethod
    def _mask_attentions(attention, image_locs):
        batch_size, num_loc, n_att = attention.size()
        tmp1 = attention.new_zeros(num_loc)
        tmp1[:num_loc] = torch.arange(0, num_loc, dtype=attention.dtype).unsqueeze(
            dim=0
        )

        tmp1 = tmp1.expand(batch_size, num_loc)
        tmp2 = image_locs.type(tmp1.type())
        tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
        mask = torch.ge(tmp1, tmp2)
        mask = mask.unsqueeze(dim=2).expand_as(attention)
        attention = attention.masked_fill(mask, 0)
        return attention

    def forward(self, image_feat_variable, question_embedding, query2,image_dims, extra={}):
        # N x K x n_att
        image_fa = self.fa_image(image_feat_variable)
        question_fa = self.fa_txt(question_embedding)
        q_fa_2 = self.fa_txt2(query2)

        if len(image_fa.size()) == 3:
            question_fa_expand = question_fa.unsqueeze(1)
            q_fa_2_expand = q_fa_2.unsqueeze(1)
        else:
            question_fa_expand = question_fa
            q_fa_2_expand = q_fa_2


        joint_feature = self.dropout(image_fa * question_fa_expand)
        raw_attn = self.lc(joint_feature)

        joint_feature_2 = self.dropout2(image_fa * q_fa_2_expand)
        raw_attn_2 = self.lc2(joint_feature_2)
        # 先相加 再 softmax
        raw_attn = raw_attn + raw_attn_2
        #### softmax
        attention = nn.functional.softmax(raw_attn, dim=1)

        # # 先softmax 再 相加
        # attention = (nn.functional.softmax(raw_attn, dim=1) + nn.functional.softmax(raw_attn_2, dim=1)) / 2

        if image_dims is not None:
            masked_attention = self._mask_attentions(attention, image_dims)
            masked_attention_sum = torch.sum(masked_attention, dim=1, keepdim=True)
            masked_attention_sum += masked_attention_sum.eq(0).float() + self.EPS
            masked_attention = masked_attention / masked_attention_sum
        else:
            masked_attention = attention

        att_reshape = masked_attention.permute(0, 2, 1)

        order_vectors = getattr(extra, "order_vectors", None)

        if order_vectors is not None:
            image_feat_variable = torch.cat(
                [image_feat_variable, order_vectors], dim=-1
            )
        tmp_embedding = torch.bmm(
            att_reshape, image_feat_variable
        )  # N x n_att x image_dim
        batch_size = att_reshape.size(0)
        image_embedding = tmp_embedding.view(batch_size, -1)

        return image_embedding, att_reshape

class ImageEmbedding_3(nn.Module):
    """
    parameters:

    input:
    image_feat_variable: [batch_size, num_location, image_feat_dim]
    or a list of [num_location, image_feat_dim]
    when using adaptive number of objects
    question_embedding:[batch_size, txt_embeding_dim]

    output:
    image_embedding:[batch_size, image_feat_dim]


    """

    EPS = 1.0e-08

    def __init__(self, img_dim, question_dim, **kwargs):
        super(ImageEmbedding_3, self).__init__()

        combine_type = kwargs["modal_combine"]["type"]
        modal_params = kwargs["modal_combine"]["params"]
        normalization = kwargs["normalization"]
        transform_type = kwargs["transform"]["type"]
        transform_params = kwargs["transform"]["params"]

        self.fa_image = ReLUWithWeightNormFC(img_dim, modal_params["hidden_dim"])

        self.fa_txt = ReLUWithWeightNormFC(question_dim, modal_params["hidden_dim"])
        self.dropout = nn.Dropout(modal_params["dropout"])
        self.lc = weight_norm(
            nn.Linear(in_features=modal_params["hidden_dim"], out_features=transform_params['out_dim']), dim=None
        )

        self.fa_txt2 = ReLUWithWeightNormFC(768, modal_params["hidden_dim"])
        self.dropout2 = nn.Dropout(modal_params["dropout"])
        self.lc2 = weight_norm(
            nn.Linear(in_features=modal_params["hidden_dim"], out_features=transform_params['out_dim']), dim=None
        )

    @staticmethod
    def _mask_attentions(attention, image_locs):
        batch_size, num_loc, n_att = attention.size()
        tmp1 = attention.new_zeros(num_loc)
        tmp1[:num_loc] = torch.arange(0, num_loc, dtype=attention.dtype).unsqueeze(
            dim=0
        )

        tmp1 = tmp1.expand(batch_size, num_loc)
        tmp2 = image_locs.type(tmp1.type())
        tmp2 = tmp2.unsqueeze(dim=1).expand(batch_size, num_loc)
        mask = torch.ge(tmp1, tmp2)
        mask = mask.unsqueeze(dim=2).expand_as(attention)
        attention = attention.masked_fill(mask, 0)
        return attention

    def forward(self, image_feat_variable, question_embedding, weight ,image_dims, extra={}):

        batch_size = image_feat_variable.size(0)

        # first weighted sum, weight comes from the previous block.
        tmp_1 = torch.bmm(
            weight, image_feat_variable
        ).view(batch_size, -1)

        image_fa = self.fa_image(image_feat_variable)
        question_fa = self.fa_txt(question_embedding)
        q_fa_2 = self.fa_txt2(tmp_1)

        if len(image_fa.size()) == 3:
            question_fa_expand = question_fa.unsqueeze(1)
            q_fa_2_expand = q_fa_2.unsqueeze(1)
        else:
            question_fa_expand = question_fa
            q_fa_2_expand = q_fa_2

        joint_feature = self.dropout(image_fa * question_fa_expand)
        raw_attn = self.lc(joint_feature)

        joint_feature_2 = self.dropout2(image_fa * q_fa_2_expand)
        raw_attn_2 = self.lc2(joint_feature_2)

        # 先相加 再 softmax
        raw_attn = raw_attn + raw_attn_2
        #### softmax
        attention = nn.functional.softmax(raw_attn, dim=1)

       # # 先softmax 再 相加
       #  attention = ( nn.functional.softmax(raw_attn, dim=1) + nn.functional.softmax(raw_attn_2, dim=1) ) / 2

        if image_dims is not None:
            masked_attention = self._mask_attentions(attention, image_dims)
            masked_attention_sum = torch.sum(masked_attention, dim=1, keepdim=True)
            masked_attention_sum += masked_attention_sum.eq(0).float() + self.EPS
            masked_attention = masked_attention / masked_attention_sum
        else:
            masked_attention = attention

        att_reshape = masked_attention.permute(0, 2, 1)

        order_vectors = getattr(extra, "order_vectors", None)

        if order_vectors is not None:
            image_feat_variable = torch.cat(
                [image_feat_variable, order_vectors], dim=-1
            )
        tmp_embedding = torch.bmm(
            att_reshape, image_feat_variable
        )  # N x n_att x image_dim

        image_embedding = tmp_embedding.view(batch_size, -1)

        return image_embedding , att_reshape
