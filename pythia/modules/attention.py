# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn

from pythia.modules.layers import GatedTanh, ModalCombineLayer, TransformLayer

import math

class AttentionLayer(nn.Module):
    def __init__(self, image_dim, question_dim, **kwargs):
        super(AttentionLayer, self).__init__()

        combine_type = kwargs["modal_combine"]["type"]
        combine_params = kwargs["modal_combine"]["params"]
        modal_combine_layer = ModalCombineLayer(
            combine_type, image_dim, question_dim, **combine_params
        )

        transform_type = kwargs["transform"]["type"]
        transform_params = kwargs["transform"]["params"]
        transform_layer = TransformLayer(
            transform_type, modal_combine_layer.out_dim, **transform_params
        )

        normalization = kwargs["normalization"]

        self.module = TopDownAttention(
            modal_combine_layer, transform_layer, normalization
        )

        if getattr(self.module, "out_dim"):
            self.out_dim = self.module.out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class ConcatenationAttention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size):
        super(ConcatenationAttention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.fa = GatedTanh(image_feat_dim + txt_rnn_embeding_dim, hidden_size)
        self.lc = nn.Linear(hidden_size, 1)

    def forward(self, image_feat, question_embedding):
        _, num_location, _ = image_feat.shape
        question_embedding_expand = torch.unsqueeze(question_embedding, 1).expand(
            -1, num_location, -1
        )
        concat_feature = torch.cat((image_feat, question_embedding_expand), dim=2)
        raw_attention = self.lc(self.fa(concat_feature))
        # softmax across locations
        attention_weights = nn.functional.softmax(raw_attention, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class ProjectAttention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout=0.2):
        super(ProjectAttention, self).__init__()
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim
        self.fa_image = GatedTanh(image_feat_dim, hidden_size)
        self.fa_txt = GatedTanh(txt_rnn_embeding_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lc = nn.Linear(hidden_size, 1)

    def compute_raw_att(self, image_feat, question_embedding):
        num_location = image_feat.shape[1]
        image_fa = self.fa_image(image_feat)
        question_fa = self.fa_txt(question_embedding)
        question_fa_expand = torch.unsqueeze(question_fa, 1).expand(
            -1, num_location, -1
        )
        joint_feature = image_fa * question_fa_expand
        joint_feature = self.dropout(joint_feature)
        raw_attention = self.lc(joint_feature)
        return raw_attention

    def forward(self, image_feat, question_embedding):
        raw_attention = self.compute_raw_att(image_feat, question_embedding)
        # softmax across locations
        attention_weights = nn.functional.softmax(raw_attention, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class DoubleProjectAttention(nn.Module):
    def __init__(self, image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout=0.2):
        super(DoubleProjectAttention, self).__init__()
        self.att1 = ProjectAttention(
            image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout
        )
        self.att2 = ProjectAttention(
            image_feat_dim, txt_rnn_embeding_dim, hidden_size, dropout
        )
        self.image_feat_dim = image_feat_dim
        self.txt_embeding_dim = txt_rnn_embeding_dim

    def forward(self, image_feat, question_embedding):
        att1 = self.att1.compute_raw_att(image_feat, question_embedding)
        att2 = self.att2.compute_raw_att(image_feat, question_embedding)
        raw_attn_weights = att1 + att2
        # softmax across locations
        attention_weights = nn.functional.softmax(raw_attn_weights, dim=1)
        attention_weights = attention_weights.expand_as(image_feat)
        return attention_weights


class TopDownAttention(nn.Module):
    EPS = 1.0e-08

    def __init__(self, combination_layer, transform_module, normalization):
        super(TopDownAttention, self).__init__()
        self.combination_layer = combination_layer
        self.normalization = normalization
        self.transform = transform_module
        self.out_dim = self.transform.out_dim

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

    def forward(self, image_feat, question_embedding, image_locs=None):
        # N x K x joint_dim
        joint_feature = self.combination_layer(image_feat, question_embedding)
        # N x K x n_att
        raw_attn = self.transform(joint_feature)
        if self.normalization.lower() == "softmax":
            attention = nn.functional.softmax(raw_attn, dim=1)
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)
                masked_attention_sum = torch.sum(masked_attention, dim=1, keepdim=True)
                masked_attention_sum += masked_attention_sum.eq(0).float() + self.EPS
                masked_attention = masked_attention / masked_attention_sum
            else:
                masked_attention = attention

        elif self.normalization.lower() == "sigmoid":
            attention = torch.sigmoid(raw_attn)
            masked_attention = attention
            if image_locs is not None:
                masked_attention = self._mask_attentions(attention, image_locs)

        return masked_attention


#########################################################################

# Spatial Self_Attention
class SpatialBertSelfAttention(nn.Module):
    """ Spatial Attention  """
    def __init__(self, config):
        super(SpatialBertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # self.scale = nn.Parameter(torch.tensor(10))

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, distance_matrix=None, head_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Add Distance matrix M
        if (distance_matrix != None):
            attention_scores = attention_scores + distance_matrix.unsqueeze(1)
            # atteniton_scores *= self.scale

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

class SpatialBertAttention(nn.Module):
    def __init__(self, config):
        super(SpatialBertAttention, self).__init__()
        self.self = SpatialBertSelfAttention(config)
        from pytorch_transformers.modeling_bert import BertSelfOutput
        self.output = BertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)


        
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, distance_matrix, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, distance_matrix, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SpatialBertLayer(nn.Module):
    def __init__(self, config):
        super(SpatialBertLayer, self).__init__()
        from pytorch_transformers.modeling_bert import (BertIntermediate,BertOutput)
        self.attention = SpatialBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, distance_matrix, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, distance_matrix, head_mask)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class SpatialBertEncoder(nn.Module):
    def __init__(self, config):
        super(SpatialBertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([SpatialBertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, distance_matrix, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, distance_matrix, head_mask[i])
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)




class NormalizeScale(nn.Module):
    def __init__(self, dim, init_norm=1):
        super(NormalizeScale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        bottom_normalized = F.normalize(bottom, p=2, dim=-1)
        bottom_normalized_scaled = bottom_normalized * self.weight
        return bottom_normalized_scaled
