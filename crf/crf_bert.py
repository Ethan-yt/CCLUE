#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2020/7/21 10:57 上午
# @Author : Ethan
# @Email : yantan@bit.edu.cn
# @File : crf_model
from abc import ABC

import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel
from transformers.file_utils import add_start_docstrings_to_model_forward, \
    add_code_sample_docstrings
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.bert.modeling_bert import BERT_INPUTS_DOCSTRING
from transformers.models.roberta.modeling_roberta import _TOKENIZER_FOR_DOC, \
    _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC

from crf.crf import CRF


class CRFBertForTokenClassification(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            crf_mask = labels != nn.CrossEntropyLoss().ignore_index
            crf_mask = crf_mask[:, 1:]
            crf_labels = labels[:, 1:]
            crf_labels = torch.relu(crf_labels)

            crf_logits = logits[:, 1:, :]
            loss = -self.crf(crf_logits, crf_labels, crf_mask, reduction='mean')

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(self, emissions):
        return self.crf.decode(torch.tensor(emissions).cuda())
