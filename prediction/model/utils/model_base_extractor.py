"""
From: https://github.com/thuiar/Self-MM
Paper: Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning for Multimodal Sentiment Analysis
"""
# self supervised multimodal multi-task learning network

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from prediction.model.utils.BertTextEncoder import BertTextEncoder
from prediction.config import BaseExtractorModelConfig


class BaseExtractorModel(nn.Module):
    def __init__(self, modal):
        super(BaseExtractorModel, self).__init__()
        self.modal = modal
        if self.modal == 'video':
            feat_in = BaseExtractorModelConfig.input_d_v
            self.model = AuViSubNet(feat_in, BaseExtractorModelConfig.v_lstm_hidden_size, BaseExtractorModelConfig.video_out, \
                                          num_layers=BaseExtractorModelConfig.v_lstm_layers,dropout=BaseExtractorModelConfig.v_lstm_dropout)
            self.post_dropout = nn.Dropout(p=BaseExtractorModelConfig.post_video_dropout)
            self.post_layer_1 = nn.Linear(BaseExtractorModelConfig.video_out, BaseExtractorModelConfig.post_video_dim)
        elif self.modal == 'audio':
            feat_in = BaseExtractorModelConfig.input_d_a
            self.model = AuViSubNet(feat_in, BaseExtractorModelConfig.a_lstm_hidden_size, BaseExtractorModelConfig.audio_out, \
                                          num_layers=BaseExtractorModelConfig.a_lstm_layers, dropout=BaseExtractorModelConfig.a_lstm_dropout)
            self.post_dropout = nn.Dropout(p=BaseExtractorModelConfig.post_audio_dropout)
            self.post_layer_1 = nn.Linear(BaseExtractorModelConfig.audio_out, BaseExtractorModelConfig.post_audio_dim)
        else:
            # self.model = BertTextEncoder(use_finetune=BaseExtractorModelConfig.bert_finetune, transformers=BaseExtractorModelConfig.transformers, pretrained=BaseExtractorModelConfig.pretrained)
            # for param in self.model.parameters():
            #     #固定bert
            #     param.requires_grad = False
            # self.post_dropout = nn.Dropout(p=BaseExtractorModelConfig.post_text_dropout)
            self.post_layer_1 = nn.Linear(BaseExtractorModelConfig.text_out, BaseExtractorModelConfig.post_text_dim)

    def forward(self, feat=[], feat_lengths=[]):
        if self.modal == 'video' or self.modal == 'audio' :
            res_h = self.post_dropout(self.model(feat, feat_lengths))
        else: #self.modal = 'text'
            # res = self.post_dropout(self.model(feat)[:,0,:])
            res_h = feat[:,0,:]
        res_h = F.relu(self.post_layer_1(res_h), inplace=False)
        return res_h


class AuViSubNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        '''
        super(AuViSubNet, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, out_size)

    def forward(self, x, lengths):
        '''
        x: (batch_size, sequence_len, in_size)
        '''
        packed_sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)
        h = self.dropout(final_states[0].squeeze())
        y_1 = self.linear_1(h)
        return y_1
