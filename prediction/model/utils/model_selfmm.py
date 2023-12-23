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
from prediction.config import SELFMMConfig


class SELF_MM(nn.Module):
    def __init__(self, senti_modal_list, finetune = False):
        super(SELF_MM, self).__init__()
        # text subnets
        self.senti_modal_list = senti_modal_list
        self.text_model = BertTextEncoder(use_finetune=SELFMMConfig.bert_finetune, transformers=SELFMMConfig.transformers, pretrained=SELFMMConfig.pretrained)

        # audio-vision subnets
        audio_in, video_in = SELFMMConfig.input_d_a, SELFMMConfig.input_d_v
        self.audio_model = AuViSubNet(audio_in, SELFMMConfig.a_lstm_hidden_size, SELFMMConfig.audio_out, \
                            num_layers=SELFMMConfig.a_lstm_layers, dropout=SELFMMConfig.a_lstm_dropout)
        self.video_model = AuViSubNet(video_in, SELFMMConfig.v_lstm_hidden_size, SELFMMConfig.video_out, \
                            num_layers=SELFMMConfig.v_lstm_layers, dropout=SELFMMConfig.v_lstm_dropout)

        # the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=SELFMMConfig.post_fusion_dropout)
        self.post_fusion_layer_1 = nn.Linear(SELFMMConfig.text_out + SELFMMConfig.video_out + SELFMMConfig.audio_out, SELFMMConfig.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(SELFMMConfig.post_fusion_dim, SELFMMConfig.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(SELFMMConfig.post_fusion_dim, 1)

        # the classify layer for text
        self.post_text_dropout = nn.Dropout(p=SELFMMConfig.post_text_dropout)
        self.post_text_layer_1 = nn.Linear(SELFMMConfig.text_out, SELFMMConfig.post_text_dim)
        self.post_text_layer_2 = nn.Linear(SELFMMConfig.post_text_dim, SELFMMConfig.post_text_dim)
        self.post_text_layer_3 = nn.Linear(SELFMMConfig.post_text_dim, 1)

        # the classify layer for audio
        self.post_audio_dropout = nn.Dropout(p=SELFMMConfig.post_audio_dropout)
        self.post_audio_layer_1 = nn.Linear(SELFMMConfig.audio_out, SELFMMConfig.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(SELFMMConfig.post_audio_dim, SELFMMConfig.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(SELFMMConfig.post_audio_dim, 1)

        # the classify layer for video
        self.post_video_dropout = nn.Dropout(p=SELFMMConfig.post_video_dropout)
        self.post_video_layer_1 = nn.Linear(SELFMMConfig.video_out, SELFMMConfig.post_video_dim)
        self.post_video_layer_2 = nn.Linear(SELFMMConfig.post_video_dim, SELFMMConfig.post_video_dim)
        self.post_video_layer_3 = nn.Linear(SELFMMConfig.post_video_dim, 1)

        self.load_state_dict({k[6:]:v for k,v in torch.load('model/pretrained_models/self_mm-mosi_lvu_outputdim300_bertfinetune.pth').items()})
        #self_mm-mosi_lvu_outputdim300_bertfinetune.pth的参数的key多了一个Model.
        for model in [self.text_model, self.audio_model, self.video_model,
                      self.post_fusion_dropout, self.post_fusion_layer_1, self.post_fusion_layer_2, self.post_fusion_layer_3,
                      self.post_text_dropout, self.post_text_layer_1, self.post_text_layer_2, self.post_text_layer_3,
                      self.post_audio_dropout, self.post_audio_layer_1,  self.post_audio_layer_2, self.post_audio_layer_3,
                      self.post_video_dropout, self.post_video_layer_1,  self.post_video_layer_2, self.post_video_layer_3]:
            for param in model.parameters():
                param.requires_grad = False
        if finetune:
            for model in [self.post_fusion_layer_2, self.post_text_layer_2, self.post_audio_layer_2, self.post_video_layer_2]:
                for param in model.parameters():
                    param.requires_grad = True


    def forward(self, text=[], text_lengths=[], audio=[], audio_lengths=[], video=[], video_lengths=[]):
        text_h = None
        audio_h = None
        video_h = None
        if 'text' in self.senti_modal_list:
            # # text
            text = self.text_model(text)[:,0,:]
            text_h = self.post_text_dropout(text)
            text_h = F.relu(self.post_text_layer_1(text_h), inplace=False)
            text_h = F.relu(self.post_text_layer_2(text_h), inplace=False)
        if 'audio' in self.senti_modal_list:
            # audio
            audio = self.audio_model(audio, audio_lengths)
            audio_h = self.post_audio_dropout(audio)
            audio_h = F.relu(self.post_audio_layer_1(audio_h), inplace=False)
            audio_h = F.relu(self.post_audio_layer_2(audio_h), inplace=False)
        if 'video' in self.senti_modal_list:
            # vision
            video = self.video_model(video, video_lengths)
            video_h = self.post_video_dropout(video)
            video_h = F.relu(self.post_video_layer_1(video_h), inplace=False)
            video_h = F.relu(self.post_video_layer_2(video_h), inplace=False)
        return text_h, audio_h, video_h

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
