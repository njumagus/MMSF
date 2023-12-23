import torch
from torch import nn
import torch.nn.functional as F

from prediction.model.utils.transformer import TransformerEncoder
from prediction.model.utils.model_base_extractor import BaseExtractorModel
from prediction.model.utils.model_selfmm import SELF_MM
from prediction.config import MMSAModelConfig


class FeatSentiPredictionModel(nn.Module):
    def __init__(self, modal_list, senti_modal_list, senti_feat_fusion_strategies = '', early_fusion=False, early_fusion_strategy='', late_fusion_strategy='', senti_model_finetune=False, senti_attention_models={}):
        """
        Construct a MulT model.
        """
        super(FeatSentiPredictionModel, self).__init__()
        self.modal_list = modal_list
        self.senti_modal_list = senti_modal_list
        self.early_fusion = early_fusion
        self.early_fusion_strategy = early_fusion_strategy
        self.late_fusion_strategy = late_fusion_strategy
        self.senti_model_finetune = senti_model_finetune
        if senti_model_finetune:
            self.senti_model = SELF_MM(self.senti_modal_list, finetune=True)
        self.senti_feat_fusion_strategies = senti_feat_fusion_strategies
        output_dim = MMSAModelConfig.output_dim  # This is actually not a hyperparameter :-)
        self.out_dropout = MMSAModelConfig.out_dropout
        if 'text' in self.modal_list:
            self.d_t = MMSAModelConfig.output_d_t
            self.classifition_input_dim_t = self.d_t
            if 'text' in self.senti_modal_list:
                self.text_model = FeatSentiFusionModel(modal='text', with_senti=True, senti_feat_fusion_strategy=self.senti_feat_fusion_strategies['text'], senti_attention_model = senti_attention_models['text'])
                if self.senti_feat_fusion_strategies['text'] == 'concat':
                    self.classifition_input_dim_t = self.d_t * 2
            else:
                self.text_model = FeatSentiFusionModel(modal='text', with_senti=False)
            if not self.early_fusion:
                #late fusion
                self.proj1_t = nn.Linear(self.classifition_input_dim_t, self.classifition_input_dim_t)
                self.proj2_t = nn.Linear(self.classifition_input_dim_t, self.classifition_input_dim_t)
                self.out_layer_t = nn.Linear(self.classifition_input_dim_t, output_dim)
        if 'audio' in self.modal_list:
            self.d_a = MMSAModelConfig.output_d_a
            self.classifition_input_dim_a = self.d_a
            if 'audio' in self.senti_modal_list:
                self.audio_model = FeatSentiFusionModel(modal='audio', with_senti=True, senti_feat_fusion_strategy=self.senti_feat_fusion_strategies['audio'], senti_attention_model = senti_attention_models['audio'])
                if self.senti_feat_fusion_strategies['audio'] == 'concat':
                    self.classifition_input_dim_a = self.d_a * 2
            else:
                self.audio_model = FeatSentiFusionModel(modal='audio', with_senti=False)

            if not self.early_fusion:
                #late fusion
                self.proj1_a = nn.Linear(self.classifition_input_dim_a, self.classifition_input_dim_a)
                self.proj2_a = nn.Linear(self.classifition_input_dim_a, self.classifition_input_dim_a)
                self.out_layer_a = nn.Linear(self.classifition_input_dim_a, output_dim)
        if 'video' in self.modal_list:
            self.d_v = MMSAModelConfig.output_d_v
            self.classifition_input_dim_v = self.d_v
            if 'video' in self.senti_modal_list:
                self.video_model = FeatSentiFusionModel(modal='video', with_senti=True, senti_feat_fusion_strategy=self.senti_feat_fusion_strategies['video'], senti_attention_model = senti_attention_models['video'])
                if self.senti_feat_fusion_strategies['video'] == 'concat':
                    self.classifition_input_dim_v = self.d_v * 2
            else:
                self.video_model = FeatSentiFusionModel(modal='video', with_senti=False)
            if not self.early_fusion:
                #late fusion
                self.proj1_v = nn.Linear(self.classifition_input_dim_v, self.classifition_input_dim_v)
                self.proj2_v = nn.Linear(self.classifition_input_dim_v, self.classifition_input_dim_v)
                self.out_layer_v = nn.Linear(self.classifition_input_dim_v, output_dim)
        if self.early_fusion:
            if self.early_fusion_strategy == 'concat':
                self.combined_dim = 0
                if 'text' in self.modal_list:
                    self.combined_dim += self.classifition_input_dim_t
                if 'audio' in self.modal_list:
                    self.combined_dim += self.classifition_input_dim_a
                if 'video' in self.modal_list:
                    self.combined_dim += self.classifition_input_dim_v
                self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
                self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
                self.out_layer = nn.Linear(self.combined_dim, output_dim)
            elif self.early_fusion_strategy == 'transformer':
                #如果要探究多模态之间的影响，就要像multiTransformer中那样，先探索其他模态对某一模态的影响，然后再把所有带影响的模态拼接再预测
                raise ValueError("to be fixed")
            else:
                raise ValueError("Unknown early fusion strategy type")
        else:
            if self.late_fusion_strategy == 'weightmean':
                self.late_fusion_weights = nn.Linear(len(self.modal_list),1)


    def forward(self, x_t=[], x_tl=[], x_ts = [], x_a=[],x_al=[], x_as = [], x_v=[], x_vl=[], x_vs = [],):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        if self.senti_model_finetune and len(self.senti_modal_list) != 0 :
            x_ts, x_as, x_vs = self.senti_model(x_t, x_tl, x_a, x_al, x_v, x_vl)
        concat_result_list = []
        if 'text' in self.modal_list:
            last_h_t = self.text_model(x_t,x_tl, x_ts)
            if 'text' not in self.senti_modal_list:
                last_h_t = last_h_t.squeeze(dim=0)
            if not self.early_fusion:
                last_ht_proj = self.proj2_t(F.dropout(F.relu(self.proj1_t(last_h_t)), p=self.out_dropout, training=self.training))
                last_ht_proj += last_h_t
                # A residual block
                output_t = F.softmax(self.out_layer_t(last_ht_proj), dim=1)
                concat_result_list.append(output_t)
            else:
                concat_result_list.append(last_h_t)
        if 'audio' in self.modal_list:
            last_h_a = self.audio_model(x_a, x_al, x_as)
            if 'audio' not in self.senti_modal_list:
                last_h_a = last_h_a.squeeze(dim=0)
            if not self.early_fusion:
                last_ha_proj = self.proj2_a(
                    F.dropout(F.relu(self.proj1_a(last_h_a)), p=self.out_dropout, training=self.training))
                last_ha_proj += last_h_a
                # A residual block
                output_a = F.softmax(self.out_layer_a(last_ha_proj), dim=1)
                concat_result_list.append(output_a)
            else:
                concat_result_list.append(last_h_a)
        if 'video' in self.modal_list:
            last_h_v = self.video_model(x_v, x_vl, x_vs)
            if 'video' not in self.senti_modal_list:
                last_h_v = last_h_v.squeeze(dim=0)
            if not self.early_fusion:
                last_hv_proj = self.proj2_v(
                    F.dropout(F.relu(self.proj1_v(last_h_v)), p=self.out_dropout, training=self.training))
                last_hv_proj += last_h_v
                # A residual block
                output_v = F.softmax(self.out_layer_v(last_hv_proj), dim=1)
                concat_result_list.append(output_v)
            else:
                concat_result_list.append(last_h_v)
        if self.early_fusion:
            if self.early_fusion_strategy == 'concat':
                last_h = torch.cat(concat_result_list, dim=1)
                last_h_proj = self.proj2(F.dropout(F.relu(self.proj1(last_h)), p=self.out_dropout, training=self.training))
                last_h_proj += last_h
                result = self.out_layer(last_h_proj)
                output = F.softmax(result, dim=1)
            elif self.early_fusion_strategy == 'transformer':
                # 如果要探究多模态之间的影响，就要像multiTransformer中那样，先探索其他模态对某一模态的影响，然后再把所有带影响的模态拼接再预测
                raise ValueError("to be fixed")
            else:
                raise ValueError("Unknown early fusion strategy type")
        else:
            output = torch.stack(concat_result_list, dim=0)
            if self.late_fusion_strategy == 'mean':
                output = torch.mean(output, dim=0) #late fusion用mean不收敛
            elif self.late_fusion_strategy == 'max':
                output = torch.max(output, dim=0).values
            elif self.late_fusion_strategy == 'weightmean':
                output = output.permute(1 ,2, 0)
                output = self.late_fusion_weights(output)
                output = output.permute(2,0,1)
                output = output.squeeze(dim=0)
        return output, concat_result_list


class FeatSentiFusionModel(nn.Module):
    def __init__(self, modal, with_senti=True, senti_feat_fusion_strategy='none', senti_attention_model = ''):
        """
        Construct a MulT model.
        """
        super(FeatSentiFusionModel, self).__init__()
        self.modal = modal
        self.with_senti = with_senti
        self.orig_extract_model = BaseExtractorModel(modal=modal)
        self.orig_d = 0
        self.attn_dropout = 0
        self.d = 0
        self.num_heads = MMSAModelConfig.num_heads
        self.layers = MMSAModelConfig.layers
        self.relu_dropout = MMSAModelConfig.relu_dropout
        self.res_dropout = MMSAModelConfig.res_dropout
        self.embed_dropout = MMSAModelConfig.embed_dropout
        self.attn_mask = MMSAModelConfig.attn_mask
        self.senti_feat_fusion_strategy = senti_feat_fusion_strategy
        self.senti_attention_model = senti_attention_model

        if modal == 'text':
            self.orig_d = MMSAModelConfig.orig_d_t
            # 因为feature 和 feature_senti的维度是一样的，所以只用一个org_d代之它们两个的维度
            self.d = MMSAModelConfig.output_d_t
            self.attn_dropout = MMSAModelConfig.attn_dropout_t
        elif modal == 'audio':
            self.orig_d = MMSAModelConfig.orig_d_a
            # 因为feature 和 feature_senti的维度是一样的，所以只用一个org_d代之它们两个的维度
            self.d = MMSAModelConfig.output_d_a
            self.attn_dropout = MMSAModelConfig.attn_dropout_a
        else:
            self.orig_d = MMSAModelConfig.orig_d_v
            # 因为feature 和 feature_senti的维度是一样的，所以只用一个org_d代之它们两个的维度
            self.d = MMSAModelConfig.output_d_v
            self.attn_dropout = MMSAModelConfig.attn_dropout_v
        # 1. Temporal convolutional layers
        self.proj = nn.Conv1d(self.orig_d, self.d, kernel_size=1, padding=0, bias=False)
        if with_senti:
            self.proj_s = nn.Conv1d(self.orig_d, self.d, kernel_size=1, padding=0, bias=False)
            if self.senti_attention_model == 'conv':
                self.attention_s_conv = nn.Conv1d(self.d, self.d, kernel_size=1, padding=0, bias=False)
                self.attention_s_pooling = nn.MaxPool1d(3, stride=1, padding=1)
            if self.senti_attention_model == 'linear':
                self.attention_s = nn.Sequential(nn.Linear(self.d, self.d), nn.ReLU(), nn.Dropout(p=self.attn_dropout))
            if self.senti_attention_model == 'transformer':
                self.attention_s = self.get_network(attention_type='senti_self_attention', layers=3)

            if self.senti_feat_fusion_strategy == 'fusion':
                # 2. Crossmodal Attentions :senti-feat-fusion
                self.trans_with_s = self.get_network(attention_type='cross_attention')
                # note:transformer的输入和输出是一样维度的，输入2*self.d（feat+feat_senti）输出也是2*self.d
            if self.senti_feat_fusion_strategy == 'concat':
                # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.):senti-feat-concat
                #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
                self.trans_mem = self.get_network(attention_type='self_attention', layers=3)

    def get_network(self, attention_type, layers=-1):
        embed_dim, attn_dropout = self.d, self.attn_dropout
        if attention_type == 'cross_attention':
            embed_dim = embed_dim
        elif attention_type == 'self_attention':
            embed_dim = 2 * embed_dim
            attn_dropout = MMSAModelConfig.attn_dropout_t
        elif attention_type == 'senti_self_attention':
            embed_dim = embed_dim
            attn_dropout = MMSAModelConfig.attn_dropout_t
        else:
            raise ValueError("Unknown attention type")

        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)

    def forward(self, x=[], xl=[], xs = []):
        x = self.orig_extract_model(x,xl)
        x = torch.unsqueeze(x, dim=1)
        xs = torch.unsqueeze(xs, dim=1)
        #text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        x = x.transpose(1, 2)
        if self.modal == 'text':
            x = F.dropout(x, p=self.embed_dropout, training=self.training)
        proj_x = x if self.orig_d == self.d else self.proj(x)
        proj_x = proj_x.permute(2, 0, 1)
        last_h = proj_x
        if self.with_senti:
            xs = xs.transpose(1, 2)
            if self.modal == 'text':
                xs = F.dropout(xs, p=self.embed_dropout, training=self.training)
            proj_xs = xs if self.orig_d == self.d else self.proj_s(xs)
            if self.senti_attention_model == 'conv':
                proj_xs = self.attention_s_conv(proj_xs)
                proj_xs = proj_xs.permute(2, 0, 1)
                proj_xs = self.attention_s_pooling(proj_xs)
            if self.senti_attention_model == 'linear' or self.senti_attention_model =='transformer':
                proj_xs = proj_xs.permute(2, 0, 1)
                proj_xs = self.attention_s(proj_xs)
            # print('senti feat:',proj_xs.shape, proj_xs[0][0])
            h = proj_xs
            if self.senti_feat_fusion_strategy == 'fusion':
                h = self.trans_with_s(proj_x, proj_xs, proj_xs)
            if self.senti_feat_fusion_strategy == 'add':
                h = proj_x + proj_xs
            if self.senti_feat_fusion_strategy == 'product':
                h = proj_x * proj_xs
            if self.senti_feat_fusion_strategy == 'concat':
                concat_h = torch.cat([proj_x, proj_xs], dim=2)
                h = self.trans_mem(concat_h)
            if type(h) == tuple:
                h = h[0]
            last_h = h[-1]
        return last_h


