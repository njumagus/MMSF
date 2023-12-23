import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from prediction.config import MMSATrainConfig
import pickle


class MMSADataset(Dataset):
    def __init__(self, mode='', modal_list=[], modal_tool_list=[], senti_modal_list=[], senti_modal_extractor='',data_root='', feat_root='', pass_videos={}):
        data_root = data_root + mode
        feat_root = feat_root + mode
        self.mode = mode
        self.modal_list = modal_list
        self.modal_tool_list = modal_tool_list
        self.senti_modal_list = senti_modal_list
        self.senti_model_finetune = MMSATrainConfig.senti_model_finetune
        self.video_feat_path = feat_root + '/video_'+self.modal_tool_list['video']+'_feats'
        self.audio_feat_path = feat_root + '/audio_'+self.modal_tool_list['audio']+'_feats'
        # self.audio_feat_path = '/media/magus/Data0/zhangbb_workspace/ICMR23/data/LVU/'+mode+'/aud_feat'
        self.text_feat_path = feat_root + '/text_'+self.modal_tool_list['text']+'_feats'
        self.video_senti_feat_path = feat_root + '/' + senti_modal_extractor + '_multi_modal_senti_feats'#+str(MMSATrainConfig.video_senti_feat_dim)
        self.audio_senti_feat_path = feat_root + '/' + senti_modal_extractor + '_multi_modal_senti_feats' #+ str(MMSATrainConfig.audio_senti_feat_dim)
        self.text_senti_feat_path = feat_root + '/' + senti_modal_extractor + '_multi_modal_senti_feats' #+ str(MMSATrainConfig.text_senti_feat_dim)
        #gt
        self.video_dict = json.load(open(data_root + '/' + mode + '_dict.json', 'r'))
        self.video_ids = list(self.video_dict.keys())
        for item in pass_videos:
            self.video_ids.remove(str(item))
        print(mode, 'video num:', len(self.video_ids))

    def __len__(self):
        return len(self.video_ids)


    def __getitem__(self, index):
        # index：video_index
        video_id = self.video_ids[index]
        video_id = video_id
        video_category_id = int(self.video_dict[video_id]['class_id'])
        cate_id = video_category_id
        video_feature = []
        video_length = 0
        video_senti_feature = []
        text_feature = []
        text_length = 0
        text_senti_feature = []
        audio_feature = []
        audio_length = 0
        audio_senti_feature = []
        if 'video' in self.modal_list:
            video_f = pickle.load(open(os.path.join(self.video_feat_path, video_id.zfill(4) + ".pkl"),'rb'))
            video_feature = video_f['vision'].astype(np.float32)
            video_length = video_f['vision_lengths']
            if not self.senti_model_finetune and 'video' in self.senti_modal_list:
                video_senti_feature = pickle.load(open(os.path.join(self.video_senti_feat_path, video_id.zfill(4) + ".pkl"),'rb'))['vision'].astype(np.float32)
        if 'audio' in self.modal_list:
            audio_f = pickle.load(open(os.path.join(self.audio_feat_path, video_id.zfill(4) + ".pkl"),'rb'))
            audio_feature = audio_f['audio'].astype(np.float32)
            audio_length = audio_f['audio_lengths']
            # audio_feature = np.load(os.path.join(self.audio_feat_path, video_id.zfill(4) + ".npy")).astype(np.float32)
            # audio_feature = audio_feature[None,:]
            # audio_length = 1
            if not self.senti_model_finetune and 'audio' in self.senti_modal_list:
                audio_senti_feature = pickle.load(open(os.path.join(self.audio_senti_feat_path, video_id.zfill(4) + ".pkl"),'rb'))['audio'].astype(np.float32)
        if 'text' in self.modal_list:
            text_f =  pickle.load(open(os.path.join(self.text_feat_path, video_id.zfill(4) + ".pkl"),'rb'))
            text_length = text_f['text_lengths']
            text_feature = text_f['text'].astype(np.float32)
            # if self.modal_tool_list['text'] == 'bert':
            #     text_feature = text_f['text_bert'].astype(np.float32)
            if not self.senti_model_finetune and 'text' in self.senti_modal_list:
                text_senti_feature = pickle.load(open(os.path.join(self.text_senti_feat_path, video_id.zfill(4) + ".pkl"),'rb'))['text'].astype(np.float32)

        video_feature = torch.Tensor(video_feature)
        video_senti_feature = torch.Tensor(video_senti_feature)
        audio_feature = torch.Tensor(audio_feature)
        audio_senti_feature = torch.Tensor(audio_senti_feature)
        text_feature = torch.Tensor(text_feature)
        text_senti_feature = torch.Tensor(text_senti_feature)

        return video_id, cate_id, video_feature, video_length, video_senti_feature, audio_feature, audio_length, audio_senti_feature, text_feature,text_length, text_senti_feature

