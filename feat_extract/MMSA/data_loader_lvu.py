import logging
import pickle
import os
import numpy as np
import torch
import json
import random
from torch.utils.data import DataLoader, Dataset

__all__ = ['LVUDataLoader']

logger = logging.getLogger('MMSA')


class LVUDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        self.__init_lvu()

    def __init_lvu(self):
        self.dataset_name = self.args['dataset_name']
        self.train_mode = self.args['train_mode']
        self.annotation_dict = {
            'Negative': 0,
            'Neutral': 1,
            'Positive': 2
        }
        if 'mosi' in self.dataset_name:
            #mosi_lvu是指lvu格式的mosi数据集
            self.data_root = "data/MOSI"
            self.feat_root = "MMSA_feat/MOSI"
        else:
            self.data_root = "data/LVU"
            self.feat_root="MMSA_feat/LVU"
        modal_setting = json.load(open('feat_extract/MMSA/config/lvu_config.json','r'))
        video_tool = modal_setting[self.dataset_name]['video']
        audio_tool = modal_setting[self.dataset_name]['audio']
        text_tool = modal_setting[self.dataset_name]['text']
        self.video_feat_path = self.feat_root+'/'+self.mode+'/'+'video_' + video_tool + '_feats'
        self.audio_feat_path = self.feat_root+'/'+self.mode+'/'+'audio_' + audio_tool + '_feats'
        self.text_feat_path = self.feat_root+'/'+self.mode+'/'+'text_' + text_tool + '_feats'
        self.raw_text_path = self.data_root+'/'+self.mode+'/'+'subtitles_txt'
        self.info_file_path = self.data_root+'/' + self.mode + '/' + self.mode + '_dict.json'
        self.info_dict = json.load(open(self.info_file_path,'r'))
        self.ids = []
        self.raw_texts = []
        self.video_feats = []
        self.audio_feats = []
        self.text_feats = []
        self.labels = []
        self.annotations = []
        self.vision_lengths = []
        self.audio_lengths = []
        if 'use_bert' in self.args and self.args['use_bert']:
            self.text_key = 'text_bert'
        for file_name in os.listdir(self.video_feat_path):
            file_id = file_name.split('.')[0]
            self.ids.append(file_id)
            raw_text = open(self.raw_text_path+'/'+file_id+'.txt','r').read()
            self.raw_texts.append(raw_text)
            video_feat = pickle.load(open(self.video_feat_path+'/'+file_name,'rb'))['vision'].astype(np.float32)
            vision_length = pickle.load(open(self.video_feat_path+'/'+file_name,'rb'))['vision_lengths']
            self.video_feats.append(video_feat)
            self.vision_lengths.append(vision_length)
            #audio_feat = pickle.load(open(self.audio_feat_path + '/' + file_name, 'rb'))['audio'].astype(np.float32)
            audio_feat = np.array(pickle.load(open(self.audio_feat_path + '/' + file_name, 'rb'))['audio']).astype(np.float32)
            audio_length = pickle.load(open(self.audio_feat_path + '/' + file_name, 'rb'))['audio_lengths']
            self.audio_feats.append(audio_feat)
            self.audio_lengths.append(audio_length)
            #text_feat = pickle.load(open(self.text_feat_path + '/' + file_name, 'rb'))[self.text_key].astype(np.float32)
            text_feat = np.array(pickle.load(open(self.text_feat_path + '/' + file_name, 'rb'))[self.text_key]).astype(
                np.float32)
            self.text_feats.append(text_feat)
            file_info = self.info_dict[str(int(file_id))]
            if 'mosi' in self.dataset_name:
                if self.train_mode == 'regression':
                    self.labels.append(file_info['label'])
                else:
                    self.labels.append(self.annotation_dict[file_info['annotation']])
                self.annotations.append(file_info['annotation'])
            else:
                # self.labels.append(random.uniform(-3,3))
                self.labels.append(int(file_info['class_id']))
                self.annotations.append(file_info['class_id'])
        self.ids = np.array(self.ids)
        self.video_feats = np.array(self.video_feats)
        self.audio_feats = np.array(self.audio_feats)
        self.text_feats = np.array(self.text_feats)
        self.raw_texts = np.array(self.raw_texts)
        self.labels = np.array(self.labels).astype(np.float32)
        self.annotations = np.array(self.annotations)
        self.vision_lengths = np.array(self.vision_lengths)
        self.audio_lengths = np.array(self.audio_lengths)
        self.labels = {'M':self.labels}

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")
        # self.args['feature_dims'][0] = self.text_feats.shape[2]
        # self.args['feature_dims'][1] = self.audio_feats.shape[2]
        # self.args['feature_dims'][2] = self.video_feats.shape[2]
        # if 'use_bert' in self.args and self.args['use_bert']:
        #     self.args['feature_dims'][0] = self.text_feats.shape[1]
        if 'need_normalized' in self.args and self.args['need_normalized']:
            self.__normalize()

    def __truncate(self):
        # NOTE: truncate input to specific length.
        def do_truncate(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if ((instance[index] == padding).all()):
                        if (index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index + 20])
                            break
                    else:
                        truncated_feature.append(instance[index:index + 20])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature

        text_length, audio_length, video_length = self.args['seq_lens']
        self.vision = do_truncate(self.vision, video_length)
        self.text = do_truncate(self.text, text_length)
        self.audio = do_truncate(self.audio, audio_length)

    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.video_feats = np.transpose(self.video_feats, (1, 0, 2))
        self.audio_feats = np.transpose(self.audio_feats, (1, 0, 2))
        # For visual and audio modality, we average across time:
        # The original data has shape (max_len, num_examples, feature_dim)
        # After averaging they become (1, num_examples, feature_dim)
        self.video_feats = np.mean(self.video_feats, axis=0, keepdims=True)
        self.audio_feats = np.mean(self.audio_feats, axis=0, keepdims=True)

        # remove possible NaN values
        self.video_feats[self.video_feats!= self.video_feats] = 0
        self.audio_feats[self.audio_feats != self.audio_feats] = 0

        self.video_feats = np.transpose(self.video_feats, (1, 0, 2))
        self.audio_feats = np.transpose(self.audio_feats, (1, 0, 2))

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return (self.text_feats.shape[2], self.audio_feats.shape[1], self.video_feats.shape[1])
        else:
            return (self.text_feats.shape[1], self.audio_feats.shape[1], self.video_feats.shape[1])

    def get_feature_dim(self):
        if 'use_bert' in self.args and self.args['use_bert']:
            return self.text.shape[1], self.audio.shape[2], self.vision.shape[2]
        else:
            return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_texts[index],
            'text': torch.Tensor(self.text_feats[index]),
            'audio': torch.Tensor(self.audio_feats[index]),
            'vision': torch.Tensor(self.video_feats[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            'annotation': self.annotations[index],
        }
        if not self.args['need_data_aligned']:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]
        return sample


def LVUDataLoader(args, num_workers):
    datasets = {
        'train': LVUDataset(args, mode='train'),
        'val': LVUDataset(args, mode='val'),
        'test': LVUDataset(args, mode='test')
    }

    if 'seq_lens' in args:
        args['seq_lens'] = datasets['train'].get_seq_len()
    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }

    return dataLoader
