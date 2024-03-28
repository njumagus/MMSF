import parser

from MSA_FET import FeatureExtractionTool
from data.config import pass_videos
import os
import pandas as pd
import json
import numpy as np
import pickle
import argparse

def generate_MMSA_label_csv():
    data_root = 'data/LVU'
    feat_root = 'MMSA_feat/LVU'
    csv_columes = ['video_id', 'clip_id', 'text', 'label', 'label_T', 'label_A', 'label_V', 'annotation', 'mode']
    csv_lines = []
    total_dict = {}
    total_dict['train'] = json.load(open(os.path.join(data_root,'train','train_dict.json')))
    total_dict['val'] = json.load(open(os.path.join(data_root,'val','val_dict.json')))
    total_dict['test'] = json.load(open(os.path.join(data_root,'test','test_dict.json')))

    os.makedirs(feat_root)

    for mode in total_dict:
        videos_to_pass = pass_videos['lvu'][mode]
        for key in total_dict[mode]:
            if int(key) in videos_to_pass:
                continue
            video_id = total_dict[mode][key]['video_id']
            speaking_style_cate = total_dict[mode][key]['class_id']
            path = os.path.join(data_root,mode,'subtitles_txt',str(video_id).zfill(4)+'.txt')
            if not os.path.isfile(path):
                continue
            text = open(path, 'r').readline()
            csv_lines.append({'video_id':mode, 'clip_id':str(video_id).zfill(4), 'text':text, 'label':0, 'label_T':0,'label_A':0,'label_V':0, 'annotation': speaking_style_cate, 'mode':mode})
    df = pd.DataFrame(csv_lines, columns=csv_columes)
    df.to_csv(os.path.join(feat_root,'label.csv'), index=False)


def padding(feature, MAX_LEN, value='zero', location='end'):
        """
        Parameters:
            mode:
                zero: padding with 0
                norm: padding with normal distribution
            location: start / end
        """
        assert value in ['zero', 'norm'], "Padding value must be 'zero' or 'norm'"
        assert location in ['start', 'end'], "Padding location must be 'start' or 'end'"

        length = feature.shape[0]
        if length >= MAX_LEN:
            return feature[:MAX_LEN, :]

        if value == "zero":
            pad = np.zeros([MAX_LEN - length, feature.shape[-1]])
        elif value == "normal":
            mean, std = feature.mean(), feature.std()
            pad = np.random.normal(mean, std, (MAX_LEN - length, feature.shape[1]))

        feature = np.concatenate((pad, feature), axis=0) if (location == "start") else \
            np.concatenate((feature, pad), axis=0)
        return feature

def paddingSequence(sequences, value='zero', location='end', is_bert=False):
        """
        Pad features to the same length according to the mean length of the features.
        """
        feature_dim = sequences[0].shape[-1]
        lengths = [s.shape[0] for s in sequences]
        # use (mean + 3 * std) as the max length
        final_length = int(np.mean(lengths) + 3 * np.std(lengths))
        if is_bert and final_length > 512:
            final_length = 512
        print('final_length:',final_length, ' mean_length:', int(np.mean(lengths)), ' std_lenght:', int(np.std(lengths)))
        final_sequence = np.zeros([len(sequences), final_length, feature_dim])
        print('fianl feat shape:', final_sequence.shape)
        for i, s in enumerate(sequences):
            if len(s) != 0:
                final_sequence[i] = padding(s, final_length, value, location)
        return final_sequence, final_length


def generate_MMSA_feat(modal, modal_tool, modes = ['train', 'val', 'test']):
    data_root = 'data/LVU'
    feat_root = 'MMSA_feat/LVU'
    feat_config_dir = os.path.join("feat_extract", "MMSA_feat_configs")

    file_info = {}
    file_total_cnt = 0
    for mode in modes:
        video_file_dir = os.path.join(data_root,mode,'videos')
        text_file_dir = os.path.join(data_root,mode,'subtitles_txt')
        feat_org_dir = os.path.join(feat_root, mode, modal + '_' + modal_tool + '_org_feats')
        if not os.path.exists(feat_org_dir):
            os.makedirs(feat_org_dir)

        for file_name in os.listdir(text_file_dir):
            file_id = file_name.split('.')[0]
            video_file_path = os.path.join(video_file_dir,file_id+'.mp4')
            text_file_path = os.path.join(text_file_dir,file_id + '.txt')
            file_info[mode+'_'+file_id]={'video_file_path':video_file_path, 'text_file_path':text_file_path}
            file_total_cnt += 1
    fet = FeatureExtractionTool(os.path.join(feat_config_dir,modal+'_'+ modal_tool + ".json"))
    processed_cnt = 0
    for key in file_info:
        mode = key.split('_')[0]
        file_id = key.split('_')[1]
        feat_org_dir=os.path.join(feat_root, mode, modal + '_' + modal_tool + '_org_feats')
        if int(file_id) in pass_videos['lvu'][mode]:
            continue
        video_file = file_info[key]['video_file_path']
        if not os.path.exists(video_file):
            continue
        text_file = file_info[key]['text_file_path']
        if not os.path.exists(text_file):
            continue
        out_file_path = os.path.join(feat_org_dir,file_id + '.pkl')
        if os.path.exists(out_file_path):
            processed_cnt += 1
            print(mode + ' '+file_id + ' process:' + str(processed_cnt) + '/' + str(file_total_cnt))
            continue
        print("fet.run_single", video_file, text_file)
        res = fet.run_single(in_file=video_file,text_file=text_file)
        processed_cnt += 1
        print(mode + ' '+file_id + ' process:'+str(processed_cnt)+'/'+str(file_total_cnt))
        with open(out_file_path, 'wb') as f:
            pickle.dump(res, f)


def padding_MSA_feats(modal, modal_tool):
    feat_root = 'MMSA_feat/LVU'
    file_keys = []
    feats = []
    text_bert_feats = []
    feat_lengths = []
    #get original feats
    if modal == 'video':
        modal_key = 'vision'
    else:
        modal_key = modal
    for mode in ['train','val', 'test']:
        data_dir = os.path.join(feat_root,mode,modal + '_'+modal_tool+'_org_feats')
        for file in os.listdir(data_dir):
            file_id = file.split('.')[0]
            file_key = mode + '_' + file_id
            file_keys.append(file_key)
            feat = pickle.load(open(data_dir + '/' + file, 'rb'))[modal_key]
            feat_length = pickle.load(open(data_dir+'/'+file,'rb'))[modal_key+'_lengths']
            feats.append(feat)
            feat_lengths.append(feat_length)
            if modal == 'text' and modal_tool == 'bert':
                text_bert_feat = pickle.load(open(data_dir + '/' + file, 'rb'))['text_bert']
                text_bert_feats.append(text_bert_feat)
    # padding features
    print(modal)
    if modal == 'text' and modal_tool == 'bert':
        feats, final_length = paddingSequence(feats, is_bert=True)
        print('text bert')
        text_bert_feats, text_bert_final_lengh = paddingSequence(text_bert_feats, is_bert=True)
        text_bert_feats = text_bert_feats.transpose(0, 2, 1)
    else:
        feats, final_length = paddingSequence(feats)
    for i, length in enumerate(feat_lengths):
        if length > final_length:
            feat_lengths[i] = final_length

    #save padding feats
    for i, file_key in enumerate(file_keys):
        result = {}
        mode = file_key.split('_')[0]
        save_dir = os.path.join(feat_root,mode,modal + '_' + modal_tool + '_feats')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_id = file_key.split('_')[1]
        result[modal_key] = feats[i]
        result[modal_key + '_lengths'] = feat_lengths[i]
        if modal == 'text'and modal_tool == 'bert':
            result['text_bert'] = text_bert_feats[i]
        out_file_path = save_dir +'/' + file_id + '.pkl'
        with open(out_file_path, 'wb') as f:
            pickle.dump(result, f)

parser=argparse.ArgumentParser(description="main script")
parser.add_argument("--label",action="store_true")
parser.add_argument("--audio",action="store_true")
parser.add_argument("--video",action="store_true")
parser.add_argument("--text",action="store_true")
args=parser.parse_args()
if args.label:
    generate_MMSA_label_csv()
elif args.audio:
    generate_MMSA_feat(modal='audio', modal_tool='librosa')
    padding_MSA_feats(modal='audio', modal_tool='librosa')
elif args.video:
    generate_MMSA_feat(modal='video', modal_tool='openface')
    padding_MSA_feats(modal='video', modal_tool='openface')
elif args.text:
    generate_MMSA_feat(modal='text', modal_tool='bert')
    padding_MSA_feats(modal='text', modal_tool='bert')


