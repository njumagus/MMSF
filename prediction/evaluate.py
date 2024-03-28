import json
from config import MMSATrainConfig
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

pass_videos = {'train': [125, 345, 780, 756], 'val': [119, 193], 'test': [8, 177, 151], }


data_root = '/media/magus/Data0/zhangbb_workspace/ICMR23/data/LVU/'
prediction_result_dir = 'results/'
modal_list = MMSATrainConfig.modal_list  # 'video', 'audio', 'text'
senti_modal_list = MMSATrainConfig.senti_modal_list
senti_feat_fusion_strategies = MMSATrainConfig.senti_feat_fusion_strategies
senti_feat_fusion_strategy_prefix = ''
for key in senti_feat_fusion_strategies:
    senti_feat_fusion_strategy_prefix += '_' + key + senti_feat_fusion_strategies[key]
test_epoch_num = 'val_best'
early_fusion = MMSATrainConfig.early_fusion
save_dir = 'MultimodalTransformer_'+str(MMSATrainConfig.video_senti_feat_dim)+'/MultimodalTransformer_senti'+senti_feat_fusion_strategy_prefix
if early_fusion:
    save_dir += '_early' + MMSATrainConfig.early_fusion_strategy
else:
    save_dir += '_late' + MMSATrainConfig.late_fusion_strategy
save_dir+='_multiloss'
save_prediction_dir = MMSATrainConfig.save_prediction_root + save_dir+'/'
if test_epoch_num == 'val_best':
    pred_path = save_prediction_dir + "_".join(modal_list) + "_senti_" + '_'.join(senti_modal_list) + '_val_best.json'
else:
    pred_path = save_prediction_dir + "_".join(modal_list) + "_senti_" + '_'.join(senti_modal_list) + '_' + str(
        test_epoch_num).zfill(6) + '.json'

pred_path = '/media/magus/Data0/zhangbb_workspace/ICMR23/code/prediction/results/MultimodalTransformer_300/MultimodalTransformer_sentifusion_lateweightmean_multiloss/video_audio_text_senti__val_best.json'
predict_dict = json.load(open(pred_path,'r'))
gt_dict = json.load(open(data_root + 'test/test_dict.json', 'r'))

def get_top_k_preds(k=1):
    result = {}
    for key in predict_dict:
        scores = predict_dict[key]
        score_dict = {'0':scores[0], '1':scores[1], '2':scores[2], '3':scores[3], '4':scores[4]}
        sorted_scores = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)[:k]
        result[key] = [item[0] for item in sorted_scores]
    return result



if __name__ == '__main__':
    pred_list = []
    gt_list = []
    pred_3_num = 0
    print(pred_path)

    preds = get_top_k_preds(k=1)
    for key in gt_dict:
        if int(key) in pass_videos['test']:
            continue
        if int(preds[key][0]) == 3:
            pred_3_num += 1
        pred_list.append(int(preds[key][0]))
        gt_list.append(int(gt_dict[key]['class_id']))
    # preds = json.load(open('/media/magus/Data0/zhangbb_workspace/ICMR23/code/lvu/outputs/20221202_163307_way_speaking/test_results.json', 'r'))
    # gt_dict = json.load(open('/media/magus/Data0/zhangbb_workspace/ICMR23/data/LVU/data_dict.json', 'r'))['video_info']['test']
    # for key in gt_dict:
    #     youtube_name = gt_dict[key]['youtube_id']
    #     if youtube_name in preds:
    #         pred_list.append(preds[youtube_name])
    #         if preds[youtube_name] == 3:
    #             pred_3_num += 1
    #         gt_list.append(int(gt_dict[key]['class_id']))
    # print('sample num:', str(len(gt_list)))


    # acc = cal_acc()
    acc = accuracy_score(gt_list, pred_list)
    f1 = f1_score(gt_list, pred_list, average='weighted')
    # acc, precision, recall, f1 = cal_fusion_matrix(gt_list=gt_list, pred_list=pred_list)
    print('pred 3 num:', pred_3_num)
    print('Acc: ' + str(acc)) #47.5
    # print('Precision: ' + str(precision))
    # print('Recall: ' + str(recall)) #
    print('F1 score: ' + str(f1)) #44.1
