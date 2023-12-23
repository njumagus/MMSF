import torch
from torch import nn
from prediction.dataset_MMSA import MMSADataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from prediction.model.model_MMSA import FeatSentiPredictionModel
from prediction.config import MMSATrainConfig
#export PYTHONPATH='xx/prediction'
import torch.optim as optim
from torchnet import meter
import numpy as np
import os
import json
import random
from sklearn.metrics import accuracy_score, f1_score
import argparse

class MMSF_Runner:
    def __init__(self): #,modal_list=None,senti_model_list=None,senti_feat_fusion_strategies=None,early_fusion=None):
        self.modal_list = MMSATrainConfig.modal_list  # 'video', 'audio', 'text'
        # if modal_list is not None:
        #     self.modal_list=modal_list
        self.senti_feat_fusion_strategies = MMSATrainConfig.senti_feat_fusion_strategies
        # if senti_feat_fusion_strategies is not None:
        #     self.senti_feat_fusion_strategies=senti_feat_fusion_strategies
        self.early_fusion = MMSATrainConfig.early_fusion
        # if early_fusion is not None:
        #     self.early_fusion=early_fusion

        self.modal_tool_list = MMSATrainConfig.modal_tool_list
        self.pass_videos_dict = MMSATrainConfig.pass_videos
        self.senti_modal_list = MMSATrainConfig.senti_modal_list  # 'video', 'entity', 'audio', 'text'
        self.senti_attention_models = MMSATrainConfig.senti_attention_models
        self.senti_attention_model_prefix = ''
        self.senti_feat_fusion_strategy_prefix = ''
        for key in self.senti_attention_models:
            self.senti_attention_model_prefix += '_' + key + self.senti_attention_models[key]
        for key in self.senti_feat_fusion_strategies:
            self.senti_feat_fusion_strategy_prefix += '_' + key + self.senti_feat_fusion_strategies[key]

        self.senti_modal_extractor = MMSATrainConfig.senti_modal_extractor
        self.senti_model_finetune = MMSATrainConfig.senti_model_finetune
        self.data_root = MMSATrainConfig.data_root
        self.feat_root = MMSATrainConfig.feat_root
        self.early_fusion_strategy = MMSATrainConfig.early_fusion_strategy
        self.late_fusion_strategy = MMSATrainConfig.late_fusion_strategy
        if self.senti_attention_models != {}:
            self.save_dir = 'MultimodalTransformer_withSentiAttention/with_senti'+self.senti_attention_model_prefix+'/'
        else:
            self.save_dir = 'MultimodalTransformer_withSentiAttention/without_senti/'
        # if with_senti_attention:
        #     save_dir = 'MultimodalTransformer_withSentiAttention'
        # else:
        #     save_dir = 'MultimodalTransformer_' + str(MMSATrainConfig.video_senti_feat_dim)
        self.save_dir += 'MultimodalTransformer_senti' + self.senti_feat_fusion_strategy_prefix
        if self.early_fusion:
            self.save_dir += '_early' + self.early_fusion_strategy
        else:
            self.save_dir += '_late' + self.late_fusion_strategy
        if not self.early_fusion and self.late_fusion_strategy == 'weightmean':
            self.save_dir+='_multiloss'
        self.save_checkpoint_dir = MMSATrainConfig.save_checkpoint_root + self.save_dir+'/'
        self.save_prediction_dir = MMSATrainConfig.save_prediction_root + self.save_dir+'/'

        if not os.path.exists(self.save_checkpoint_dir):
            os.makedirs(self.save_checkpoint_dir)
        if not os.path.exists(self.save_prediction_dir):
            os.makedirs(self.save_prediction_dir)
        self.lossfile = open(self.save_checkpoint_dir + "_".join(self.modal_list)+ "_senti_" +'_'.join(self.senti_modal_list) + "_lossfile.txt", 'a+')

    def my_collate_fn(self,batch):
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            print("No val data!!!")
            batch = [[torch.from_numpy(np.zeros([1, 1]))]]
        return default_collate(batch)

    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子

    def train(self):
        lr = MMSATrainConfig.lr
        batch_size = MMSATrainConfig.batch_size
        train_dataset = MMSADataset(mode='train', modal_list = self.modal_list, modal_tool_list=self.modal_tool_list, senti_modal_list = self.senti_modal_list, senti_modal_extractor = self.senti_modal_extractor, data_root=self.data_root, feat_root=self.feat_root, pass_videos = self.pass_videos_dict['train'])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=self.my_collate_fn, shuffle=True)
        val_dataset = MMSADataset(mode='val', modal_list = self.modal_list, modal_tool_list=self.modal_tool_list, senti_modal_list = self.senti_modal_list, senti_modal_extractor = self.senti_modal_extractor, data_root=self.data_root, feat_root=self.feat_root, pass_videos = self.pass_videos_dict['val'])
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=self.my_collate_fn, shuffle=True)
        test_dataset = MMSADataset(mode='test', modal_list=self.modal_list, modal_tool_list=self.modal_tool_list,
                                  senti_modal_list=self.senti_modal_list, senti_modal_extractor=self.senti_modal_extractor,
                                  data_root=self.data_root, feat_root=self.feat_root, pass_videos=self.pass_videos_dict['test'])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.my_collate_fn, shuffle=True)

        model = FeatSentiPredictionModel(modal_list=self.modal_list, senti_modal_list=self.senti_modal_list, senti_feat_fusion_strategies = self.senti_feat_fusion_strategies, early_fusion=self.early_fusion, early_fusion_strategy = self.early_fusion_strategy, late_fusion_strategy = self.late_fusion_strategy, senti_model_finetune = self.senti_model_finetune, senti_attention_models=self.senti_attention_models).cuda()
        model.train()
        for name, param in model.named_parameters():
            print('param:', name)
            self.lossfile.write('param:'+ name + '\n')
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([{'params': [param for name, param in model.named_parameters()]}, ], lr=lr)  # , weight_decay=0.0000001, momentum=0.9
        optimizer.zero_grad()
        # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
        loss_meter = meter.AverageValueMeter() #计算list的平均值和方差
        confusion_matrix = meter.ConfusionMeter(k=MMSATrainConfig.category_num)
        previous_loss = 1000
        best_val_acc = 0
        best_val_test_acc = 0
        best_val_epoch_num = 0

        best_test_acc = 0
        best_test_f1 = 0
        best_test_epoch_num = 0
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=MMSATrainConfig.when_to_decay_lr, factor=0.1, verbose=True)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15, 30], gamma=0.1) #26.3
        for epoch in range(MMSATrainConfig.max_epoch):
            loss_meter.reset()
            confusion_matrix.reset()
            for i, (video_id, cate_id, video_feature,video_length, video_senti_feature, audio_feature,audio_length, audio_senti_feature, text_feature, text_length, text_senti_feature) in enumerate(train_dataloader):
                cate_id = cate_id.cuda()
                video_feature = video_feature.cuda()
                video_senti_feature = video_senti_feature.cuda()
                audio_feature = audio_feature.cuda()
                audio_senti_feature = audio_senti_feature.cuda()
                text_feature = text_feature.cuda()
                text_senti_feature = text_senti_feature.cuda()
                output, modal_output_list = model(x_t = text_feature, x_tl = text_length, x_ts = text_senti_feature, x_a = audio_feature, x_al = audio_length, x_as = audio_senti_feature, x_v = video_feature, x_vl = video_length, x_vs = video_senti_feature)
                loss = 0
                loss += criterion(output, cate_id)
                if not self.early_fusion and len(self.modal_list)>1:
                    for modal_output in modal_output_list:
                        loss += criterion(modal_output, cate_id)
                # 更新统计指标以及可视化
                loss_meter.add(loss.data.cpu())
                confusion_matrix.add(output.data, cate_id.data)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if i % MMSATrainConfig.print_freq == ( MMSATrainConfig.print_freq - 1 ):
                    print("Epoch: " + str(epoch) + " Batch: "+str(i) + " loss: "+str(loss_meter.value()[0]))
                    self.lossfile.write("Epoch: " + str(epoch) + " Batch: "+str(i) + " loss: "+str(loss_meter.value()[0])+'\n')
            #*****val
            model.eval()
            val_cm, val_accuracy, val_top1_acc,val_f1, val_loss = self.val('val',model, val_dataloader)
            test_cm, test_accuracy, test_top1_acc,test_f1, test_loss = self.val('test',model, test_dataloader)
            model.train()
            scheduler.step(val_loss)
            #scheduler.step()
            print("===Epoch: " + str(epoch) + ' val loss: '+ str(val_loss)  + ' val_top1_acc: ' + str(val_top1_acc)+ ' val_f1: ' + str(val_f1))
            print("===Epoch: " + str(epoch) + ' test loss: '+ str(test_loss)  + ' test_top1_acc: ' + str(test_top1_acc)+ ' test_f1: ' + str(test_f1))
            self.lossfile.write("===Epoch: " + str(epoch) + ' val loss: '+ str(val_loss) + ' val_top1_acc: ' + str(val_top1_acc)+'\n')
            self.lossfile.write("===Epoch: " + str(epoch) + ' test loss: '+ str(test_loss)  + ' test_top1_acc: ' + str(test_top1_acc)+ ' test_f1: ' + str(test_f1)+'\n')
            if val_top1_acc > best_val_acc:
                # 根据验证集的结果保存最好的模型
                best_val_acc = val_top1_acc
                best_val_test_acc = test_top1_acc
                best_val_epoch_num = epoch
                torch.save(model.state_dict(),
                           self.save_checkpoint_dir + "_".join(self.modal_list) + "_senti_" + '_'.join(self.senti_modal_list) + '_val_best.pt')
            if test_top1_acc > best_test_acc:
                best_test_acc = test_top1_acc
                best_test_f1 = test_f1
                best_test_epoch_num = epoch
                # torch.save(model.state_dict(), save_checkpoint_dir + "_".join(modal_list) + "_senti_" + '_'.join(senti_modal_list) + '_test_best.pt')
            torch.save(model.state_dict(), self.save_checkpoint_dir + "_".join(self.modal_list) + "_senti_" + '_'.join(self.senti_modal_list) + '_last.pt')
            print('===best val epoch num:'+ str(best_val_epoch_num)+ ' best val acc:'+ str(best_val_acc)+ ' best val test acc:'+ str(best_val_test_acc))
            print('===best test epoch num:'+ str(best_test_epoch_num)+ ' best test acc:'+ str(best_test_acc)+ ' best test f1:'+ str(best_test_f1))
            # print(save_checkpoint_dir + "_".join(modal_list) + "_senti_" + '_'.join(senti_modal_list)+'_'+str(epoch).zfill(6))
            self.lossfile.write('===best val epoch num:' + str(best_val_epoch_num) + ' best val acc:' + str(best_val_acc) + ' best val test acc:' + str(best_val_test_acc)+'\n')
            self.lossfile.write('===best test epoch num:' + str(best_test_epoch_num) + ' best test acc:' + str(best_test_acc) + ' best test f1:' + str(best_test_f1)+'\n')

            if epoch % MMSATrainConfig.save_freq == (MMSATrainConfig.save_freq - 1 ):
                # torch.save(model.state_dict(),save_checkpoint_dir + "_".join(modal_list) + "_senti_" + '_'.join(senti_modal_list)+'_'+str(epoch).zfill(6)+'.pt')
                print(self.save_checkpoint_dir + "_".join(self.modal_list) + "_senti_" + '_'.join(self.senti_modal_list)+'_'+str(epoch).zfill(6)+'.pt')
                self.lossfile.write(self.save_checkpoint_dir + "_".join(self.modal_list) + "_senti_" + '_'.join(self.senti_modal_list)+'_'+str(epoch).zfill(6)+'.pt\n')
            # 如果损失不再下降，则降低学习率
            if loss_meter.value()[0] > previous_loss:
                lr = lr * MMSATrainConfig.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            previous_loss = loss_meter.value()[0]
        self.lossfile.close()



    def val(self,mode, model, val_dataloader):

        criterion = nn.CrossEntropyLoss().cuda()
        loss_meter = meter.AverageValueMeter() #计算list的平均值和方差
        confusion_matrix = meter.ConfusionMeter(MMSATrainConfig.category_num)
        result = {}
        for i, (video_id, cate_id, video_feature,video_length, video_senti_feature, audio_feature,audio_length, audio_senti_feature, text_feature, text_length, text_senti_feature) in enumerate(val_dataloader):
            cate_id = cate_id.cuda()
            video_feature = video_feature.cuda()
            video_senti_feature = video_senti_feature.cuda()
            audio_feature = audio_feature.cuda()
            audio_senti_feature = audio_senti_feature.cuda()
            text_feature = text_feature.cuda()
            text_senti_feature = text_senti_feature.cuda()
            with torch.no_grad():
                output, modal_output_list =  model(x_t = text_feature, x_tl = text_length, x_ts = text_senti_feature, x_a = audio_feature, x_al = audio_length, x_as = audio_senti_feature, x_v = video_feature, x_vl = video_length, x_vs = video_senti_feature)
            loss = 0
            loss += criterion(output, cate_id)
            if not self.early_fusion and len(self.modal_list) > 1:
                for modal_output in modal_output_list:
                    loss += criterion(modal_output, cate_id)
            # 更新统计指标以及可视化
            loss_meter.add(loss.data.cpu())
            confusion_matrix.add(output.data, cate_id.data)
            for j, video in enumerate(video_id):
                result[str(video)] = output.data.cpu().numpy().tolist()[j]
        cm_value = confusion_matrix.value()
        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / \
                       (cm_value.sum())
        top1_acc, f1 = self.cal_metric(mode, json.load(open(MMSATrainConfig.data_root + mode + '/'+mode+'_dict.json','r')), result)
        loss = loss_meter.value()[0]

        return confusion_matrix, accuracy, top1_acc, f1,  loss


    def test(self,epoch_num='val_best'):
        batch_size = MMSATrainConfig.batch_size
        test_dataset = MMSADataset(mode='test', modal_list = self.modal_list, modal_tool_list=self.modal_tool_list, senti_modal_list = self.senti_modal_list, senti_modal_extractor = self.senti_modal_extractor, data_root=self.data_root, feat_root=self.feat_root, pass_videos = self.pass_videos_dict['test'])
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=self.my_collate_fn)
        model = FeatSentiPredictionModel(modal_list=self.modal_list, senti_modal_list=self.senti_modal_list, senti_feat_fusion_strategies = self.senti_feat_fusion_strategies, early_fusion=self.early_fusion, early_fusion_strategy = self.early_fusion_strategy, late_fusion_strategy = self.late_fusion_strategy, senti_model_finetune = self.senti_model_finetune, senti_attention_models=self.senti_attention_models).cuda()
        model.eval()
        if epoch_num == 'val_best':
            model_path = self.save_checkpoint_dir + "_".join(self.modal_list) + "_senti_" + '_'.join(self.senti_modal_list)+ '_val_best.pt'
        else:
            model_path = self.save_checkpoint_dir + "_".join(self.modal_list)+ "_senti_" + '_'.join(self.senti_modal_list) + '_'+str(epoch_num).zfill(6)+'.pt'
        print(model_path)
        model.load_state_dict(torch.load(model_path))
        result = {}
        for i, (video_id, cate_id, video_feature,video_length, video_senti_feature, audio_feature,audio_length, audio_senti_feature, text_feature, text_length, text_senti_feature) in enumerate(test_dataloader):
            # if i==1:
            #     for name, param in model.named_parameters():
            #         print('here:', name,param)
            #     break
            cate_id = cate_id.cuda()
            video_feature = video_feature.cuda()
            video_senti_feature = video_senti_feature.cuda()
            audio_feature = audio_feature.cuda()
            audio_senti_feature = audio_senti_feature.cuda()
            text_feature = text_feature.cuda()
            text_senti_feature = text_senti_feature.cuda()
            with torch.no_grad():
                output, modal_output_list = model(x_t = text_feature, x_tl = text_length, x_ts = text_senti_feature, x_a = audio_feature, x_al = audio_length, x_as = audio_senti_feature, x_v = video_feature, x_vl = video_length, x_vs = video_senti_feature)
            for j, video in enumerate(video_id):
                result[str(video)] = output.data.cpu().numpy().tolist()[j]
        if epoch_num == 'val_best':
            json.dump(result,open(self.save_prediction_dir + "_".join(self.modal_list)+ "_senti_" + '_'.join(self.senti_modal_list) + '_val_best.json', 'w'))
        else:
            json.dump(result, open(self.save_prediction_dir+"_".join(self.modal_list)+ "_senti_" + '_'.join(self.senti_modal_list)+'_'+str(epoch_num).zfill(6)+'.json','w'))

    def cal_metric(self, mode, gt_dict, predict_dict):
        pred_list = []
        gt_list = []
        for key in predict_dict:
            predict_scores = predict_dict[key]
            predict_cate_id = predict_scores.index(max(predict_scores))
            pred_list.append(predict_cate_id)
            gt_list.append(int(gt_dict[key]['class_id']))
        acc = accuracy_score(gt_list, pred_list)
        f1 = f1_score(gt_list, pred_list, average='weighted')

        return acc, f1

    def evaluate(self, mode, test_epoch_num='val_best'):
        #top1-accuracy
        gt = json.load(open(self.data_root + mode + '/'+mode+'_dict.json','r'))
        if test_epoch_num == 'val_best':
            pred_path = self.save_prediction_dir + "_".join(self.modal_list)+ "_senti_" + '_'.join(self.senti_modal_list) + '_val_best.json'
        else:
            pred_path = self.save_prediction_dir+"_".join(self.modal_list)+ "_senti_" + '_'.join(self.senti_modal_list)+'_'+str(test_epoch_num).zfill(6)+'.json'
        print(pred_path)
        self.lossfile.write(str(pred_path)+'\n')
        predict_result = json.load(open(pred_path,'r'))
        acc, f1 = self.cal_metric(mode, gt_dict=gt, predict_dict=predict_result)
        print('top1 accuracy: '+str(acc)+' f1 score:'+str(f1))
        self.lossfile.write('top1 accuracy: '+str(acc)+ ' f1 score:'+str(f1)+'\n')
        return acc, f1

if __name__ == '__main__':
    # # ======train========
    #setup_seed(MMSATrainConfig.seed_idx)
    parser=argparse.ArgumentParser(description="main script")
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--test", action="store_true")
    args=parser.parse_args()
    runner=MMSF_Runner()
    if args.train:
        runner.train()
    if args.test:
        runner.test(epoch_num='val_best')
        runner.evaluate(mode='test')


