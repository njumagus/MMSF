from feat_extract.MMSA import MMSA_run, MMSA_run_test
#这个脚本主要是通过MMSA benchmark获得多模态情绪与训练模型然后跑出多模态的情绪特征
# 1.MMSA是一个拥有很多baseline的仓库，MMIM和SelfMM都是其中的可选项，它的环境是magus12上的一个docker环境MMSA-py38-cuda11
# 2.MMSA的主要的改动是加了一个data_loader_lvu用于加载lvu数据，然后在run中使用lvu_dataloader,在run的131行将结果进行保存，在utils/metricsTop中将acc和f1实时打印出来
# 3.它的启动程序在feat_extract/MMSA_senti_feat.py中,在这个脚本的main中指定在x数据集上训练/测试x模型，然后直接执行脚本即可


def train_mmim(dataset_name = 'lvu',train_mode = 'regression'):
    MMSA_run('mmim', dataset_name=dataset_name, seeds=[1111], gpu_ids=[0],
             model_save_dir='feat_extract/MMSA_'+train_mode+'_checkpoints',train_mode=train_mode)

def test_mmim(dataset_name='lvu', train_mode='regression'):
    MMSA_run_test('mmim', dataset_name=dataset_name, seeds=[1111], gpu_ids=[0],
                 model_save_dir='feat_extract/MMSA_' + train_mode + '_checkpoints', train_mode=train_mode, get_feats = False)

def train_selfmm(dataset_name = 'lvu',train_mode = 'regression'):
    MMSA_run('self_mm', dataset_name=dataset_name, seeds=[1111], gpu_ids=[0],
             model_save_dir='feat_extract/MMSA_'+train_mode+'_checkpoints',train_mode=train_mode)

def test_selfmm(dataset_name = 'lvu',train_mode = 'regression'):
    MMSA_run_test('self_mm', dataset_name=dataset_name, seeds=[1111], gpu_ids=[0],
                  model_save_dir='feat_extract/MMSA_'+train_mode+'_checkpoints', res_save_dir=  "MMSA_feat/LVU",train_mode=train_mode)



if __name__ == '__main__':
    # train_selfmm(dataset_name='mosi_lvu', train_mode='regression')
    test_selfmm(dataset_name='lvu',train_mode='regression')