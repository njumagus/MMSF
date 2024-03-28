from feat_extract.MMSA import MMSA_run, MMSA_run_test

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