class MMSATrainConfig(object):
    #optional params
    # modal_list = ['video']  # 'video', 'audio', 'text'
    # senti_modal_list = ['video']

    # modal_list = ['audio']  # 'video', 'audio', 'text'
    # senti_modal_list = ['audio']

    # modal_list = ['text']  # 'video', 'audio', 'text'
    # senti_modal_list = ['text']

    # modal_list = ['video','audio']  # 'video', 'audio', 'text'
    # senti_modal_list = ['video', 'audio']

    # modal_list = ['video','text']  # 'video', 'audio', 'text'
    # senti_modal_list = ['video', 'text']

    # modal_list = ['audio','text']  # 'video', 'audio', 'text'
    # senti_modal_list = ['audio', 'text']

    modal_list = ['video','audio','text']  # 'video', 'audio', 'text'
    senti_modal_list = ['video', 'audio', 'text']
    senti_feat_fusion_strategies = {'video': 'fusion', 'audio': 'fusion', 'text': 'fusion'}
    early_fusion = False

    modal_tool_list = {'video': 'openface', 'audio': 'librosa', 'text': 'bert'}
    # load_model_path = 'checkpoints/model.pth'  # 加载预训练的模型的路径，为None代表不加载
    senti_attention_models = {'video':'conv', 'audio':'conv', 'text':'conv'}
    #senti_feat_fusion_strategies = {'video':'product', 'audio':'product', 'text':'product'}
    senti_modal_extractor = 'self_mm'
    video_senti_feat_dim = 300
    audio_senti_feat_dim = 300
    text_senti_feat_dim = 300
    senti_model_finetune = False
    # senti_attention_model = 'transformer'
    early_fusion_strategy = 'concat'
    late_fusion_strategy = 'weightmean'

    seed_idx = 10
    batch_size = 32  # batch size 16
    print_freq = 10  # print info every N batch
    save_freq = 10  # save model every N epoches
    max_epoch = 100
    lr = 2e-3  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 2e-4
    when_to_decay_lr = 20

    # "batch_size": 16,
    # "learning_rate_bert": 5e-5,
    # "learning_rate_audio": 0.005,
    # "learning_rate_video": 0.005,
    # "learning_rate_other": 0.001,
    # "weight_decay_bert": 0.001,
    # "weight_decay_audio": 0.001,
    # "weight_decay_video": 0.001,
    # "weight_decay_other": 0.001,

    #static params
    category_num = 5
    pass_videos = {'train': [125, 345, 780, 756], 'val': [119, 193], 'test': [8, 177, 151], }
    data_root ='data/LVU/'
    feat_root = 'MMSA_feat/LVU/'
    save_checkpoint_root = 'prediction/checkpoints/'
    save_prediction_root = 'prediction/results/'

class MMSAModelConfig(object):
    orig_d_t = 300 #64
    orig_d_a = 300 #16
    orig_d_v = 300 #32
    output_d_t = 300 #30
    output_d_a = 300 #30
    output_d_v = 300 #30
    num_heads = 5
    layers = 5
    attn_dropout_t = 0.1
    attn_dropout_a = 0.0
    attn_dropout_v = 0.0
    relu_dropout = 0.1
    res_dropout = 0.1
    out_dropout = 0.0
    embed_dropout = 0.25
    attn_mask = True
    output_dim = 5

class BaseExtractorModelConfig(object):
    bert_finetune = False
    input_d_t = 768
    input_d_a = 33
    input_d_v = 709
    a_lstm_hidden_size = 300 #16
    v_lstm_hidden_size = 300 #64
    a_lstm_layers = 1
    v_lstm_layers = 1
    text_out = 768
    audio_out = 300 #16
    video_out = 300 #32
    a_lstm_dropout = 0.0
    v_lstm_dropout = 0.0
    t_bert_dropout = 0.1
    post_text_dim = 300 #64
    post_audio_dim = 300 #16
    post_video_dim = 300 #32
    post_text_dropout = 0.1
    post_audio_dropout = 0.1
    post_video_dropout = 0.0
    H = 1.0
    transformers = 'bert'
    pretrained = 'model/pretrained_models/bert-base-uncased'


class SentiAttentionTrainConfig(object):
    senti_modal = 'text'
    pass_videos = {'train': [125, 345, 780, 756], 'val': [119, 193], 'test': [8, 177, 151], }
    senti_modal_extractor = 'self_mm'
    seed_idx = 10
    lr = 1e-3  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    batch_size = 16
    print_freq = 10  # print info every N batch
    save_freq = 100  # save model every N epoches
    max_epoch = 1000
    category_num = 5
    when_to_decay_lr = 20
    data_root = 'data/LVU/'
    feat_root = 'MMSA_feat/LVU/'

    #mlp param
    modal_input_d_t = 300 #64
    modal_input_d_a = 300 #16
    modal_input_d_v = 300 #32
    model_dropout_t = 0.1
    model_dropout_a = 0.1
    model_dropout_v = 0.0

    #attention transformer
    num_heads = 5
    layers = 5
    res_dropout = 0.1
    relu_dropout = 0.1
    embed_dropout = 0.25
    attn_mask = True
    attn_dropout_t = 0.1
    attn_dropout_a = 0.0
    attn_dropout_v = 0.0

class SELFMMConfig(object):
    fintune = True
    bert_finetune = False
    transformers = 'bert'
    pretrained = 'bert-base-uncased'
    input_d_t = 768
    input_d_a = 33
    input_d_v = 709
    a_lstm_hidden_size = 300 #16
    v_lstm_hidden_size = 300 #64
    a_lstm_layers = 1
    v_lstm_layers = 1
    a_lstm_dropout = 0.0
    v_lstm_dropout = 0.0
    t_bert_dropout = 0.1
    text_out = 768
    audio_out = 300 #16
    video_out =300 # 32
    post_fusion_dim = 768
    post_text_dim = 300
    post_audio_dim = 300
    post_video_dim = 300
    post_fusion_dropout = 0.0
    post_text_dropout = 0.1
    post_audio_dropout = 0.1
    post_video_dropout = 0.0

class DefaultConfig(object):
    #optional params
    modal_list = ['text']  # 'video', 'entity', 'audio', 'text'
    # load_model_path = 'checkpoints/model.pth'  # 加载预训练的模型的路径，为None代表不加载
    batch_size = 32  # batch size
    print_freq = 5  # print info every N batch
    save_freq = 10  # save model every N epoches
    max_epoch = 100
    lr = 0.001  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数
    #static params
    category_num = 5
    pass_videos = {'train':[125, 345, 780, 756], 'val':[119, 193], 'test':[8, 177 ,151], }
    save_checkpoint_dir = 'prediction/checkpoints/MultimodalNet_tmp/'
    save_prediction_dir = 'prediction/results/MultimodalNet_tmp/'
    data_root = 'data/LVU/'
    feat_root = 'MMSA_feat/LVU/'



