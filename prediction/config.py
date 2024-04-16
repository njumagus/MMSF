class MMSATrainConfig(object):
    # optional params sets
    # modal_list = ['video']
    # senti_modal_list = [] # using ['video'] for w/ sentiment
    # early_fusion = True # only using True when applying single modal
    # batch_size = 8
    # lr = 1e-5
    # weight_decay = 2e-5

    # modal_list = ['audio']
    # senti_modal_list = [] # # using ['audio'] for w/ sentiment
    # early_fusion = True # only using True when applying single modal
    # batch_size = 8
    # lr = 1e-5
    # weight_decay = 2e-5

    # modal_list = ['text']
    # senti_modal_list = [] # using ['text'] for w/ sentiment
    # early_fusion = True # only using True when applying single modal
    # batch_size = 8
    # lr = 1e-5
    # weight_decay = 2e-5

    # modal_list = ['video','audio']
    # senti_modal_list = [] # using ['video', 'audio'] for w/ sentiment
    # early_fusion = False # False for late fusion, True for early fusion
    # batch_size = 16
    # lr = 1e-5
    # weight_decay = 2e-5

    # modal_list = ['video','text']
    # senti_modal_list = [] # using ['video', 'text'] for w/ sentiment
    # early_fusion = False # False for late fusion, True for early fusion
    # batch_size = 16
    # lr = 1e-5
    # weight_decay = 2e-5

    # modal_list = ['audio','text']
    # senti_modal_list = ['audio', 'text'] # using ['audio', 'text'] for w/ sentiment
    # early_fusion = False # False for late fusion, True for early fusion
    # batch_size = 16
    # lr = 1e-5
    # weight_decay = 2e-5

    modal_list = ['video','audio','text']
    senti_modal_list = ['video','audio','text'] # using ['video','audio','text'] for w/ sentiment
    early_fusion = False # False for late fusion, True for early fusion
    batch_size = 32
    lr = 2e-3
    weight_decay = 2e-4

    senti_feat_fusion_strategies = {'video': 'fusion', 'audio': 'fusion', 'text': 'fusion'}

    senti_attention_models = {'video': 'conv', 'audio': 'conv', 'text': 'conv'}
    modal_tool_list = {'video': 'openface', 'audio': 'librosa', 'text': 'bert'}
    # load_model_path = 'checkpoints/model.pth'
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
    print_freq = 10  # print info every N batch
    save_freq = 10  # save model every N epoches
    max_epoch = 100
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
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
    # ids of videos that cannot be accessed when our paper was published: 125 and 345 in the training set, 119 and 193 in the validation set, 8 and 117 in the test set
    # ids of videos that will encounter errors in feature extraction: 780 and 756 in the training set, 151 in the test set
    # During the reproducibility process, more videos are inaccessible, the videos need to be passed are listed in comments
    # pass_videos = {'train':[125, 345, 780, 756], 'val':[119, 193], 'test':[8, 177, 151], } #{'train': [5, 7, 11, 15, 16, 21, 22, 23, 24, 26, 28, 29, 30, 35, 38, 40, 45, 49, 55, 56, 57, 59, 61, 62, 65, 68, 70, 72, 73, 74, 75, 78, 82, 87, 88, 89, 91, 92, 93, 94, 96, 97, 99, 105, 113, 115, 120, 122, 125, 126, 128, 130, 134, 139, 140, 144, 148, 158, 162, 163, 165, 166, 172, 173, 174, 175, 177, 178, 182, 185, 188, 194, 196, 197, 201, 204, 205, 206, 207, 208, 213, 216, 217, 219, 220, 224, 225, 229, 231, 237, 241, 242, 243, 244, 247, 248, 254, 255, 259, 261, 263, 268, 273, 282, 283, 285, 287, 288, 293, 295, 300, 303, 306, 308, 312, 318, 321, 323, 329, 330, 334, 335, 337, 338, 341, 344, 345, 351, 355, 356, 360, 361, 362, 363, 366, 368, 370, 373, 380, 383, 386, 387, 389, 392, 394, 397, 398, 401, 411, 414, 420, 421, 426, 427, 428, 432, 436, 439, 440, 441, 442, 444, 446, 451, 453, 460, 463, 464, 466, 471, 474, 477, 478, 481, 483, 484, 486, 488, 494, 495, 500, 502, 503, 505, 507, 510, 511, 512, 515, 516, 518, 522, 523, 524, 525, 526, 527, 535, 538, 543, 544, 545, 548, 549, 551, 552, 553, 554, 555, 556, 559, 560, 563, 565, 569, 571, 572, 573, 574, 575, 576, 578, 580, 581, 583, 585, 586, 587, 588, 590, 591, 592, 593, 596, 598, 602, 610, 611, 613, 614, 615, 616, 617, 619, 620, 622, 623, 624, 625, 626, 630, 632, 637, 638, 639, 642, 643, 644, 646, 647, 649, 650, 651, 652, 653, 654, 655, 656, 658, 661, 662, 663, 665, 666, 668, 669, 674, 676, 680, 685, 686, 687, 688, 692, 695, 702, 704, 705, 709, 711, 713, 716, 722, 725, 726, 741, 742, 745, 747, 750, 756, 759, 762, 764, 767, 768, 770, 771, 773, 774, 775, 777, 780, 783, 788, 789, 792, 800, 802, 803, 806, 807, 808, 811, 813, 821, 823, 829, 831, 836, 839, 842, 844, 846, 848, 849, 851, 852, 853, 854, 856, 858, 859, 860, 862, 863, 868, 871, 874, 879, 882, 885, 887, 892, 893, 896, 897, 899, 900, 901, 905, 907, 915, 918, 922, 924, 926, 927, 930, 931, 933, 936, 937, 938], 'val': [3, 9, 12, 14, 15, 16, 17, 19, 21, 24, 25, 28, 31, 34, 35, 37, 40, 44, 46, 47, 51, 54, 56, 58, 60, 61, 65, 67, 70, 73, 74, 77, 79, 80, 81, 82, 84, 85, 87, 88, 91, 93, 94, 95, 100, 102, 105, 106, 114, 116, 118, 119, 122, 125, 126, 127, 131, 133, 136, 139, 140, 142, 147, 149, 153, 154, 157, 160, 162, 164, 175, 178, 179, 185, 188, 190, 192, 193, 197, 201, 202, 204], 'test': [1, 3, 7, 8, 10, 11, 13, 14, 20, 26, 29, 30, 31, 49, 50, 51, 56, 58, 59, 61, 63, 64, 68, 70, 73, 76, 77, 82, 83, 87, 90, 92, 94, 95, 96, 104, 105, 108, 113, 114, 115, 117, 118, 124, 125, 127, 128, 129, 131, 136, 137, 138, 141, 142, 143, 144, 147, 148, 149, 150, 151, 157, 159, 161, 162, 164, 166, 174, 175, 176, 177, 178, 180, 185, 186, 188, 189, 191, 192, 199], }
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
    # load_model_path = 'checkpoints/model.pth'
    batch_size = 32  # batch size
    print_freq = 5  # print info every N batch
    save_freq = 10  # save model every N epoches
    max_epoch = 100
    lr = 0.001  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4
    #static params
    category_num = 5
    pass_videos = {'train':[125, 345, 780, 756], 'val':[119, 193], 'test':[8, 177 ,151], }
    save_checkpoint_dir = 'prediction/checkpoints/MultimodalNet_tmp/'
    save_prediction_dir = 'prediction/results/MultimodalNet_tmp/'
    data_root = 'data/LVU/'
    feat_root = 'MMSA_feat/LVU/'



