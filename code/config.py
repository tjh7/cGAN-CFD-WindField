

class Config:
    IMAGE_SIZE = 256
    BATCH_SIZE = 5
    EPOCHS = 20000 
    LEARNING_RATE = 2e-4

    CONDITION_CSV = r'D:\gan\xin\taifeng_qiangduiliu\condition.csv'
    BUILDING_DIR = r'D:/gan/xin/buildings'
    WINDFIELD_DIR = r'D:\gan\xin\taifeng_qiangduiliu\cfd'

    CHECKPOINT_DIR = r'D:\gan\xin\taifeng_qiangduiliu\checkpoint_100'  
    SAMPLE_SAVE_PATH = r'D:/gan/xin/samples_MaxAE'  #
    LOSS_SAVE_PATH = r'D:\gan\xin\taifeng_qiangduiliu'

    L1_LAMBDA = 100                    
    SAVE_EVERY_N_EPOCH = 100 
