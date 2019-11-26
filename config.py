import torch

configurations = {
    1: dict(

        TRAIN_DATA = '/home/hgh/rubbish/data/Trainset', 
        TEST_DATA = '/home/hgh/rubbish/data/Test_B',
        
        MODEL_ROOT = '/home/hgh/rubbish/model', 
        BEST_MODEL_ROOT = '/home/hgh/rubbish/best_model',
        LOG_ROOT = '/home/hgh/rubbish/logs', 
        SUB_ROOT = '/home/hgh/rubbish/submission',     

        NUM_CLASS = 2,
        INPUT_SIZE =  (224,224),
        TRAIN_BATCH_SIZE = 32,
        TEST_BATCH_SIZE = 250,
        NUM_EPOCH = 20,
        DEVIVE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu'),
        RGB_MEAN = [0.485, 0.456, 0.406],
        RGB_STD =  [0.229, 0.224, 0.225],


        MODEL_NAME = 'se_resnext101_32x4d'#,'se_resnext101_32x4d', 'senet154'
        # METRIC_NAME = 'Softmax',# Softmax, Am_softmax, ArcFace

    )
    }
