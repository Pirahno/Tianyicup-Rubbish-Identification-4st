import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os, collections, time
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from transforms import RandomErasing, SelectROI
from tensorboardX import SummaryWriter
from config import configurations
from utils import *
from dataloader import rubbishDataSet
import warnings

import pandas as pd
warnings.filterwarnings('ignore')

# todu:
# done : k-fold
# 鏂偣璁粌
# done:tensorboardx
# done:pretraindmodels


# train val k fold
def kfold_trainval(data_root = None, n_splits = 5):
    dd_list = [i for i in os.listdir(data_root) if i.split('.')[1] == 'jpg']
    k_train, k_val = [], []
    kf = KFold(n_splits = n_splits, shuffle=True, random_state=1)
    for kfold, (train_data, val_data) in enumerate(kf.split(dd_list), start=1):
        k_train.append([dd_list[i] for i in train_data])
        k_val.append([dd_list[i] for i in val_data])
    return k_train, k_val

# train
def train(kfold_index = 0, model = None, k_train = None, k_val = None, cfg = None, data_transforms = None, writer = None):

    # criterion = LabelSmoothSoftmaxCE(lb_pos = 0.95, lb_neg = 0.05)
    criterion = nn.CrossEntropyLoss()



    model = model.to(cfg['DEVIVE'])

    # update feature
    model_params = update_params(model)

    # layer4 set low lr, fc set high lr
    layer4 = list(map(id, model.layer4.parameters()))
    fc_params = filter(lambda x:id(x) not in layer4, model_params )
    optimizer_ft = optim.Adam([{'params':list(model.layer4.parameters()), 'lr':0.005},
                     {'params':fc_params}], 0.001, weight_decay=1e-6, amsgrad=True)

    
    # optimizer_ft = optim.Adam(model_params, lr=0.001)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=5, factor = 0.1, verbose=True,mode="max")
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, 10)


    loss_hist = collections.deque(maxlen = 500)
    max_auc = 0.0

    for epoch in range(cfg['NUM_EPOCH']): # start training process        
        # train stage
        train = rubbishDataSet(file_path = cfg['TRAIN_DATA'], 
                                kfold_data = k_train[kfold_index],
                                transform = data_transforms['train'])
        val = rubbishDataSet(file_path = cfg['TRAIN_DATA'], 
                                kfold_data = k_val[kfold_index],
                                transform = data_transforms['val'])

        train_loader = DataLoader(dataset=train,
                                batch_size=cfg['TRAIN_BATCH_SIZE'],
                                shuffle=True,
                                num_workers=4,
                                pin_memory = False
                                )

        val_loader = DataLoader(dataset=val,
                                batch_size=cfg['TEST_BATCH_SIZE'],
                                shuffle=True,
                                num_workers=4,
                                )

        model.train()  # set to training mode
        
        # time start
        start_time = time.time()
        for idx, (inputs, labels) in enumerate(train_loader):
            # compute output
            inputs = inputs.to(cfg['DEVIVE'])
            labels = labels.to(cfg['DEVIVE']).long()
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # auc
            result = nn.functional.softmax(outputs,dim = 1)[:,1].data.cpu().numpy()
                
            fpr, tpr, thresholds = metrics.roc_curve(labels.data.cpu().numpy(), result, pos_label=1)
            # break
                
            # acc
            acc = accuracy(outputs, labels)[0]
            loss_hist.append(loss.data.cpu().numpy())

            optimizer_ft.zero_grad()
            loss.backward()
            optimizer_ft.step()

            if idx% 10  == 0:
                print('epoch:{0}, batch:{1}, loss:{2}, train_acc:{3}, train_auc:{4}'.format(epoch, idx, np.mean(loss_hist), acc.data.cpu().numpy(), metrics.auc(fpr, tpr)))
        
        # writer.add_scalar("Training_Loss", np.mean(loss_hist), epoch + 1)
        pred_all = []
        y_all = []   
        # val stage
        model.eval()
        with torch.no_grad():
            
            for idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(cfg['DEVIVE'])
                labels = labels.to(cfg['DEVIVE']).long()
                outputs = model(inputs)

                result = nn.functional.softmax(outputs,dim = 1)[:,1].data.cpu().numpy()
                y = labels.data.cpu().numpy()

                pred_all.extend(result)
                y_all.extend(y)
                print('eval {}/{}'.format(len(pred_all), len(val)), end = '\r')
        

        fpr, tpr, thresholds = metrics.roc_curve(y_all, pred_all, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        
        if auc > max_auc:
            print('get better auc')
            max_auc = auc
            # save best model
            torch.save(model.state_dict(),os.path.join(cfg['BEST_MODEL_ROOT'], '{}_fold_best_model_{}.pth'.format(kfold_index, cfg['MODEL_NAME'])))

            # save every stage best model
            torch.save(model.state_dict(),os.path.join(cfg['MODEL_ROOT'], '{}_fold_{:.4f}_{}.pth'.format(kfold_index, auc, cfg['MODEL_NAME'])))
        writer.add_scalar("Training_Loss", np.mean(loss_hist), epoch + 1)
        writer.add_scalar("val_auc", auc, epoch + 1)

        print('*'*40)
        print('kflod:{0}, epoch:{1}, val_auc:{2}, best_auc:{3}'.format(kfold_index, epoch, auc, max_auc))
        print('train and val time use {:.2f} seconds'.format(time.time() - start_time))
        print('*'*40)
        # scheduler.step(max_auc)
        scheduler.step()

# val test
def val_test(kfold_index = 0, model = None, k_val = None, cfg = None, data_transforms = None):
    model = model.to(cfg['DEVIVE'])
    
    test = rubbishDataSet(file_path = cfg['TEST_DATA'], 
                                transform = data_transforms['test'],
                                is_train = False)
                                
    val = rubbishDataSet(file_path = cfg['TRAIN_DATA'], 
                                kfold_data = k_val[kfold_index],
                                transform = data_transforms['val'])

    test_loader = DataLoader(dataset=test,
                        batch_size=cfg['TEST_BATCH_SIZE'],
                        shuffle=False,
                        num_workers=4,
                        pin_memory = True
                        )

    val_loader = DataLoader(dataset=val,
                                batch_size=cfg['TEST_BATCH_SIZE'],
                                shuffle=False,
                                num_workers=4,
                                )
                                
    model.eval()
    
    # get test df 
    print('start test fold ', kfold_index)
    pic = []
    pred = []
    for _, (data, pic_id) in enumerate(test_loader):
        with torch.no_grad():
            inputs = data.to(cfg['DEVIVE'])
            outputs = model(inputs)

            result = nn.functional.softmax(outputs,dim = 1)[:,1].data.cpu().numpy()
            
            pic.extend(list(pic_id))
            pred.extend(result)
            print('test stage {}/{}'.format(len(pic), len(test)), end = '\r')

    test_df = pd.DataFrame({'pic_id':pic, 'pred':pred})
    sub_name = os.path.join(cfg['SUB_ROOT'], 'test_fold_{}.csv'.format(kfold_index))
    test_df.to_csv(sub_name, index = 0)
    print('save test_fold_{0}'.format(kfold_index))
    
    #get val df
    print('start val fold ', kfold_index)
    labels = []
    pred = []
    for _, (data, label) in enumerate(val_loader):
        with torch.no_grad():
            inputs = data.to(cfg['DEVIVE'])
            outputs = model(inputs)


            result = nn.functional.softmax(outputs,dim = 1)[:,1].data.cpu().numpy()
            
            labels.extend(label.numpy())
            pred.extend(result)
            print('val stage {}/{}'.format(len(labels), len(val)), end = '\r')

    val_df = pd.DataFrame({'pred':pred, 'label':labels})
    sub_name = os.path.join(cfg['SUB_ROOT'], 'val_fold_{}.csv'.format(kfold_index))
    val_df.to_csv(sub_name, index = 0)
    print('save val_fold_{0}'.format(kfold_index))
    print('*'*50)
    
if __name__ == '__main__':

    # load configurations
    cfg = configurations[1]

    # data augmentation
    data_transforms = {
    'train': transforms.Compose([
        RandomErasing(p = 1,is_ract = False),
        transforms.RandomRotation(35),
        transforms.RandomAffine(degrees =30, translate = (0.2,0.2), scale = (0.7, 0.95), shear = 45),
        SelectROI(use_minrect = True),


        transforms.Resize([256,256]),
        transforms.CenterCrop(cfg['INPUT_SIZE']),
        # transforms.RandomCrop(cfg['INPUT_SIZE']),
        
        # transforms.Resize(cfg['INPUT_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['RGB_MEAN'], cfg['RGB_STD'])
    ]),
    
    'val': transforms.Compose([
        RandomErasing(p = 1,is_ract = False),
        transforms.RandomRotation(35),
        transforms.RandomAffine(degrees =30, translate = (0.2,0.2), scale = (0.7, 0.95), shear = 45),
        SelectROI(use_minrect = True),

        transforms.Resize([256,256]),
        transforms.CenterCrop(cfg['INPUT_SIZE']),
        # transforms.RandomCrop(cfg['INPUT_SIZE']),
        # transforms.Resize(cfg['INPUT_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['RGB_MEAN'], cfg['RGB_STD'])
    ]),
    
    'test': transforms.Compose([
        SelectROI(use_minrect = True),
        transforms.Resize([256,256]),
        # transforms.RandomCrop(cfg['INPUT_SIZE']),
        transforms.CenterCrop(cfg['INPUT_SIZE']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['RGB_MEAN'], cfg['RGB_STD'])
    ])
    }

    
    #   k fold
    k_train, k_val = kfold_trainval(data_root = cfg['TRAIN_DATA'])

    # for i in range(5):
    #     print('train stage {} fold'.format(i) )
        
    #     # load pretrained model
    #     model = get_pretrainedmodels(model_name = cfg['MODEL_NAME'],
    #                         num_outputs = cfg['NUM_CLASS'],
    #                         pretrained = True,
    #                         freeze_conv = False
    #     )
    #     print('load model')

    #     # make log & time dir 
    #     now_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    #     new_dir = os.path.join(cfg['LOG_ROOT'], now_time)
    #     if not os.path.exists(new_dir):
    #         os.mkdir(new_dir)
    #     writer = SummaryWriter(new_dir)
        
    #     # train stage
    #     train(kfold_index = i, 
    #         model = model, 
    #         k_train = k_train, 
    #         k_val = k_val,
    #         cfg = cfg,
    #         data_transforms = data_transforms,
    #         writer = writer)
            
    print('start text')
    
    
    for i in range(5):
        print('test stage {} fold'.format(i) )
        #   test        
        
        # load pretrained model
        model = get_pretrainedmodels(model_name = cfg['MODEL_NAME'],
                            num_outputs = cfg['NUM_CLASS'],
                            pretrained = False,
                            freeze_conv = False
        )
        model.load_state_dict(torch.load(os.path.join(cfg['BEST_MODEL_ROOT'], '{0}_fold_best_model_{1}.pth'.format(i, cfg['MODEL_NAME']))))
        print('load model')

        # val stage
        val_test(kfold_index = i, 
            model = model, 
            k_val = k_val, 
            cfg = cfg, 
            data_transforms = data_transforms)       