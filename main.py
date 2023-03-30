import sys, os
import shutil
import time
import random
import numpy as np
import copy
from datetime import datetime
from distutils.dir_util import copy_tree #for recursive filecopying
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt


import config
from celeba_dataset import CelebaDataset
import models

#import schedulers
from utils import Logger, AverageMeter, Bar, ModelTimer, savefig, adjust_learning_rate, accuracy, print_attribute_acc, create_dir_ifne, add_weight_decay, mixup_data
from attribute_select_ft import *

import pandas as pd
import matplotlib.pyplot as plt

device=torch.device("mps")

##CelebA
import os, glob
local_download_path = '../data/img_align_celeba'

model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name])) # and name.islower() and not name.startswith("__"))
print(f"Available Models: {model_names}")

criterion = get_criterion()

dataloaders, attribute_names = load_dataloaders()
criterion = get_criterion()
#optimizer = get_optimizer(model)


if __name__=="__main__":
    freeze_support()

    best_prec1, model = load_inference_model(device, config.bestmodel_fname) # checkpoint_fname bestmodel_fname
    #prec1, top1= validate(dataloaders['test'], model)
    #print(f"=> Original accuracy: {prec1}")
    
    #attr_acc = print_attribute_acc(top1, attribute_names)
    service_prec1, service_top1, service_cert = service_validate(dataloaders['service'], model)
    original_prec1, original_top1, original_cert = service_validate(dataloaders['original'], model)
    print(f"=> Original accuracy: {original_prec1}, Service_accuracy: {service_prec1}")
    
    attr_acc = print_attribute_acc(service_top1, service_attribute_names)
    
    #print(f"df.mean type: {type(service_cert)}")
    cert_frame=pd.DataFrame({"service_skin_cert": service_cert, 
                  "original_skin_cert":original_cert}, index=service_attribute_names)
    
    plot_skin_certf(cert_frame)
    cert_frame.to_csv("./print/cert_frame.csv", index=False)
    
    if config.test_preds_fname:
        json.dump(attr_acc, open(config.test_preds_fname,'w'))

    #     best_prec1, mtimer, _, _, _, = resume_checkpoint(model, optimizer, config.ckp_logger_fname, config.bestmodel_fname)# config.bestmodel_fname  config.checkpoint_fname
    #     #print(model)
    #     test_loss, prec1, top1 = validate(dataloaders['test'], model, criterion)
    #     print(f"=> Best test accuracy: {prec1}, Model val acc: {best_prec1}")
    #     print_attribute_acc(top1, attribute_names)







# import sys, os
# import shutil
# import time
# import random
# import numpy as np
# import copy
# from datetime import datetime
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import pandas as pd
# import json
# from pprint import pprint

# import torch
# import torch.nn as nn
# import torchvision
# from torchvision import datasets, models, transforms

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2
# import config

# from tqdm import tqdm

# from celeba_dataset import CelebaDataset, ServiceTestset
# import models
# from utils import Logger, ModelTimer, AverageMeter, accuracy, print_attribute_acc

# import seaborn as sns

# from attribute_select_ft import *


# sns.set()

# # set the backend of matplotlib to the 'inline' backend


# # check PyTorch version and cuda status
# print(torch.__version__, torch.cuda.is_available())

# # define device
# device = torch.device("cuda") # force cpu
# torch.cuda.set_device(0)

# seed_everything(seed=config.manual_seed)#config.manual_seed

# dataloaders, attribute_names = load_dataloaders(albu_transforms = True, img_h=IMAGE_H, img_w=IMAGE_W)
# test_dataset, test_loader, attribute_names = load_testset(albu_transforms = True, img_h=IMAGE_H, img_w=IMAGE_W)


# checkpoint = torch.load('/Users/yerinyoon/Documents/cubig/mobile_attribute_select/checkpoints/model_best.pth.tar')
# model.load_state_dict(checkpoint['state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer'])

# if True:
#     real_batch = next(iter(test_loader))
#     plt.figure(figsize=(12,12))
#     plt.axis("off")
#     plt.title("Private RealUser Images")
#     plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    
    
# config.INFERENCE_DIR = './checkpoints'
# lfile = 'model_best.pth.tar'
# inf_models = {}
# ctr = 0
# for filename in os.listdir(config.INFERENCE_DIR):
#     if filename == lfile:
#         best_prec1, model = load_inference_model(device, os.path.join(config.INFERENCE_DIR,lfile))
#         del model
#         inf_models[ctr] = (os.path.join(config.INFERENCE_DIR,lfile), dirt, run, best_prec1)
#         ctr += 1
                
# print(f'==> {len(inf_models)} inference model(s) found.')

# keydict = {mid: (name, acc) for mid, (_, name, _, acc) in inf_models.items()}
# pprint(f'{keydict}')


# SAVE_FILES = True
# selected_model = int(input("Enter model index: "))
# p_run = inf_models[selected_model][2]
# p_model_name = inf_models[selected_model][1]
# run_dir = os.path.join(config.INFERENCE_DIR, p_model_name)
# p_model_acc, p_model = load_inference_model(device, inf_models[selected_model][0]) 
# #print(f"=> best model val: {p_model_acc}")


# val_prec1, val_top1 = validate(dataloaders['val'], p_model)
# print(f"=> Best val accuracy: {val_prec1}")
# v_attr_acc = print_attribute_acc(val_top1, attribute_names)

# test_prec1, test_top1 = validate(dataloaders['test'], p_model)
# print(f"=> Best test accuracy: {test_prec1}")
# test_attr_acc = print_attribute_acc(test_top1, attribute_names)


# if SAVE_FILES:
#     json_save_dir = os.path.join(run_dir, p_run)
#     vpfile = os.path.join(json_save_dir, "val_preds.json")
#     json.dump(v_attr_acc, open(vpfile,'w'))
#     tpfile = os.path.join(json_save_dir, "test_preds.json")
#     json.dump(test_attr_acc, open(tpfile,'w'))
    
# del dataloaders

# maxk = 1
# preds = pd.DataFrame(index=test_dataset.imagenames, columns=attribute_names)
# preds.index.name = "Images"
# p_model.eval()

# for X, names in tqdm(test_loader, disable=False):
#     inputs = X.to(device, non_blocking=True)

#     top_k_preds = []
#     with torch.no_grad():
#         outputs = p_model(inputs) # 40, BS
    
#         for attr_scores in outputs:
#             _, attr_preds = attr_scores.topk(maxk, 1, True, True)
#             top_k_preds.append(attr_preds.t())
            
#     all_preds = torch.cat(top_k_preds, dim=0) 

#     all_preds = all_preds.permute(1,0).cpu()
#     all_preds[all_preds == 0] = -1
#     for j in range(len(names)):
#         preds.loc[names[j], :] = all_preds[j]


# if SAVE_FILES:
#     pfile = os.path.join(run_dir, "predictions.csv")
#     ptxtfile = os.path.join(run_dir, "predictions.txt")
#     preds.to_csv(ptxtfile, sep=' ', header=False)
#     preds.to_csv(pfile, index=True)
    
# stat_df = pd.DataFrame(index = attribute_names)
# stat_df.loc[:,'Testset'] = (preds.iloc[:,:] == 1).mean(axis=0)*100
# stat_df = stat_df.sort_values('Testset', ascending=False)
# fig, ax = plt.subplots()
# stat_df.plot(title='CelebA Private Testset Prediction Frequency Distribution', 
#              kind='bar', figsize=(20, 5), ax=ax, color='green')
# for p in ax.patches:
#     value = round(p.get_height(),2)
#     ax.annotate(str(value), xy=(p.get_x(), p.get_height()))
# plt.savefig('private_test.png',dpi=160, bbox_inches='tight')
# print(preds[(preds['Young']==1) & (preds['Gray_Hair']==1)].index)
# print(preds[(preds['Male']==-1) & (preds['Mustache']==1)].index)
# print(preds[(preds['Male']==-1) & (preds['Goatee']==1)].index)
# print(preds[(preds['Gray_Hair']==1) & (preds['Blond_Hair']==1)].index)
# print(preds[(preds['Male']==-1) & (preds['No_Beard']==-1)].index)


# print(preds[(preds['Rosy_Cheeks']==1) & (preds['Rosy_Cheeks']==1)].index)
# print(len(preds[(preds['Rosy_Cheeks']==1) & (preds['Rosy_Cheeks']==1)].index))
# print(preds[(preds['Wearing_Necklace']==1) & (preds['Wearing_Necklace']==1)].index)
# print(len(preds[(preds['Wearing_Necklace']==1) & (preds['Wearing_Necklace']==1)].index))

# inv_normalize = transforms.Normalize(
#    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
#    std=[1/0.229, 1/0.224, 1/0.225]
# )

# get_celeb_prediction(preds, name = 'George_W_Bush', first_img=True) # Male, Gray_Hair, Blonde, Necktie

# get_celeb_prediction(preds, name = 'Colin_Powell', first_img=True)  # Male, Gray_Hair, Necktie

# get_celeb_prediction(preds, name = 'Ariel_Sharon', first_img=True) # Male, Gray_Hair, Necktie