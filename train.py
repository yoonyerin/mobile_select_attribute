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

print(f"=> Training model: {not config.evaluate}")
if __name__=="__main__":
    freeze_support()
    if config.evaluate:
        best_prec1, model = load_inference_model(device, config.bestmodel_fname) # checkpoint_fname bestmodel_fname
        test_loss, prec1, top1 = validate(dataloaders['test'], model, criterion)
        print(f"=> Best test accuracy: {prec1}, Model val acc: {best_prec1}")
        attr_acc = print_attribute_acc(top1, attribute_names)
        if config.test_preds_fname:
            json.dump(attr_acc, open(config.test_preds_fname,'w'))
    else:
        
        best_prec1, model_timer, lr, start_epoch, logger, model, optimizer = resume_checkpoint(device, config.ckp_logger_fname, config.ckp_resume)
        run_name, run_time = get_run_name_time(model, criterion, optimizer, start_epoch)
        mtimer = trainer(dataloaders, model, criterion, optimizer, logger, start_epoch, best_prec1, run_name, model_timer)
        print(f"=> Model trained time: {mtimer}")


    if not config.evaluate:
        config.evaluate = True
        #model = create_model(device)
        dataloaders, attribute_names = load_dataloaders()
        criterion = get_criterion()
        #optimizer = get_optimizer(model)
        
        best_prec1, model = load_inference_model(device, config.bestmodel_fname) # checkpoint_fname bestmodel_fname
        #best_prec1, mtimer, _, _, logger, = resume_checkpoint(model, optimizer, config.ckp_logger_fname, config.checkpoint_fname)
        test_loss, prec1, top1 = validate(dataloaders['test'], model, criterion)
        print(f"=> Best test accuracy: {prec1}, Model val acc: {best_prec1}")
        attr_acc = print_attribute_acc(top1, attribute_names)
        if config.test_preds_fname:
            json.dump(attr_acc, open(config.test_preds_fname,'a+'))
    #     best_prec1, mtimer, _, _, _, = resume_checkpoint(model, optimizer, config.ckp_logger_fname, config.bestmodel_fname)# config.bestmodel_fname  config.checkpoint_fname
    #     #print(model)
    #     test_loss, prec1, top1 = validate(dataloaders['test'], model, criterion)
    #     print(f"=> Best test accuracy: {prec1}, Model val acc: {best_prec1}")
    #     print_attribute_acc(top1, attribute_names)
