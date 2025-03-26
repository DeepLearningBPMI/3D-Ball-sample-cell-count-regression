from medcam import medcam
import os
import torch
import pathlib
import pickle
import numpy as np
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import MSELoss
import tifffile
from torchsummary import summary
from tqdm import tqdm  # Import tqdm for a progress bar
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from volumentations import *
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import KFold
from random import sample
from functools import partial
from sklearn.metrics import mean_absolute_error
from PIL import Image
from torchvision.transforms.transforms import ToPILImage
import torchvision
import cv2
from matplotlib import pyplot as plt
from preprocess_3d import TiffDataGenerator, resize_volume
from resnet_3d_train import resnet10, resnet18, resnet34, resnet50,  resnet101, resnet152, resnet200

code_base = '/scistor/guest/mzu240/BAL/'
_DATA_SELECTOR = '*.tif'
classes_name = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Macrophages', 'Others']
batch_size = 1
epochs = 1
lr = 0.0011738626463730737
dropout =0.29494577133115363
D=155
W=400
H=400

def run(dataset):
    # ------------------------------------- Configuration options -------------------------------------------

    # Get all patient directories
    dirs = os.listdir(dataset)
    test_dirs =  dirs

    # Start print
    print('--------------------------------')
    test_folders = test_dirs

    # ------------------------------------- step 1/5 : Loading data -------------------------------------------
    test_images = [str(img) for folder in test_folders for img in pathlib.Path(os.path.join(dataset_dir, folder)).rglob(_DATA_SELECTOR)]
    print(f"Total test images: {len(test_images)}") 
    test_labels = [str(pathlib.Path(case).parents[0] / 'labels.txt') for case in test_images]
    print(f"Total training labels: {len(test_labels)}")

    test_generator = TiffDataGenerator(test_images, test_labels, D, W,H, batch_size, augmentations=False, clahe = True)  
    print("Total batches in test_generator:", len(test_generator))

    # ------------------------------------ step 2/5 : load model------------------------------------
    model = resnet50(D, W, H, dropout)
    model.load_state_dict(torch.load('/scistor/guest/mzu240/BAL/Results_3D/04-28 16:23/resnet3d50_5stackspercase_lr0.0011738626463730737_dropout0.29494577133115363/model.pth'))
    model = model.to('cuda')
    print("model:", model)

    model = medcam.inject(model, output_dir='activation', backend='gcam', layer='layer4', label='best', save_maps=True)

    for batch in test_generator:
        _ = model(batch[0])


if __name__ == '__main__':

    dataset_dir = code_base + "test-5classes-fake3d-155/"
    run(dataset_dir) 


    