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
from torch.nn import MSELoss, L1Loss
import tifffile
from sklearn.model_selection import train_test_split
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
import pandas as pd
from random import sample
from functools import partial
from sklearn.metrics import mean_absolute_error
from preprocess_3d import TiffDataGenerator
from resnet_3d_train_parallel import resnet50
from class_weight import weight_calculation, WeightedMSELoss, WeightedMAELoss
from torch.optim.lr_scheduler import StepLR

code_base = '/scistor/guest/mzu240/BAL/'
_DATA_SELECTOR = '*.tif'
classes_name = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Macrophages']
batch_size = 1
epochs = 500
lr = 0.003
neuro_hidden =32
dropout =0.3
D= 40
W= 400
H= 400

def run(dataset):
    # ------------------------------------- Configuration options -------------------------------------------
   
    # For fold results
    results = {}
  
    # Set fixed random number seed
    torch.manual_seed(42)
  
    # Get all patient directories
    dataset_dir = code_base + "cleaning_3d_20slices_4types/"
    dirs = os.listdir(dataset_dir)

    # Categorize samples by types
    bal_samples = [d for d in dirs if "BAL" in d and "BALI" not in d]
    bali_samples = [d for d in dirs if "BALI" in d and "Hep" not in d]
    hep_samples = [d for d in dirs if "Hep" in d]

    # Combine remaining samples
    all_samples = hep_samples

    # Randomly split the remaining samples into training and validation sets
    train_dirs, val_dirs = train_test_split(all_samples, test_size=0.2, random_state=42)

    # Combine the initial selected validation samples with the randomly split validation samples
    val_dirs = val_dirs
    train_dirs = train_dirs
    
    # Start print
    print('--------------------------------')

    result_dir = code_base + '/Results_3D_1024'
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d %H:%M')
    log_dir =  os.path.join(result_dir, time_str, f'resnet3d50_Hep_lr{lr}_dropout{dropout}_adam_neurons_hidden{neuro_hidden}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)

     # Extract patient folders for the current fold
    train_folders = train_dirs
    val_folders = val_dirs

    
    print(f'Training folder:')
    for folder in train_folders:
            print(os.path.join(dataset_dir, folder))
    

    print(f'Validation folders:')
    for folder in val_folders:
            print(os.path.join(dataset_dir, folder))


    # ------------------------------------- step 1/5 : Loading data -------------------------------------------
    train_images = [str(img) for folder in train_folders for img in pathlib.Path(os.path.join(dataset_dir, folder)).rglob(_DATA_SELECTOR)]
    print(f"Total training images: {len(train_images)}") 
    train_labels = [str(pathlib.Path(case).parents[0] / 'labels.txt') for case in train_images]
    print(f"Total training labels: {len(train_labels)}")
    # calclulate the class weight
    # cl_weight_tr = weight_calculation(train_labels)
    # print("Class Weights:", cl_weight_tr)

    val_images = [str(img) for folder in val_folders for img in pathlib.Path(os.path.join(dataset_dir, folder)).rglob(_DATA_SELECTOR)]
    print(f"Total validation images: {len(val_images)}") 
    val_labels = [str(pathlib.Path(case).parents[0] / 'labels.txt') for case in val_images]
    print(f"Total validation labels: {len(val_labels)}")

    train_generator = TiffDataGenerator(train_images, train_labels, D, W,H, batch_size, clahe = True, augmentations=True)
    val_generator = TiffDataGenerator(val_images, val_labels, D, W, H, batch_size, clahe = True, augmentations=False)   
    print("Total batches in train_generator:", len(train_generator))
    print("Total batches in val_generator:", len(val_generator))

    # ------------------------------------ step 2/5 : create network ------------------------------------
    model = resnet50(D, W, H, neuro_hidden, dropout)
    model = model.to('cuda')
    if torch.cuda.is_available():
        model = model.cuda()
    summary(model, input_size=(3, D, W, H), device='cuda')

  

#   # ------------------------------------ step 3/5 : choose loss and optimizer ------------------------------------
    criterion = nn.L1Loss()    
    optimizer = optim.SGD(model.parameters(), lr=lr)  
    min_valid_loss = np.inf     
    
    # ------------------------------------ step 4/5 : training --------------------------------------------------  
    for epoch in range (0, epochs):
        # print epoch
        print(f'Starting epoch {epoch+1}')
        
        #set current loss value
        train_loss = 0.0
        # weighted_train_loss =0.0
        # scheduler.step()
        #Iterate over the training data
        for batch in train_generator:
            # Check if the batch is None (no more data)
            if batch is None:
                break

            inputs, labels = batch
            print("train batch - Inputs shape:", inputs.shape, "Labels shape:", labels.shape)

            # # Transfer Data to GPU if available
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            
            optimizer.zero_grad()

            outputs = model(inputs)
        
            # loss_fn = WeightedMSELoss(weights=cl_weight_tr)
            # loss = loss_fn(outputs, labels)
           
            loss = criterion(outputs, labels)

            # weighted_loss_fn = WeightedMAELoss(weights=cl_weight_tr)
            # weighted_loss = weighted_loss_fn (outputs, labels)
            loss.backward()
            # weighted_loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # weighted_train_loss += weighted_loss.item()

        writer.add_scalars('Loss_group', {'train_loss': train_loss/ len(train_generator) }, epoch +1)
        # writer.add_scalars ('Loss_group', {'weighted_train_loss': weighted_train_loss/ len(train_generator) }, epoch +1)
 

        valid_loss = 0.0
        model.eval()
        for batch in val_generator:
            # Check if the batch is None (no more data)
            if batch is None:
                break
                    
            inputs, labels = batch
            print("Validation batch - Inputs shape:", inputs.shape, "Labels shape:", labels.shape)
                    
            # # Transfer Data to GPU if available
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()


            outputs = model(inputs)
            
            # loss_fn = WeightedMSELoss(weights=cl_weight_val)
            # loss = loss_fn(outputs, labels)
            loss = criterion(outputs, labels)
           
            valid_loss += loss.item()

        writer.add_scalars('Loss_group', {'val_loss': valid_loss / len(val_generator) }, epoch +1)

        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_generator)} \t\t Validation Loss: {valid_loss / len(val_generator)}')

        if  valid_loss < min_valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
         
            # Saving State Dict
            # Saving the model
            save_path = f'{log_dir}/model.pth'
            torch.save(model.state_dict(), save_path)
        
    # Process is complete.
    print('Training process has finished. Saving trained model.')

if __name__ == '__main__':

  dataset_dir = code_base + "cleaning_3d_20slices_4types/"
#   save_dir = os.path.join(code_base, "preprocess")
#   if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
  
  run(dataset_dir)