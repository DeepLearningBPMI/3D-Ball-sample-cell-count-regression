import os
import torch
import pathlib
import pickle
import numpy as np
import torch.nn as nn
import math
import pandas as pd
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
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from PIL import Image
import cv2
from preprocess_3d import TiffDataGenerator
from resnet_3d_train_parallel import  resnet50


code_base = '/scistor/guest/mzu240/BAL/'
_DATA_SELECTOR = '*.tif'
classes_name = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Macrophages']
batch_size = 1
lr = 0.0012
neuro_hidden =32
dropout =0.3
D=40
W=400
H=400

def run(dataset):
    # ------------------------------------- Configuration options -------------------------------------------
   
    # For fold results
    results = {}
  
    # Set fixed random number seed
    torch.manual_seed(42)
  
    # Get all patient directories
    dirs = os.listdir(dataset)

    # Categorize samples by types
    bal_samples = [d for d in dirs if "BAL" in d and "BALI" not in d]
    bali_samples = [d for d in dirs if "BALI" in d and "Hep" not in d]
    hep_samples = [d for d in dirs if "Hep" in d]
    
    # Combine remaining samples
    all_samples = bal_samples +bali_samples

    # Make dataframe for data split
    samples = []
    for r in all_samples:
        if 'BALI' in r:
            samples.append('BALI')
        else:
            samples.append(' BAL')    
    df = pd.DataFrame()
    df['dirs'] = all_samples
    df['sample'] = samples

    test_dirs =  df.dirs
    test_folders = test_dirs
   
    # ------------------------------------- step 1/5 : Loading data -------------------------------------------
    test_images = [str(img) for dir in test_folders for img in pathlib.Path(os.path.join(dataset, dir)).rglob(_DATA_SELECTOR)]
    print(f"Total test images: {len(test_images)}")
    test_labels = [str(pathlib.Path(case).parents[0] / 'labels.txt') for case in test_images]
    print(f"Total test labels: {len(test_labels)}")
    # cl_weight_test = weight_calculation(test_labels)
    # print("Class Weights:", cl_weight_test)
    
    print(f'test_folders:')
    for folder in test_folders:
            print(os.path.join(dataset_dir, folder))

        
    test_generator = TiffDataGenerator(test_images, test_labels,  D, W,H, batch_size, augmentations=False, clahe = True)
    print("Total batches in test_generator:", len(test_generator))

     # ------------------------------------ step 5/5 : testing -------------------------------------------------- 
    # Print about testing
    print('Starting testing')
        
    model = resnet50(D, W, H, neuro_hidden, dropout)
    criterion = nn.L1Loss() 
    model.load_state_dict(torch.load('/scistor/guest/mzu240/BAL/Results_3D/08-08 11:25/resnet3d50_lr0.0012_dropout0.3_SGD_32neurons/model.pth'))
    print("Model loaded for testing.")
    model.eval()

    #log
    result_dir = code_base + '/test-results-BAL'
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d %H:%M')
    log_dir =  os.path.join(result_dir, time_str, f'ResNet3d_test__lr{lr}_dropout{dropout}_08_08_11:25')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)


    # test_loss = 0.0
    # Create empty lists to store metrics for each sample  
    mae_scores = []

    # Evaluationfor this fold
    with torch.no_grad():
        #Iterate over the test data
        index = 0
        for batch in test_generator:
            # Check if the batch is None (no more data)
            if batch is None:
                break

            inputs, labels = batch
            print("test batch - Inputs shape:", inputs.shape, "Labels shape:", labels.shape)
            
            # # Transfer Data to GPU if available
            # if torch.cuda.is_available():
            #    inputs, labels = inputs.cuda(), labels.cuda()

            index = index +1
            predictions = model(inputs)

            # loss_fn = WeightedMSELoss(weights=cl_weight_test)
            # loss = criterion(predictions, labels)
            
            # test_loss += loss.item()
            # writer.add_scalars('Loss_group', {'test_loss': loss}, index)
            true_labels = labels.numpy()
            predicted_labels = predictions.numpy()

            # Iterate over classes to plot individual bars
            for i in range(true_labels.shape[1]):  
                true_labels_i = true_labels[:, i]  # Take mean if there are multiple samples
                predicted_labels_i = predicted_labels[:, i]  # Take mean if there are multiple samples

                mae_i = mean_absolute_error(true_labels[:, i], predicted_labels[:, i])
                mae_scores.append(mae_i)
                # Add bar plot for each class
                writer.add_scalars('True_vs_Predicted/Class_{}'.format(i + 1),
                        {'True_Label': true_labels_i, 'Predicted_Label': predicted_labels_i},
                            global_step=index)
                writer.add_scalars('MAE/Class_{}'.format(i + 1),
                        {'MAE':mae_i}, global_step=index)
                  
        # Calculate average metrics across all samples
        avg_mae = np.mean(mae_scores)  
        print("Average MAE:", avg_mae)

    
if __name__ == '__main__':

  dataset_dir = code_base + "dataset_test/"
 
  run(dataset_dir)