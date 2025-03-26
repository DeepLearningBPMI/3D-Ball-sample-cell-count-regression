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
from sklearn.model_selection import train_test_split
from torch.nn import MSELoss, L1Loss
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
import optuna
from optuna.trial import TrialState
from random import sample
from functools import partial
from sklearn.metrics import mean_absolute_error
from preprocess_3d import TiffDataGenerator
from resnet_3d import resnet10, resnet18, resnet34, resnet50,  resnet101, resnet152, resnet200
from class_weight_cpu import weight_calculation, WeightedMSELoss


code_base = '/scistor/guest/mzu240/BAL/'

_DATA_SELECTOR = '*.tif'
classes_name = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Macrophages', 'Others']
classes = 5
batch_size = 1
epochs = 20
D=155
W=400
H=400

def objective(trial):
    # ------------------------------------- Configuration options -------------------------------------------
 
    # Set fixed random number seed
    torch.manual_seed(42)

    # Get all patient directories
    dataset_dir = code_base + "test-5classes-fake3d-155/"
    dirs = os.listdir(dataset_dir)
    # Split the dataset into training and validation sets
    train_dirs, val_dirs = train_test_split(dirs, test_size=0.2, random_state=42)
 
    # Start print
    print('--------------------------------')

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
    #calclulate the class weight
    # calclulate the class weight
    cl_weight_tr = weight_calculation(train_labels)
    print("Class Weights:", cl_weight_tr)

    val_images = [str(img) for folder in val_folders for img in pathlib.Path(os.path.join(dataset_dir, folder)).rglob(_DATA_SELECTOR)]
    print(f"Total validation images: {len(val_images)}") 
    val_labels = [str(pathlib.Path(case).parents[0] / 'labels.txt') for case in val_images]
    print(f"Total validation labels: {len(val_labels)}")
 
    
    train_generator = TiffDataGenerator(train_images, train_labels, D, W,H, batch_size, clahe = True, augmentations=True)
    val_generator = TiffDataGenerator(val_images, val_labels, D, W,H, batch_size, clahe = True, augmentations=False)
    print("Total batches in train_generator:", len(train_generator))
    print("Total batches in val_generator:", len(val_generator))
   
   
    # ------------------------------------ step 2/5 : generate model ------------------------------------ 
    drop_fc1 = trial.suggest_float("drop_fc1", 0.1, 0.4)         # Dropout for FC1 layer
    p = trial.suggest_int ("p",1,11)
    num_neurons = 2**p
    # drop_fc1 = 0
    model = resnet50(trial, D, W, H, drop_fc1,num_neurons)
    model = model.to('cpu')
    summary(model, input_size=(3,D,W,H), device='cpu')

    # if torch.cuda.is_available():
    #     model = model.cuda()
  
    # ------------------------------------ step 3/5 : choose loss and optimizer ------------------------------------
    # criterion_name = trial.suggest_categorical("criterion", ["MSELoss", "MAELoss"])
    criterion = nn.L1Loss()  
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # lr = 0.0011738626463730737
    # optimizer = optim.SGD(model.parameters(), lr=lr) 
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr) 
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99) 
    # if criterion_name == "MSELoss":
    #     criterion = nn.MSELoss()
    # elif criterion_name == "MAELoss":
    #     criterion = nn.L1Loss()  # Using L1Loss for Mean Absolute Error

    model.train()
    # ------------------------------------ step 4/5 : training --------------------------------------------------  
    for epoch in range (0, epochs):
        # print epoch
        print(f'Starting epoch {epoch+1}')
        
        #set current loss value
        train_loss = 0.0
        weighted_train_loss =0.0
        # scheduler.step()
        #Iterate over the training data
        for batch in train_generator:
            # Check if the batch is None (no more data)
            if batch is None:
                break

                # # Limiting training data for faster epochs.
                # if batch_idx * batch_size >= N_TRAIN_EXAMPLES:
                #    break

            inputs, labels = batch
            print("train batch - Inputs shape:", inputs.shape, "Labels shape:", labels.shape)

            # Transfer Data to GPU if available
            # if torch.cuda.is_available():
            #     inputs, labels = inputs.cuda(), labels.cuda()

            
            optimizer.zero_grad()
            outputs = model(inputs)
            # loss_fn = WeightedMSELoss(weights=cl_weight_tr)
            loss = criterion(outputs, labels)
            weighted_loss_fn = WeightedMSELoss(weights=cl_weight_tr)
            weighted_loss = weighted_loss_fn (outputs, labels)
            # loss.backward()
            loss.backward()

            optimizer.step()
 
            train_loss += loss.item()
            weighted_train_loss += weighted_loss.item() 

        valid_loss = 0.0
        model.eval()
        for batch in val_generator:
            # Check if the batch is None (no more data)
            if batch is None:
                break
                    
            inputs, labels = batch
            print("Validation batch - Inputs shape:", inputs.shape, "Labels shape:", labels.shape)
                    
            # # Transfer Data to GPU if available
            # if torch.cuda.is_available():
            #     inputs, labels = inputs.cuda(), labels.cuda()


            outputs = model(inputs)
            # loss_fn = WeightedMSELoss(weights=cl_weight_val)
            loss = criterion(outputs, labels)
                    
            valid_loss += loss.item()

        print(f'Epoch {epoch+1} \t\t Weighted_train_loss: {weighted_train_loss  / len(train_generator)} \t\t Training Loss: {train_loss / len(train_generator)} \t\t Validation Loss: {valid_loss / len(val_generator)}')

    
        trial.report(valid_loss / len(val_generator), epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            # Process is complete.

    print('Training process has finished. Saving trained model.')
        
    return valid_loss
     


if __name__ == '__main__':
    
    storage_name = "sqlite:///optuna_resnet50_dataset155slicepercase.db"
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3), direction="minimize",
        study_name="3DBAL_MedianPrunner_withaugmentation8", storage=storage_name,load_if_exists=True)
    study.optimize(objective, n_trials=30)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    
    trial = study.best_trial
    print("Best trial:")
    
    best_params = study.best_params
    print("best_params:")

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save results to csv file
    df = study.trials_dataframe().drop(['datetime_start', 'datetime_complete', 'duration'], axis=1)  # Exclude columns
    df = df.loc[df['state'] == 'COMPLETE']        # Keep only results that did not prune
    df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')                  # Sort based on accuracy
    df.to_csv('optuna_results.csv', index=False)  # Save to csv file

    # Display results in a dataframe
    print("\nOverall Results (ordered by loss):\n {}".format(df))

    # Find the most important hyperparameters
    most_important_parameters = optuna.importance.get_param_importances(study, target=None)

    # Display the most important hyperparameters
    print('\nMost important hyperparameters:')
    for key, value in most_important_parameters.items():
        print('  {}:{}{:.2f}%'.format(key, (15-len(key))*' ', value*100))
