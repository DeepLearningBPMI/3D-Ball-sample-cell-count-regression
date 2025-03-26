import os
import torch
import pathlib
import torch.nn as nn
from tqdm import tqdm  # Import tqdm for a progress bar
from torch import nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import ToPILImage
import torchvision
import cv2
from preprocess_3d import TiffDataGenerator, resize_volume
from resnet_3d_train import resnet10, resnet18, resnet34, resnet50,  resnet101, resnet152, resnet200
from medcam import medcam

code_base = '/scistor/guest/mzu240/BAL/'
_DATA_SELECTOR = '*.tif'
classes_name = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Macrophages']
batch_size = 1
epochs = 1
lr = 0.00117
dropout =0.295
D=62
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

    test_generator = TiffDataGenerator(test_images, test_labels, D, W,H, batch_size, clahe = True, augmentations=False)  
    print("Total batches in test_generator:", len(test_generator))

    # ------------------------------------ step 2/5 : load model------------------------------------
    model = resnet50(D, W, H, dropout)
    model.load_state_dict(torch.load('/scistor/guest/mzu240/BAL/Results_3D/07-03 21:17/resnet3d50_5stackspercase_lr0.00117_dropout0.295_SGD_32neurons/model.pth'))
    model = model.to('cuda')
    print("model:", model)
    model.eval()
    model = medcam.inject(model, output_dir='attention_maps', backend='gcam', layer='layer4', label='best', save_maps=True)
  
    
    k=-1
    # ------------------------------------ step 4/5 : activation map --------------------------------------------------  
    for batch in test_generator:
        # Check if the batch is None (no more data)
        if batch is None:
            break

        inputs, labels = batch
        print("test batch - Inputs shape:", inputs.shape, "Labels shape:", labels.shape)

        # # Transfer Data to GPU if available
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        
        # locate the visulization cases. test_images[k]
        k = k +1 
        
        outputs = model(inputs)
        print ("outputs.shape", outputs.shape)
  
        criterion = nn.L1Loss()  
        loss = criterion(outputs, labels)
        model.zero_grad()
        # loss.backward()

if __name__ == '__main__':

  dataset_dir = code_base + "validation_dataset"
  activation_dir = code_base + "activation"
  run(dataset_dir)