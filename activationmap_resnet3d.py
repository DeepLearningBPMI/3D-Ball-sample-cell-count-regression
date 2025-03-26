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
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, SoftmaxOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
from torchvision.transforms.transforms import ToPILImage
import torchvision
import cv2
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from preprocess_3d import TiffDataGenerator, resize_volume
from resnet_3d_train import resnet10, resnet18, resnet34, resnet50,  resnet101, resnet152, resnet200
from matplotlib.colors import ListedColormap

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
    model.load_state_dict(torch.load('/scistor/guest/mzu240/BAL/Results_3D/07-03 21:17/resnet3d50_5stackspercase_lr0.00117_dropout0.295_SGD_32neurons/model.pth'))
    model = model.to('cuda')
    print("model:", model)
    
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

        k = k +1
        for j in range(5):    
            activation=[]
            grad=[]     
            def forward_hook(module,input,output):
                activation.append(output)

            def backward_hook(module,grad_in,grad_out):
                grad.append(grad_out[0])
            # Add hook to get the tensors

            model.layer4[-1].register_forward_hook(forward_hook)
            model.layer4[-1].register_full_backward_hook(backward_hook)
            outputs = model(inputs)
            print ("outputs.shape", outputs.shape)
            print(torch.argmax(outputs)) # print the maximume value of the outputs
        
            #class for neutrophils 0, eosinophils 1, lymphocytes 2, macrophages/monocytes 3, others 4.
            criterion = nn.MSELoss()  
            loss = criterion(outputs, labels[:, j])
        
            model.zero_grad()
            loss.backward()

            # get the gradients and activations collected in the hook
            grads=grad[0].cpu().data.numpy().squeeze() # 去掉batch维度,depth 维度，因为depth=1
            fmap=activation[0].cpu().data.numpy().squeeze()
            print("fmap.shape",fmap.shape)
            print("grads.shape",grads.shape)

            # Get the mean value of the gradients of every featuremap
            tmp=grads.reshape([grads.shape[0],-1]) #flatten
            weights=np.mean(tmp,axis=1)
            print("weights.shape",weights.shape)

            cam = np.zeros(grads.shape[1:])
            for i,w in enumerate(weights):
                cam += w*fmap[i,:]
            
            cam=(cam>0)*cam
            print("cam.shape",cam.shape)
            print(cam)
            cam=cam/cam.max()*255
            print(cam)
            print(cam > 255*0.85)
            print("cam.shape",cam.shape)

          
            # slice,_= inputs[0].max(dim=1) # select first batch
            slice = inputs[0, :,10,:,:] #CWH
            print("slice.shape",slice.shape)
            npic= slice.permute(1,2,0) #WHC
            npic_np = npic.cpu().numpy()
            print("npic.shape",npic.shape)
            
            
            cam = cv2.resize(cam[j,:,:],(npic.shape[1],npic.shape[0]))
            print("cam.shape",cam.shape)
        
            # colors = [(1, 0.6, 0.8),  # Pink
            #          (0, 1, 0)]       # Green

            # # Create the custom colormap
            # cmap = ListedColormap(colors)
            
            # heatmap = cmap(np.uint8(cam))
            heatmap=cv2.applyColorMap(np.uint8(cam),cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # npic_np_float = npic_np.astype(np.float32) / 255.0
            cam_img= 0.6*npic_np+ 0.4*heatmap
            # Blend the heatmap with the original image
            # cam_img_float = cv2.addWeighted(npic_np_float, 1, heatmap[:, :, :3], 0.5, 0, dtype=cv2.CV_32F)
            # cam_img = (cam_img_float* 255).astype(np.uint8)
            print(cam_img.shape)

            #Save images
            # Save images to subfolders with the same name as the input images
            dir_name = os.path.basename(os.path.dirname(test_images[k]))
            print("dir name:", dir_name)
            input_filename = os.path.basename(test_images[k]) 
            print("input name:", input_filename)
            save_dir = os.path.join(activation_dir, dir_name, input_filename, f"class_{j}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save slice image
            slice_img = Image.fromarray(np.uint8(npic_np))
            slice_img.save(os.path.join(save_dir, input_filename + "_slice.jpg"))

            # Save heatmap
            heatmap_uint8 = (heatmap * 255).astype(np.uint8)
            # Create an Image object from the heatmap array
            heatmap_img = Image.fromarray(heatmap_uint8, 'RGB')
            # heatmap_img = Image.fromarray(heatmap)
            heatmap_img.save(os.path.join(save_dir,input_filename + "_heatmap.png"))

            # Save overlay image
            overlay_img = Image.fromarray(np.uint8(cam_img))
            overlay_img.save(os.path.join(save_dir, input_filename + "_overlay.jpg"))

        

if __name__ == '__main__':

  dataset_dir = code_base + "visualization/"
  activation_dir = code_base + "activation_resnet3d"
  run(dataset_dir)