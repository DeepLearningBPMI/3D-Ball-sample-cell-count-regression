# this project is used to preprocess the data (clahe, augmentation, resize，scaling, centering, standarlize pixel values), and aslo generate data batch, image:BCDWH, label
# creat m.zhou@vu.nl, 2024.04.26
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import tifffile
from volumentations import *
from torch import nn


def apply_clahe_3d(image, clip_limit=2.0, grid_size=(8, 8)):
    result_image = np.zeros_like(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    # Apply CLAHE slice by slice
    for c in range(image.shape[-1]):
        for i in range(image.shape[0]):
            result_image[i,:,:,c] = clahe.apply(image[i,:,:,c])

    # print("result_image.shape:",result_image.shape)
    return result_image

# AUGMENTATIONS_TRAIN = Compose([
#     Flip(0, p=0.5),
#     Flip(1, p=0.5),
#     # Flip(2, p=0.5),
#     # RandomRotate90((1, 2), p=0.5),
#     # Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
#     ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
# ])
def get_augmentation(image_size):
    return Compose([
        # Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
        # RandomCropFromBorders(crop_value=0.1, p=0.5),
        # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
        # Resize(patch_size, interpolation=1, resize_type=0, always_apply=True, p=1.0),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        # Flip(2, p=0.5),
        # RandomRotate90((1, 2), p=0.5),
        # GaussianNoise(var_limit=(0, 5), p=0.2),
        # RandomGamma(gamma_limit=(80, 120), p=0.2),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ], p=1.0)

def normalize_3d_image(image):
    """
    Normalize a 3D image with multiple channels.
    - image (torch.Tensor): Input 3D image tensor of shape (D, H, W, C)
    Returns:
    - torch.Tensor: Normalized 3D image tensor
    """
    # Initialize an empty list to hold the normalized channels
    normalized_channels = []
    
    for cc in range(image.shape[-1]):
        channel = image[...,cc]
        # Scale channel to [0, 1]
        channel = channel / (channel.max() + 1e-8)
        
        # Calculate mean and std for the current channel
        mean = channel.mean()
        std = channel.std() + 1e-8
        
        # Normalize the current channel
        normalized_channel = (channel - mean) / std
        
        # Append the normalized channel to the list
        normalized_channels.append(normalized_channel)
    
    # Stack the normalized channels back together
    normalized_image = torch.stack(normalized_channels, dim=-1)
    
    return normalized_image


class TiffDataGenerator(Dataset):
    def __init__(self, image_filenames, label_filenames, D, W, H, batch_size=1, clahe = True, augmentations=False):
        self.image_filenames = image_filenames
        self.label_filenames = label_filenames
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.clahe = clahe
        self.D = D
        self.W= W
        self.H = H
        
    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.image_filenames))
        
        # Check if there are no more images left
        if start_idx >= len(self.image_filenames):
           print("All images have been successfully loaded.")
           raise StopIteration

        batch_x = self.image_filenames[start_idx:end_idx]
        batch_y = self.label_filenames[start_idx:end_idx]
        print("Batch_x:", batch_x)
        print("Batch_y:", batch_y)
        
        images = []
        labels = []

        for i, (image_filename, label_filename) in enumerate(zip(batch_x, batch_y)):
            # Load multi-channel image from TIFF file
            print(f"Loading image: {image_filename}")
            try:
                image = tifffile.imread(image_filename)
                #print("Image shape:", image.shape)
                print(f"Successful loading image file {image_filename}")

                if self.clahe:
                    image = apply_clahe_3d(image=image)
                print("Image shape:", image.shape)    
                print("Image type:", image.dtype)
                
                # Perform augmentations if required
                if self.augmentations:
                    aug_channels = []
                    image_size = image.shape[0:3]
                    print(image_size)
                    for c in range(image.shape[-1]):
                        channel = image[...,c]
                        print("channel shape:", channel.shape)
                        aug = get_augmentation(image_size) 
                        data = {'image': channel}
                        aug_data = aug(**data)
                        aug_channel = aug_data['image']
                        aug_channels.append(aug_channel)
                    image = np.stack(aug_channels, axis=-1)
                    print("aug_image shape:", image.shape)  
            
                image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
                if torch.any(torch.isnan(image)):
                   raise ValueError("NaN values found in image after augmentation.")
                print("Image shape after torch conversion:", image.shape)
                
                # normalize the image
                image = normalize_3d_image(image)
                print("Image shape after normalization:", image.shape)
                if torch.any(np.isnan(image)):
                    raise ValueError("NaN values found in image after normalization.")
                print("Image shape after normalization:", image.shape)

                # print("Image type:", image.dtype)
                # Read and process labels
                with open(label_filename, 'r') as label_file:
                    label_str = label_file.read()
             
                # Split the string using a comma separator and convert to numbers
                label = [float(x) for x in label_str.split(',')]
                # Convert the list of numbers to a PyTorch tensor
                label_tensor = torch.as_tensor(label, dtype=torch.float)    
                
                #print("Image shape:", image.shape)

                
                # print("Image type:", image.dtype)
                # Perform clahe if required
           
                # Resize and preprocess the image
                image = np.moveaxis(image.numpy(), -1, 0)  # Adjust axes as needed
                print("Image shape:", image.shape)
                image = resize_volume(image, self.D, self.W, self.H)  # Implement your resize_volume function
                print(" Resized Image shape:", image.shape)

                # Convert to PyTorch format
                image = torch.tensor(image, dtype=torch.float32)
                #print("Image shape torchtransform:", image.shape)
            
                images.append(image)
                labels.append(label_tensor)
            
            except Exception as e:
                print(f"Error loading image file {image_filename}: {e}")
                continue

           
        images = torch.stack(images, dim=0)
        labels = torch.stack(labels, dim=0)
         
        return images, labels

def resize_volume(img,D,W,H):
    """Resize across z-axis"""
    
    img = torch.tensor(img, dtype=torch.float32)

    # Add batch and channel dimensions
    img = img.unsqueeze(0)

    # Set the desired depth
    desired_depth = D
    desired_width = W
    desired_height = H

    img = torch.nn.functional.interpolate(
        img,  
        size=(desired_depth, desired_width, desired_height),
        mode='trilinear',
        align_corners=False
    )
    return img.squeeze().numpy()

def adjust_brightness_contrast(image, brightness=30, contrast=30):
    """
    Adjust the brightness and contrast of an image.
    Brightness and contrast values are in the range [-255, 255].
    """
    # Clip contrast and brightness values
    contrast = np.clip(contrast, -255, 255)
    brightness = np.clip(brightness, -255, 255)

    # Apply contrast
    image = image.astype(np.float32)
    if contrast != 0:
        factor = (259 * (contrast + 255)) / (255 * (259 - contrast))
        image = 128 + factor * (image - 128)
    
    # Apply brightness
    image = image + brightness

    # Clip to valid range
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)