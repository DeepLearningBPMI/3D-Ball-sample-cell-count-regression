import os
import torch
import pathlib
import numpy as np
import torch.nn.functional as F
import math
from torch import nn, Tensor
from PIL import Image
import cv2
from preprocess_3d import TiffDataGenerator, resize_volume
from resnet_3d_train import resnet50
from typing import Optional, List
import torchvision.transforms as transforms
from matplotlib import cm
from torchvision.transforms.functional import to_pil_image

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

    k=-1 ##folder order
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
        feature_map = []
        def forward_hook(module, inp, outp):     # 定义hook
            feature_map.append(outp)    # 把输出装入字典feature_map
       
        # Add hook to get the tensors
        model.layer4[-1].register_forward_hook(forward_hook)
    
        with torch.no_grad():
            outputs = model(inputs)
            print ("outputs.shape", outputs.shape)
  
        print(feature_map[0].size())
        
        cls = torch.argmax(outputs).item()
        weights = model.get('fc1').weight.data[cls,:] #获取类别对应的权重 
        print("weights.shape",weights.shape)

        cam = (weights.view(*weights.shape,1,1,1)*feature_map[0].squeeze(0)).sum(0)

        def _normalize(cams: Tensor) -> Tensor:
            """CAM normalization"""
            cams.sub_(cams.flatten(start_dim=-3).min(-1).values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
            cams.div_(cams.flatten(start_dim=-3).max(-1).values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

            return cams
        
        cam = _normalize(F.relu(cam, inplace=True)).cpu()
        mask = to_pil_image(cam.detach().numpy(), mode='F')

        def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
             """Overlay a colormapped mask on a background image

            Args:
                img: background image
                mask: mask to be overlayed in grayscale
                colormap: colormap to be applied on the mask
                alpha: transparency of the background image

            Returns:
                overlayed image
            """

            if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
                raise TypeError('img and mask arguments need to be PIL.Image')

            if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
                raise ValueError('alpha argument is expected to be of type float between 0 and 1')

            cmap = cm.get_cmap(colormap)    
            # Resize mask and apply colormap
            overlay = mask.resize(inputs.size, resample=Image.BICUBIC)
            overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
            # Overlay the image with the mask
            overlayed_img = Image.fromarray((alpha * np.asarray(inputs) + (1 - alpha) * overlay).astype(np.uint8))

            return overlayed_img
        

        for h in range(4):
            # Save overlay image dir
            dir_name = os.path.basename(os.path.dirname(test_images[k]))
            print("dir name:", dir_name)
            input_filename = os.path.basename(test_images[k])
            print("input name:", input_filename)
            save_dir = os.path.join(activation_dir, dir_name, input_filename, f"class_{h}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for j in range(inputs.shape[2]):
                # Select the j-th slice
                slice_img = inputs[0, :, j, :, :].cpu().numpy().transpose(1, 2, 0)  # BDWHC to WHC
                print("slice_img.shape:", slice_img.shape)
                
                # Normalize slice_img to [0, 255]
                min_value = np.percentile(slice_img, 5)
                max_value = np.percentile(slice_img, 95)
                slice_img_normalized = (slice_img - min_value) / (max_value - min_value) * 155
                slice_img_normalized = adjust_brightness_contrast(slice_img_normalized, brightness=50, contrast=50)
                npic_np = slice_img_normalized.astype(np.uint8)

                # Convert numpy array to PIL Image
                slice_img_pil = Image.fromarray(npic_np)
                # Save slice image
                slice_img_pil.save(os.path.join(save_dir, f"{input_filename}_slice_{j}.png"))

                # Apply colormap
                heatmap = cv2.applyColorMap(np.uint8(cam[:, :, j].numpy() * 255), cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                heatmap_rgb_normalized = (heatmap_rgb - heatmap_rgb.min()) / (heatmap_rgb.max() - heatmap_rgb.min()) * 255

                # Overlay heatmap on slice image
                cam_img = 0.6 * npic_np + 0.4 * heatmap_rgb_normalized
                print(cam_img.shape)

                # Convert overlay image to PIL format for saving
                overlay_img = Image.fromarray(np.uint8(cam_img))
                overlay_img.save(os.path.join(save_dir, f"{input_filename}_slice_{j}_overlay.png"))

                # Save the heatmap image
                heatmap_img = Image.fromarray(np.uint8(heatmap_rgb), 'RGB')
                heatmap_img.save(os.path.join(save_dir, f"{input_filename}_slice_{j}_heatmap.png"))


if __name__ == '__main__':

  dataset_dir = code_base + "validation_dataset"
  activation_dir = code_base + "activation"
  run(dataset_dir)