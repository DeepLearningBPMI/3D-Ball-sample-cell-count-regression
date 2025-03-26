import os
import torch
import pathlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from preprocess_3d import TiffDataGenerator, resize_volume
from resnet_3d_train_parallel import resnet50
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

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
lr = 0.003
neuro_hidden=32
dropout =0.3
D=40
W=400
H=400

def run(dataset):
    # ------------------------------------- Configuration options -------------------------------------------

    # Get all patient directories
    dirs = os.listdir(dataset)
    test_dirs =  dirs

    # Categorize samples by types
    bal_samples = [d for d in dirs if "BAL" in d and "BALI" not in d]
    bali_samples = [d for d in dirs if "BALI" in d and "Hep" not in d]
    hep_samples = [d for d in dirs if "Hep" in d]

    # Combine remaining samples
    all_samples = bali_samples+bali_samples
    test_dirs = all_samples

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
    model = resnet50(D, W, H, neuro_hidden, dropout)
    model.load_state_dict(torch.load('/scistor/guest/mzu240/BAL/Results_ResNet_BAL_1115_rareintraining/11-15 14:57/resnet3d50_BAL_lr0.0012_dropout0.3_SGD_neurons_hidden32/model.pth'))
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
        activation=[]
        grad=[]     
        def forward_hook(module,input,output):
            activation.append(output)

        # Choose a layer to capture activations from
        layer = model.layer4[-1]  # Example, adjust based on your model architecture

        # Register the hook
        hook = layer.register_forward_hook(forward_hook)
        # Ensure your model is in evaluation mode
        model.eval()
      
         # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            print ("outputs.shape", outputs.shape) #[1,4] 1 batch, 4 classes, 

        # Extract activations from the hook
        fmap = activation[0].cpu().data.numpy().squeeze()  # Remove batch dimension
        print("fmap shape:", fmap.shape)  #[2048,2,13,13]

        def backward_hook(module,grad_in,grad_out):
            grad.append(grad_out[0])
    
        # Register the backward hook
        hook = layer.register_full_backward_hook(backward_hook)

        # Compute gradients for specific outputs
        model.zero_grad()
        # Forward pass
        outputs = model(inputs)
    
        output_index = 0   # class for neutrophils 0, eosinophils 1, lymphocytes 2, macrophages/monocytes 3
        
        outputs[0, output_index].backward(retain_graph=True)
        
        # Get gradients
        grads = grad[0].cpu().data.numpy().squeeze()
        print("grads.shape",grads.shape) #[2048,2,13,13]
        weights = np.mean(grads, axis=(1, 2, 3)) #[2048,]

        # Generate CAM
        cam = np.zeros(grads.shape[1:])
        print("cam.shape",cam.shape) #[2,13,13]
        for i,w in enumerate(weights):
            cam += w*fmap[i]
        
        cam=(cam>0)*cam
        print("cam.shape",cam.shape) 
        print(cam)
        cam=cam/cam.max()*255
        print(cam)
        print(cam > 255*0.85)
        print("cam.shape",cam.shape) #[2,13,13]


        # Adjust the dimensions to match the expected format (N, C, D, H, W)
        cam_h = torch.tensor(cam, dtype=torch.float32).unsqueeze(0)  # Add batch and channel dimensions
        cam_h = resize_volume(torch.tensor(cam_h), D, W, H)
        print("cam_h.shape:", cam_h.shape)
        cam_h = cam_h.transpose(1,2,0)
        print("cam_h.shape:", cam_h.shape)

        # Save overlay image dir
        dir_name = os.path.basename(os.path.dirname(test_images[k]))
        print("dir name:", dir_name)
        input_filename = os.path.basename(test_images[k]) 
        print("input name:", input_filename)
        save_dir = os.path.join(activation_dir, dir_name, input_filename, f"class_{output_index}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Iterate over slices
        for j in range(inputs.shape[2]):
            # Select the j-th slice
            slice_img = inputs[0, :, j, :, :].cpu().numpy().transpose(1, 2, 0) #BCDWH---WHC
            print("slice_img.shape:", slice_img.shape)
            
            # Normalize slice_img to [0, 1]
            slice_img_normalized = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())
            npic_np = np.float32(slice_img_normalized)
            
    
            # Apply colormap
            heatmap =  cam_h[:,:,j]
            heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) 
            cam_image=show_cam_on_image(npic_np, heatmap_normalized)

            # Convert overlay image to PIL format for saving
            overlay_img = Image.fromarray(np.uint8(cam_image))
            overlay_img.save(os.path.join(save_dir, input_filename + f"slice_{j}_overlay.png"))
  
        # # Iterate over slices
        # for j in range(inputs.shape[2]):
        #     # Select the j-th slice
        #     slice_img = inputs[0, :, j, :, :].cpu().numpy().transpose(1, 2, 0) #BCDWH---WHC
        #     print("slice_img.shape:", slice_img.shape)
            
        #     # Normalize slice_img to [0, 255]
        #     slice_img_normalized = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255
        #     slice_img_normalized = adjust_brightness_contrast(slice_img_normalized, brightness=50, contrast=30)
        #     npic_np = slice_img_normalized.astype(np.uint8)
            
        #     # # Convert numpy array to PIL Image
        #     # slice_img_pil = Image.fromarray(npic_np)
            
        #     # # Save slice image
        #     # slice_img_pil.save(os.path.join(save_dir, f"{input_filename}_slice_{j}.png"))
            
        #     # Apply colormap
        #     heatmap = cv2.applyColorMap(np.uint8(cam_h[:,:,j]).astype(np.uint8), cv2.COLORMAP_JET)
        #     heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
        #     # Overlay heatmap on slice image
        #     cam_img= 0.6*npic_np+ 0.4*heatmap_rgb
        #     print(cam_img.shape)

        #     # Convert overlay image to PIL format for saving
        #     overlay_img = Image.fromarray(np.uint8(cam_img))
        #     overlay_img.save(os.path.join(save_dir, input_filename + f"slice_{j}_overlay.png"))

            
        #     # Create an Image object from the heatmap array
        #     heatmap_img = Image.fromarray(heatmap_rgb, 'RGB')
        #     # heatmap_img = Image.fromarray(heatmap_rgba[:, :, :3], 'RGB')
        #     # heatmap_img = Image.fromarray(heatmap)
        #     heatmap_img.save(os.path.join(save_dir,input_filename + f"slice_{j}_heatmap.png"))


if __name__ == '__main__':

  dataset_dir = code_base + "dataset_test"
  activation_dir = code_base + "activation_resnet_1115_bal"
  run(dataset_dir)