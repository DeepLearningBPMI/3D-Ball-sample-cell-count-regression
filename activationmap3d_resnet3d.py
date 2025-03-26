import os
import torch
import pathlib
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from preprocess_3d import TiffDataGenerator, resize_volume
from resnet_3d_train import resnet50
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

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

# def create_custom_heatmap(data, bounds, colors):
#     # Normalize data to [0, 1]
#     data_normalized = (data - data.min()) / (data.max() - data.min())
    
#     # Create custom colormap
#     cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=len(bounds) - 1)
#     norm = BoundaryNorm(bounds, cmap.N)
    
#     # # Apply colormap
#     heatmap = cmap(norm(data_normalized))
    
#     # # Convert to 8-bit RGB image
#     heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)


#     # # Create an RGBA image with black and varying alpha
#     # heatmap_rgba = np.zeros((data.shape[0], data.shape[1], 4), dtype=np.float32)
#     # heatmap_rgba[:, :, 3] = data_normalized  # Alpha channel

#     # # Convert to 8-bit RGBA image
#     # heatmap_rgba = (heatmap_rgba * 255).astype(np.uint8)
    
#     return heatmap_rgb

# Custom colormap
# bounds = [0, 0.25, 0.5, 0.75, 1]  # Define the boundaries for each color
# colors = ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']  # Blue, Green, Yellow, Red

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

        def backward_hook(module,grad_in,grad_out):
            grad.append(grad_out[0])

        # Add hook to get the tensors
        model.layer4[-1].register_forward_hook(forward_hook)
        model.layer4[-1].register_full_backward_hook(backward_hook)
    
           
        outputs = model(inputs)
        print ("outputs.shape", outputs.shape) #[1,4] 1 batch, 4 classes, 
  
        criterion = nn.L1Loss()  
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
       

        # get the gradients and activations collected in the hook
        grads=grad[0].cpu().data.numpy().squeeze() # 去掉batch维度,
        fmap=activation[0].cpu().data.numpy().squeeze()
        print("fmap.shape",fmap.shape) #[2024,2,13,13]
        print("grads.shape",grads.shape) #[2048,2,13,13]

        # Get the mean value of the gradients of every featuremap
        # tmp=grads.reshape([grads.shape[0],-1]) #flatten
        weights=np.mean(grads,axis=(1,2,3))  #[2048,]
        print("weights.shape",weights.shape)

        cams = []
        
        for class_index in range(outputs.shape[1]):
            cam = np.zeros(grads.shape[1:])
            output_value = outputs[0, class_index].item()
            print("cam.shape",cam.shape) #[2,13,13]
            for i,w in enumerate(weights):
                cam += w*fmap[i]*output_value
            
            cam=(cam>0)*cam
            print("cam.shape",cam.shape)
            print(cam)
            cam=cam/cam.max()*255
            print(cam)
            print(cam > 255*0.85)
            print("cam.shape",cam.shape)
            # Append CAM for this class
            cams.append(cam.astype(np.uint8))

        cams = np.array(cams)
        print("CAMs shape:", cams.shape) #[4,2,13,13]
        

        # class for neutrophils 0, eosinophils 1, lymphocytes 2, macrophages/monocytes 3
        for h in range (4): 
            #resize cams
            cam_h = cams[h]
            print("cam_h.shape:", cam_h.shape)

            # Adjust the dimensions to match the expected format (N, C, D, H, W)
            cam_h = torch.tensor(cam_h, dtype=torch.float32).unsqueeze(0)  # Add batch and channel dimensions
            cam_h = resize_volume(torch.tensor(cam_h), D, W, H)
            print("cam_h.shape:", cam_h.shape)
            cam_h = cam_h.transpose(1,2,0)
            print("cam_h.shape:", cam_h.shape)

            # Save overlay image dir
            dir_name = os.path.basename(os.path.dirname(test_images[k]))
            print("dir name:", dir_name)
            input_filename = os.path.basename(test_images[k]) 
            print("input name:", input_filename)
            save_dir = os.path.join(activation_dir, dir_name, input_filename, f"class_{h}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Iterate over slices
            for j in range(inputs.shape[2]):
                # Select the j-th slice
                slice_img = inputs[0, :, j, :, :].cpu().numpy().transpose(1, 2, 0) #BCDWH---WHC
                print("slice_img.shape:", slice_img.shape)
               
                # Normalize slice_img to [0, 255]
                slice_img_normalized = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255
                slice_img_normalized = adjust_brightness_contrast(slice_img_normalized, brightness=50, contrast=30)
                npic_np = slice_img_normalized.astype(np.uint8)
                
                # Convert numpy array to PIL Image
                slice_img_pil = Image.fromarray(npic_np)
                
                # Save slice image
                slice_img_pil.save(os.path.join(save_dir, f"{input_filename}_slice_{j}.png"))
                
                # Apply colormap
                heatmap = cv2.applyColorMap(np.uint8(cam_h[:,:,j]).astype(np.uint8), cv2.COLORMAP_JET)
                heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                # Normalize heatmap to [0, 255] if not already normalized
                # heatmap_rgb_normalized = (heatmap_rgb - heatmap_rgb.min()) / (heatmap_rgb.max() - heatmap_rgb.min()) * 255
                # heatmap_rgb = create_custom_heatmap(cam_h[:, :, j], bounds, colors)

                # # Generate custom heatmap with alpha channel
                # heatmap_rgba = create_custom_heatmap(cam_h[:, :, j])
                # # Convert heatmap to RGBA PIL Image
                # heatmap_rgba_pil = Image.fromarray(heatmap_rgba, 'RGBA')
                # # Convert slice image to RGBA PIL Image
                # slice_img_pil_rgba = slice_img_pil.convert("RGBA")

                #  # Overlay heatmap on slice image with transparency
                # overlay_img = Image.alpha_composite(slice_img_pil_rgba, heatmap_rgba_pil)
                # overlay_img.save(os.path.join(save_dir, input_filename + f"slice_{j}_overlay.png"))

                # Overlay heatmap on slice image
                cam_img= 0.6*npic_np+ 0.4*heatmap_rgb
                print(cam_img.shape)

                # Convert overlay image to PIL format for saving
                overlay_img = Image.fromarray(np.uint8(cam_img))
                overlay_img.save(os.path.join(save_dir, input_filename + f"slice_{j}_overlay.png"))

               
                # Create an Image object from the heatmap array
                heatmap_img = Image.fromarray(heatmap_rgb, 'RGB')
                # heatmap_img = Image.fromarray(heatmap_rgba[:, :, :3], 'RGB')
                # heatmap_img = Image.fromarray(heatmap)
                heatmap_img.save(os.path.join(save_dir,input_filename + f"slice_{j}_heatmap.png"))


if __name__ == '__main__':

  dataset_dir = code_base + "visualization"
  activation_dir = code_base + "activation_resnet3d50"
  run(dataset_dir)