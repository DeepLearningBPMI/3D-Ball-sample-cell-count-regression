import torch
import numpy as np
import torch.nn as nn
import tifffile
import os
import cv2
from resnet_3d_train import resnet50
from preprocess_3d import apply_clahe_3d, normalize_3d_image, resize_volume,  adjust_brightness_contrast
from PIL import Image
import matplotlib as plt

# ------------------------------------ step 1/5 : premeters seeting-----------------------------------
dropout =0.295
D=62
W=400
H=400
class_ = {0: 'N.', 1:'E.', 2: 'L.', 3:'M.'}
code_base = '/scistor/guest/mzu240/BAL/'
activation_dir = code_base + "activation_resnet3d50NEW"

# ------------------------------------ step 2/5 : load data-----------------------------------
image_path = "/scistor/guest/mzu240/BAL/visualization/2024.01.16-BALI006-Hep006/1644 to 1674-20slices.tif"
image = tifffile.imread(image_path)
print("Image shape:", image.shape) #DWHC
image = apply_clahe_3d(image=image)    
image = torch.tensor(image, dtype=torch.float32)  # Convert to tensor
image = normalize_3d_image(image)
image = np.moveaxis(image.numpy(), -1, 0)  # Adjust axes as needed CDWH
print("Image shape:", image.shape)
image = resize_volume(image, D, W, H)  # Implement your resize_volume function
print(" Resized Image shape:", image.shape)
# Convert to PyTorch format
input = torch.tensor(image, dtype=torch.float32)
inputs = input.unsqueeze(0) # add batch dimention BCDWH
print("INPUTS shape torchtransform:", inputs.shape)
inputs= inputs.cuda()

# ------------------------------------ step 3/5 : load model and model eval------------------------------------
cam_clas = [] #store the CAM of all classes
for cla in range (4):
    model = resnet50(D, W, H, dropout)
    model.load_state_dict(torch.load('/scistor/guest/mzu240/BAL/Results_3D/07-03 21:17/resnet3d50_5stackspercase_lr0.00117_dropout0.295_SGD_32neurons/model.pth'))
    model = model.to('cuda')
    print("model:", model)
    model.eval()
    activation=[]
    grad=[]     
    def forward_hook(module,input,output):
        activation.append(output)

    # Choose a layer to capture activations from (model_features)
    layer = model.layer4[-1]  

    # Register the hook
    hook = layer.register_forward_hook(forward_hook)

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

    # ------------------------------------ step 4/5 : get CAM------------------------------------
    # class for neutrophils 0, eosinophils 1, lymphocytes 2, macrophages/monocytes 3
    outputs[0, cla].backward(retain_graph=True)
    # Get gradients
    grads = grad[0].cpu().data.numpy().squeeze()
    print("grads.shape",grads.shape) #[2048,2,13,13]
    weights = np.mean(grads, axis=(1, 2, 3)) #[2048,]
    cam = np.zeros(grads.shape[1:])
    print("cam.shape",cam.shape) #[2,13,13]
    for i,w in enumerate(weights):
        cam += w*fmap[i]
    cam=(cam>0)*cam 
    cam=cam/cam.max()*255
    print(cam > 255*0.85)
    print("cam.shape",cam.shape) #[2,13,13]

    # Adjust the dimensions to match the expected format (N, C, D, H, W)
    cam_h = torch.tensor(cam, dtype=torch.float32).unsqueeze(0)  # Add batch and channel dimensions
    cam_h = resize_volume(torch.tensor(cam_h), D, W, H)
    print("cam_h.shape:", cam_h.shape)
    cam_h = cam_h.transpose(1,2,0) #WHD
    print("cam_h.shape:", cam_h.shape)
    
    # ------------------------------------ step 5/5 : Saving------------------------------------
    for j in range(inputs.shape[2]):

        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        save_dir = os.path.join(activation_dir, image_filename, f"class_{cla}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Select the j-th slice
        slice_img = inputs[0, :, j, :, :].cpu().numpy().transpose(1, 2, 0) #BCDWH---WHC
        # print("slice_img.shape:", slice_img.shape)
        
        slice_img_normalized = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min()) * 255
        slice_img_normalized = adjust_brightness_contrast(slice_img_normalized, brightness=50, contrast=30)
        npic_np = slice_img_normalized.astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(cam_h[:,:,j]), cv2.COLORMAP_JET)
        heatmap_rgb= cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay heatmap on slice image
        cam_img= 0.6*npic_np+ 0.4*heatmap_rgb
        #print(cam_img.shape) #WHC

        # Convert overlay image to PIL format for saving
        overlay_img = Image.fromarray(np.uint8(cam_img))
        overlay_img.save(os.path.join(save_dir, f"slice_{j}_overlay.png"))
       
        if  j == 30:
            middleslice = npic_np
            img = cv2.imread(os.path.join(save_dir, f"slice_{j}_overlay.png"))
            text = '%s %.2f%% %s %.2f%% %s %.2f%% %s %.2f%%' % (class_[0], outputs[0, 0].item()*100, class_[1], outputs[0, 1].item()*100, class_[2], outputs[0, 2].item()*100, class_[3], outputs[0, 3].item()*100) 
            cv2.putText(img, text, (10,40), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = (255, 255, 255), thickness =2, lineType = cv2.LINE_AA)
            cv2.imwrite(os.path.join(save_dir, f"slice_{j}_results.png"), img)

        # Create an Image object from the heatmap array
        heatmap_img = Image.fromarray(heatmap_rgb, 'RGB')
        # heatmap_img = Image.fromarray(heatmap_rgba[:, :, :3], 'RGB')
        # heatmap_img = Image.fromarray(heatmap)
        heatmap_img.save(os.path.join(save_dir, f"slice_{j}_heatmap.png"))
    
    cam_cla = cam_h[:,:,30]
    print("cam_cla.shape:", cam_cla.shape)
    cam_clas.append(cam_cla)

cam_clas_stacked = np.stack(cam_clas,axis =0)
print("cam_clas_stacked.shape:", cam_clas_stacked.shape)
# Normalize each activation map to [0, 1]
cam_clas_stacked = (cam_clas_stacked - cam_clas_stacked.min(axis=(1, 2), keepdims=True)) / (
    cam_clas_stacked.max(axis=(1, 2), keepdims=True) - cam_clas_stacked.min(axis=(1, 2), keepdims=True)
)
print("cam_clas_stacked.shape:", cam_clas_stacked.shape)

# Initialize an empty image to store the combined heatmap
combined_heatmap = np.zeros((H, W, 3))

# Define colors for each class (red, green, blue, and yellow)
colors = [
    [1, 0, 0],  # Red for class 0
    [0, 1, 0],  # Green for class 1
    [0, 0, 1],  # Blue for class 2
    [1, 1, 0],  # Yellow for class 3
]

# Combine each normalized activation map into the combined heatmap
for i in range(4):
    # Convert each heatmap to RGB format
    heatmap_rgb = np.zeros((H, W, 3))
    for c in range(3):
        heatmap_rgb[:, :, c] = cam_clas_stacked[i] * colors[i][c]
    
    # Overlay the heatmap onto the combined_heatmap
    combined_heatmap += heatmap_rgb
print("combined_heatmap .shape:", combined_heatmap .shape)

# Clip values to ensure they are within [0, 1] range
combined_heatmap = np.clip(combined_heatmap, 0, 1)
combined_heatmap_255 = (combined_heatmap * 255).astype(np.uint8)
# Create an overlay image with the heatmap and original image
overlay_image = 0.6* middleslice +0.4* combined_heatmap_255
overlay_imgage = Image.fromarray(np.uint8(overlay_image))
results_dir = os.path.join(activation_dir, image_filename)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
overlay_img.save(os.path.join(results_dir, f"midlleslice_resluts.png"))
img1 = cv2.imread(os.path.join(results_dir, f"midlleslice_resluts.png"))
text = '%s %.2f%% %s %.2f%% %s %.2f%% %s %.2f%%' % (class_[0], outputs[0, 0].item()*100, class_[1], outputs[0, 1].item()*100, class_[2], outputs[0, 2].item()*100, class_[3], outputs[0, 3].item()*100) 
cv2.putText(img, text, (10,40), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = (255, 255, 255), thickness =2, lineType = cv2.LINE_AA)
results_dir = os.path.join(activation_dir, image_filename)
cv2.imwrite(os.path.join(results_dir, f"midlleslice_resluts.png"), img1)

