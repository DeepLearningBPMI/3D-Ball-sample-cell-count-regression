import os
import torch
import pathlib
import numpy as np
import torch.nn as nn
import tifffile
from torch.autograd import Function
from PIL import Image
import cv2
from preprocess_3d import TiffDataGenerator, resize_volume
from resnet_3d_train import resnet50
from torch.nn.functional import interpolate
from torchvision import models


class GradCam3d(Function):
    def __init__(self, model, target_layer):
        super(GradCam3d, self).__init__ ()
        self.model= model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradient = None
        
    def forward(self,x):
        self.feature_maps =[]
        self.gradient =[]
        for module_pos, module in self.model.module.named_children():
            x= module(x)
            if module_pos == self.target_layer:
                x.register_hook(self.save_gradient)
                self.feature_maps.append(x)
        return x
    def backward(self, grad_output):
        if grad_output.is_cuda:
            grad_output = grad_output.cpu()

        grad_output= grad_output.data.numpy()[0]
        self.feature_maps[-1].grad = torch.from_numpy(grad_output)
        for i in range(len(self.feature_maps)-1,0,-1):
            x= self.feature_maps[i]
            if x.grad is not None:
                continue

            x.retain_grad()
            y=self.feature_maps[i-1]
            if y.grad is not None:
                continue

            y.retain_grad()
            x_size = x.size()
            y_size = y.size()
            if x_size[2]!=y_size[2]or x_size[3]!=y_size[3]or x_size[4]!= y_size[4]:
                y= interpolate(y, size=(x_size[2],x_size[3],x_size[4]),mode='trilinear', align_corners=False)

            x.backward(retain_graph=True)
            grad = y.grad
            if grad is not None:
                grad = grad.data.numpy()[0]
                self.gradient.append(grad)
        return self.gradient
    
    def save_gradient(self, grad):
            if grad.is_cuda:
                grad = grad.cpu()
            self.gradient.append(grad.data.numpy())
def get_heatmap(gradient_maps):
    heatmap =np.sum(gradient_maps,axis=0)
    heatmap =np.where(heatmap >0, heatmap,0)
    heatmap_max = np.max(heatmap)
    if heatmap_max != 0:
        heatmap/=heatmap_max
    return heatmap

def grad_cam_3d(model, x, target_layer):
    model.eval()
    grad_cam=GradCam3d(model,target_layer)
    output = model(x)
    output =output.cpu().data.numpy()[0]
    output_index= np.argmax(output)
    grad_maps = grad_cam(torch.FloatTensor(x)).backward(torch.FloatTensor([1.0]))
    heatmap = get_heatmap(grad_maps)
    return heatmap, output_index

# 加载模型
model = models.resnet50(pretrained=True)
model.fc =torch.nn.Sequential(torch.nn.Linear(2048, 2))
model.cuda()
model.eval()

# 加载数据
image = tifffile.imread("/scistor/guest/mzu240/BAL/visualization/2023.12.12-BAL007/208 to 238-17slices.tif")
x = np.array(image)
x=torch.from_numpy(x).unsqueeze(0).float()
x= x.cuda()
# 进行Grad-CAM可视化
heatmap,output_index=grad_cam_3d(model,x,'layer4')



# code_base = '/scistor/guest/mzu240/BAL/'
# _DATA_SELECTOR = '*.tif'
# classes_name = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Macrophages']
# batch_size = 1
# epochs = 1
# lr = 0.00117
# dropout = 0.295
# D = 62
# W = 400
# H = 400

# def run(dataset):
#     # ------------------------------------- Configuration options -------------------------------------------

#     # Get all patient directories
#     dirs = os.listdir(dataset)
#     test_dirs = dirs

#     # Start print
#     print('--------------------------------')
#     test_folders = test_dirs

#     # ------------------------------------- step 1/5 : Loading data -------------------------------------------
#     test_images = [str(img) for folder in test_folders for img in pathlib.Path(os.path.join(dataset, folder)).rglob(_DATA_SELECTOR)]
#     print(f"Total test images: {len(test_images)}") 
#     test_labels = [str(pathlib.Path(case).parents[0] / 'labels.txt') for case in test_images]
#     print(f"Total test labels: {len(test_labels)}")

#     test_generator = TiffDataGenerator(test_images, test_labels, D, W, H, batch_size, clahe=True, augmentations=False)  
#     print("Total batches in test_generator:", len(test_generator))

#     # ------------------------------------ step 2/5 : load model------------------------------------
#     model = resnet50(D, W, H, dropout)
#     model.load_state_dict(torch.load('/scistor/guest/mzu240/BAL/Results_3D/07-03 21:17/resnet3d50_5stackspercase_lr0.00117_dropout0.295_SGD_32neurons/model.pth'))
#     model = model.to('cuda')
#     print("model:", model)

#     k = -1
#     # ------------------------------------ step 4/5 : activation map --------------------------------------------------  
#     for batch in test_generator:
#         # Check if the batch is None (no more data)
#         if batch is None:
#             break

#         inputs, labels = batch
#         print("test batch - Inputs shape:", inputs.shape, "Labels shape:", labels.shape)

#         # Transfer Data to GPU if available
#         if torch.cuda.is_available():
#             inputs, labels = inputs.cuda(), labels.cuda()

#         k += 1
#         activation = []
#         grad = []     
        
#         def forward_hook(module, input, output):
#             activation.append(output)

#         def backward_hook(module, grad_in, grad_out):
#             grad.append(grad_out[0])

#         # Add hook to get the tensors
#         model.layer4[-1].register_forward_hook(forward_hook)
#         model.layer4[-1].register_full_backward_hook(backward_hook)
    
#         outputs = model(inputs)
#         print("outputs.shape", outputs.shape)
  
#         criterion = nn.L1Loss()  
#         loss = criterion(outputs, labels)
#         model.zero_grad()
#         loss.backward()

#         # get the gradients and activations collected in the hook
#         grads = grad[0].cpu().data.numpy().squeeze()  # Remove batch dimension
#         fmap = activation[0].cpu().data.numpy().squeeze()
#         print("fmap.shape", fmap.shape)
#         print("grads.shape", grads.shape)

#         # Get the mean value of the gradients of every featuremap
#         weights = np.mean(grads, axis=(1, 2, 3)) 
#         print("weights.shape", weights.shape)

#         cams = []
        
#         for class_index in range(outputs.shape[1]):
#             cam = np.zeros(grads.shape[1:])
#             print("cam.shape", cam.shape)
#             for i, w in enumerate(weights):
#                 cam += w * fmap[i] * outputs[0, class_index].item()  
            
#             cam = (cam > 0) * cam
#             print("cam.shape", cam.shape)
#             print(cam)
#             cam = cam / cam.max() * 255
#             print(cam)
#             print(cam > 255 * 0.85)
#             print("cam.shape", cam.shape)
#             # Append CAM for this class
#             cams.append(cam.astype(np.uint8))

#         cams = np.array(cams)
#         print("CAMs shape:", cams.shape)

#         # Class for neutrophils 0, eosinophils 1, lymphocytes 2, macrophages/monocytes 3
#         for h in range(4): 
#             cam_h = cams[h]
#             print("cam_h.shape:", cam_h.shape)

#             # Adjust the dimensions to match the expected format (N, C, D, H, W)
#             cam_h = torch.tensor(cam_h, dtype=torch.float32).unsqueeze(0)  # Add batch and channel dimensions
#             cam_h = resize_volume(torch.tensor(cam_h), D, W, H)
#             print("cam_h.shape:", cam_h.shape)
#             cam_h = cam_h.transpose(1, 2, 0)
#             print("cam_h.shape:", cam_h.shape)

#             # Save overlay image dir
#             dir_name = os.path.basename(os.path.dirname(test_images[k]))
#             print("dir name:", dir_name)
#             input_filename = os.path.basename(test_images[k]) 
#             print("input name:", input_filename)
#             save_dir = os.path.join(activation_dir, dir_name, input_filename, f"class_{h}")
#             if not os.path.exists(save_dir):
#                 os.makedirs(save_dir)

#             # Iterate over slices
#             for j in range(inputs.shape[2]):
#                 # Select the j-th slice
#                 slice_img = inputs[0, :, j, :, :].cpu().numpy().transpose(1, 2, 0)  # BCDWH---WHC
#                 print("slice_img.shape:", slice_img.shape)
#                 # Calculate the minimum and maximum values for normalization
#                 min_value = np.percentile(slice_img, 5)  # 5th percentile as the minimum value
#                 max_value = np.percentile(slice_img, 95)  # 95th percentile as the maximum value
#                 # Normalize slice_img to [0, 200]
#                 slice_img_normalized = (slice_img - min_value) / (max_value - min_value) * 155

#                 # Normalize slice_img to [0, 255]
#                 slice_img_normalized = adjust_brightness_contrast(slice_img_normalized, brightness=50, contrast=50)
#                 npic_np = slice_img_normalized.astype(np.uint8)
                
#                 # Convert numpy array to PIL Image
#                 slice_img_pil = Image.fromarray(npic_np)
                
#                 # Save slice image
#                 slice_img_pil.save(os.path.join(save_dir, f"{input_filename}_slice_{j}.png"))
                
#                 # Apply colormap
#                 heatmap = cv2.applyColorMap(np.uint8(cam_h[:, :, j]).astype(np.uint8), cv2.COLORMAP_JET)
#                 heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#                 # Normalize heatmap to [0, 255] if not already normalized
#                 heatmap_rgb_normalized = (heatmap_rgb - heatmap_rgb.min()) / (heatmap_rgb.max() - heatmap_rgb.min()) * 255
               
#                 # Overlay heatmap on slice image
#                 cam_img = 0.6 * npic_np + 0.4 * heatmap_rgb
#                 print(cam_img.shape)

#                 # Convert overlay image to PIL format for saving
#                 overlay_img = Image.fromarray(np.uint8(cam_img))
#                 overlay_img.save(os.path.join(save_dir, input_filename + f"_slice_{j}_overlay.png"))

#                 # Create an Image object from the heatmap array
#                 heatmap_img = Image.fromarray(heatmap_rgb, 'RGB')
#                 heatmap_img.save(os.path.join(save_dir, input_filename + f"_slice_{j}_heatmap.png"))

# if __name__ == '__main__':
#     dataset_dir = code_base + "validation_dataset"
#     activation_dir = code_base + "activation"
#     run(dataset_dir)
