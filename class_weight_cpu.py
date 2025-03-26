#creat m.zhou@vu.nl,2024.04.26 

import torch
import torch.nn as nn
import numpy as np

def weight_calculation(txt_file):
    sample_num = len(txt_file)
    all_labels = []
    for label in txt_file:
      # Read the contents of the text file
      with open(label, 'r') as file:
        # Read the line and split it into numbers
        numbers = [float(x.strip()) for x in file.readline().split(',')]

        # Append the 1D array to the list
        all_labels.append(numbers)
    # Convert the list of 1D arrays to a 2D array
    two_dimensional_array = np.array(all_labels).reshape(-1,4)
    print("Class Weights all:", two_dimensional_array)
    class_weights = [round((1/sum(column))*(sample_num / 4),3) for column in zip(*two_dimensional_array)]
    return class_weights


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = torch.tensor(weights, dtype = torch.float32)
        # if torch.cuda.is_available():
        #     self.weights = self.weights.cuda()

    def forward(self, output, target):
        # Convert weights to a PyTorch tensor
        # if torch.cuda.is_available():
        #         target = target.cuda()
        # loss = ((output - target) ** 2) * self.weights / self.weights.sum()
        
        return (self.weights*(output - target) ** 2).sum()  / self.weights.sum()
    
# def weighted_mse_loss(output, target, weight):
#     output = torch.tensor(output)
#     target = torch.tensor(target)
#     weight = torch.tensor(weight)

#     if torch.cuda.is_available():
#         output, target, weight = output.cuda(), target.cuda(),weight.cuda()
#     weighted_mse = torch.mean(weight * (output - target) ** 2)

#     return weighted_mse