import torch
import torch.nn as nn
import torch.nn.functional as F

class Dropconnect(nn.Module):
    def __init__(self, input_dim, output_dim, drop_prob=0.3):
        super(Dropconnect,self).__init__()
        self.drop_prob = drop_prob
        self.weight = nn.Parameter(torch.randn(output_dim, input_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))
    def forward(self, x):
        if self.training:
            #Generate a mask of the same shape as the weights
            mask = torch.rand(self.weight.size())>self.drop_prob
            #Apply Dropconnect: multiply weight by mask
            drop_weight = self.weight * mask.float().to(self.weight.device)
        else:
            #Do not apply Dropconnect when testing, but adjust weights to reflect dropout rates
            drop_weight=self.weight*(1-self.drop_prob)
        return F.linear(x, drop_weight, self.bias)

#使用DropConnect层的示例网络
class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet,self).__init__ ()
        self.dropconnect_layer = Dropconnect(20,10,drop_prob=0.3)
        self.fc2 = nn.Linear(10, 2)
    def forward(self, x):
        x=F.relu(self.dropconnect_layer(x))
        x= self.fc2(x)
        return x

model = ExampleNet()
input = torch.randn(5, 20) #假设的输入
output = model(input)
print(output)