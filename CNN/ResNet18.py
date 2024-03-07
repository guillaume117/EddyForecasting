"""
Here we are building the necessary classes for constructing ResNet18 with a hook 
on a middle convolutional layer. The 'Hooked' argument allows switching the network between 
training mode (hooked = False) or evaluation mode (hooked = True). Optionally, one can move the 
layer on which the hook is applied to see what happens in more depth. 
I invite you to have fun experimenting with it.
"""

import torch
import torch.nn as nn


class BasicBlockStd(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockStd, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()



    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #
        out = self.bn2(out)
        
        out += self.shortcut(x)
        out = self.relu(out)

        return out


class BasicBlockFinal(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlockFinal, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
 
   
    
    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)        
        out += self.shortcut(x)

        return out




class ResNet18(nn.Module):
    def __init__(self, num_classes=10,hooked=True,num_depth=1):
        super(ResNet18, self).__init__()
        self.hooked = hooked
        self.conv1 = nn.Conv2d(num_depth, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, 64, 2, stride=1)
        self.layer2 = self.make_layer(64, 128, 2, stride=2)
        self.layer3 = self.make_layer(128, 256, 2, stride=2)
        self.layer4 = self.make_layer(256,256,1,stride=2)
        self.layer5 = self.make_last_layer(256, 512, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.gradients=None
        self.features_conv=None
        self.sfM=nn.Softmax(dim=1)

   

    def make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlockStd(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlockStd(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    
    def make_last_layer(self, in_channels, out_channels, blocks, stride):
        layers = [BasicBlockFinal(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlockFinal(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        if self.hooked == True:
            x.register_hook(self.activations_hook)
            self.features_conv=x

        x = self.layer2(x)
    
        "Here is the hook!!"
        
        x = self.layer3(x)
        
       
       
        x = self.layer4(x)
       
       
        x = self.layer5(x)
    
        
       
    
        x = self.relu(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.sfM(x)

        return x
    
    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
       
      
        return self.features_conv
