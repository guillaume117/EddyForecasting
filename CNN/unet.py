import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
import torch.nn.functional as F






class UNet(nn.Module):
    def __init__(self, n_class, verbose= False):
        super().__init__()
        
        
        self.e11 = nn.Conv2d(10, 64, kernel_size=3, padding=1) 
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) 
        self.gnorm_e12 = nn.GroupNorm(32, num_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) 
        self.gnorm_e22 = nn.GroupNorm(32, num_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

 
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.gnorm_e32 = nn.GroupNorm(32, num_channels=256) 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

  
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) 
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 
        self.gnorm_e42 = nn.GroupNorm(32, num_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 

        
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) 
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) 
        self.gnorm_e52 = nn.GroupNorm(32, num_channels=1024)


        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.gnorm_d12 = nn.GroupNorm(32, num_channels=512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.gnorm_d22 = nn.GroupNorm(32, num_channels=256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.gnorm_d32 = nn.GroupNorm(32, num_channels=128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.gnorm_d42 = nn.GroupNorm(32, num_channels=64)
        self.dropout = nn.Dropout(p=0.05)
  
        self.outconv = nn.Conv2d(64, n_class*10, kernel_size=1)
        self.verbose = verbose

        


    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe11 = self.dropout(xe11)
        xe12 = relu(self.e12(xe11))
        xe12 = self.gnorm_e12(xe12)
        xe12 = self.dropout(xe12)    
        xp1  = self.pool1(xe12)
        xe21 = relu(self.e21(xp1))
        xe21 = self.dropout(xe21) 
        xe22 = relu(self.e22(xe21))
        xe22 = self.dropout(xe22)
        xe22 = self.gnorm_e22(xe22)
        xp2  = self.pool2(xe22)
        xe31 = relu(self.e31(xp2))
        xe31 = self.dropout(xe31)
        xe32 = relu(self.e32(xe31))
        xe32 = self.dropout(xe32)
        xe32 = self.gnorm_e32(xe32)
        xp3  = self.pool3(xe32)
        xe41 = relu(self.e41(xp3))
        xe41 = self.dropout(xe41)
        xe42 = relu(self.e42(xe41))
        xe42 = self.dropout(xe42) 
        xe42 = self.gnorm_e42(xe42)
        xe42 = self.dropout(xe42)
        xp4  = self.pool4(xe42)
        xe51 = relu(self.e51(xp4)) 
        xe52 = relu(self.e52(xe51))
        xe52 = self.gnorm_e52(xe52)
        # Decoder
        xu1  = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))
        xd12 = self.gnorm_d12(xd12)
        xu2  = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))
        xd22 = self.gnorm_d22(xd22)
        xu3  = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))
        xd32 = self.gnorm_d32(xd32)
        xu4  = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))
        xd42 = self.gnorm_d42(xd42)
        # Output layer
        out = self.outconv(xd42)
        reshaped_output = out.view(out.size(0), 3, 10, out.size(2), out.size(3))
        softmax_output  = F.softmax(reshaped_output, dim=1)
        out = softmax_output.view_as(out)
        if (self.verbose == True) :
            print("xe11",xe11.shape)
            print("xe12",xe12.shape)
            print("xp1",xp1.shape)
            print("xe21",xe21.shape)
            print("xe22",xe22.shape)
            print("xp2",xp2.shape)
            print("x31",xe31.shape)
            print("xe32",xe32.shape)
            print("xp3",xp3.shape)
            print("xe41",xe41.shape)
            print("xe42",xe42.shape)
            print("xp4",xp4.shape)
            print("xe51",xe51.shape)
            print("xe52",xe52.shape)
            print("xu1",xu1.shape)
            print("cat U1 and e42")
            print("xu11",xu11.shape)
            print("xd11",xd11.shape) 
            print("xd12",xd12.shape)
            print("xu2",xu2.shape)
            print("cat xu2 and xe32")
            print("xu22",xu22.shape)
            print("xd21",xd21.shape)
            print("xd22",xd22.shape)
            print("xu3",xu3.shape)
            print('cat xu3 and xe22')
            print("xu33",xu33.shape)
            print("xd31",xd31.shape)
            print("xd32",xd32.shape)
            print("xu4",xu4.shape)
            print("cat xu4 and xe12")
            print("xu44",xu44.shape)
            print("xd41",xd41.shape)
            print("xd42",xd42.shape)
            print("out shape",out.shape)

        return out