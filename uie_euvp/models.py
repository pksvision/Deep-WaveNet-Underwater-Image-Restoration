import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset


"""# Channel and Spatial Attention"""
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class Conv2D_pxp(nn.Module):

    def __init__(self, in_ch, out_ch, k,s,p):
        super(Conv2D_pxp, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p)
        self.bn = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.PReLU(out_ch)

    def forward(self, input):
        return self.relu(self.bn(self.conv(input)))


class CC_Module(nn.Module):

    def __init__(self):
        super(CC_Module, self).__init__()   

        print("Color correction module for underwater images")

        self.layer1_1 = Conv2D_pxp(1, 32, 3,1,1)
        self.layer1_2 = Conv2D_pxp(1, 32, 5,1,2)
        self.layer1_3 = Conv2D_pxp(1, 32, 7,1,3)

        self.layer2_1 = Conv2D_pxp(96, 32, 3,1,1)
        self.layer2_2 = Conv2D_pxp(96, 32, 5,1,2)
        self.layer2_3 = Conv2D_pxp(96, 32, 7,1,3)
        
        self.local_attn_r = CBAM(64)
        self.local_attn_g = CBAM(64)
        self.local_attn_b = CBAM(64)

        self.layer3_1 = Conv2D_pxp(192, 1, 3,1,1)
        self.layer3_2 = Conv2D_pxp(192, 1, 5,1,2)
        self.layer3_3 = Conv2D_pxp(192, 1, 7,1,3)


        self.d_conv1 = nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.d_bn1 = nn.BatchNorm2d(num_features=32)
        self.d_relu1 = nn.PReLU(32)

        self.global_attn_rgb = CBAM(35)

        self.d_conv2 = nn.ConvTranspose2d(in_channels=35, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.d_bn2 = nn.BatchNorm2d(num_features=3)
        self.d_relu2 = nn.PReLU(3)


    def forward(self, input):
        input_1 = torch.unsqueeze(input[:,0,:,:], dim=1)
        input_2 = torch.unsqueeze(input[:,1,:,:], dim=1)
        input_3 = torch.unsqueeze(input[:,2,:,:], dim=1)
        
        #layer 1
        l1_1=self.layer1_1(input_1) 
        l1_2=self.layer1_2(input_2) 
        l1_3=self.layer1_3(input_3) 

        #Input to layer 2
        input_l2=torch.cat((l1_1,l1_2),1)
        input_l2=torch.cat((input_l2,l1_3),1)
        
        #layer 2
        l2_1=self.layer2_1(input_l2) 
        l2_1 = self.local_attn_r(torch.cat((l2_1, l1_1),1))

        l2_2=self.layer2_2(input_l2) 
        l2_2 = self.local_attn_g(torch.cat((l2_2, l1_2),1))

        l2_3=self.layer2_3(input_l2) 
        l2_3 = self.local_attn_b(torch.cat((l2_3, l1_3),1))
        
        #Input to layer 3
        input_l3=torch.cat((l2_1,l2_2),1)
        input_l3=torch.cat((input_l3,l2_3),1)
        
        #layer 3
        l3_1=self.layer3_1(input_l3) 
        l3_2=self.layer3_2(input_l3) 
        l3_3=self.layer3_3(input_l3) 

        #input to decoder unit
        temp_d1=torch.add(input_1,l3_1)
        temp_d2=torch.add(input_2,l3_2)
        temp_d3=torch.add(input_3,l3_3)

        input_d1=torch.cat((temp_d1,temp_d2),1)
        input_d1=torch.cat((input_d1,temp_d3),1)
        
        #decoder
        output_d1=self.d_relu1(self.d_bn1(self.d_conv1(input_d1)))
        output_d1 = self.global_attn_rgb(torch.cat((output_d1, input_d1),1))
        final_output=self.d_relu2(self.d_bn2(self.d_conv2(output_d1)))
        
        return final_output 
