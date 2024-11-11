#'Enhanced' Model - should perform better than the other model - trained for 30 epochs

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleAttention(nn.Module):
    def __init__(self,num_scales):
        super(ScaleAttention,self).__init__()
        self.attention=nn.Sequential(
            nn.Conv2d(num_scales,num_scales,kernel_size=1),
            nn.Softmax(dim=1)  #Apply soft-max across scale
        )

    def forward(self,scale_features):
        stacked_features=torch.stack(scale_features,dim=1)
        attention_input=stacked_features.mean(dim=2)  # Removed keepdim=True
        attention_weights=self.attention(attention_input)
        weighted_features=stacked_features*attention_weights.unsqueeze(2)
        return weighted_features.sum(dim=1) 
    
class ScaleWiseConv(nn.Module):
    def __init__(self,in_channels,out_channels,num_scales=3,kernel_size=3):
        super(ScaleWiseConv, self).__init__()
        self.num_scales=num_scales
        # Convolution layers for each scale
        self.convs=nn.ModuleList([
            nn.Conv2d(in_channels,out_channels,kernel_size,padding=kernel_size//2)
            for _ in range(num_scales)
        ])
        self.norms=nn.ModuleList([nn.InstanceNorm2d([out_channels,256,256]) for _ in range(num_scales)])
        self.attention = ScaleAttention(num_scales)

    def forward(self,x):
        feature_pyramid=[]
        for i in range(self.num_scales):
            # Create progressively smaller versions of the input
            scaled_input=F.interpolate(x,scale_factor=1/(2 ** i),mode='bilinear',align_corners=True)
            conv_out=self.convs[i](scaled_input)
            norm_out=self.norms[i](conv_out)
            upsampled_output=F.interpolate(norm_out,size=x.size()[2:],mode='bilinear',align_corners=True)
            feature_pyramid.append(upsampled_output)
        return self.attention(feature_pyramid)

class ResidualBlock(nn.Module):
    def __init__(self,channels,num_scales=3,kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.scale_conv1=ScaleWiseConv(channels,channels,num_scales,kernel_size)
        self.scale_conv2=ScaleWiseConv(channels,channels,num_scales,kernel_size)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        residual=x
        x=self.scale_conv1(x)
        x=self.relu(x)
        x=self.scale_conv2(x)
        return x+residual

class ScaleWiseNetwork(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,num_features=64,num_blocks=8,num_scales=3,scale_factor=2):
        super(ScaleWiseNetwork, self).__init__()
        # Initializing the convolution layers
        self.initial_conv=nn.Conv2d(in_channels,num_features,kernel_size=3,padding=1)
        self.initial_norm=nn.LayerNorm([num_features,256,256])
        self.blocks=nn.Sequential(*[ResidualBlock(num_features,num_scales) for _ in range(num_blocks)])
        self.final_conv=nn.Conv2d(num_features,out_channels*(scale_factor ** 2),kernel_size=3,padding=1)
        self.pixel_shuffle=nn.PixelShuffle(scale_factor)
    
    def forward(self,x):
        x=self.initial_conv(x)
        x=self.initial_norm(x)
        x=self.blocks(x)
        x=self.final_conv(x)
        x=self.pixel_shuffle(x)
        return x
