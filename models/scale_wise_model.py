#'Original' model, is simpler than the adpative model. Use this for faster training, although better results may not be obtained. trained to 20 epochs - change if needed
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleWiseConv(nn.Module):
    def __init__(self,in_channels,out_channels,num_scales=3,kernel_size=3):
        super(ScaleWiseConv,self).__init__()
        self.num_scales=num_scales
        self.convs=nn.ModuleList([
            nn.Conv2d(in_channels,out_channels,kernel_size,padding=kernel_size//2)
            for _ in range(num_scales)
        ])
        #self.scale_weights = nn.Parameter(torch.ones(num_scales)) #for weighted fp
    def forward(self,x):
        feature_pyramid=[]
        input_scale=x
        for i in range(self.num_scales):
            scaled_input=F.interpolate(input_scale,scale_factor=1/(2**i),mode='bilinear',align_corners=True)
            conv_out=self.convs[i](scaled_input)
            upsampled_output=F.interpolate(conv_out,size=x.size()[2:],mode='bilinear',align_corners=True)
            feature_pyramid.append(upsampled_output)
        return sum(feature_pyramid)
    
class ResidualBlock(nn.Module):
    def __init__(self,channels,num_scales=3,kernel_size=3):
        super(ResidualBlock,self).__init__()
        self.scale_conv=ScaleWiseConv(channels,channels,num_scales,kernel_size)
        self.relu=nn.ReLU(inplace=True)
    
    def forward(self,x):
        residual=x
        x=self.scale_conv(x)
        x=self.relu(x)
        x=self.scale_conv(x)
        return x+residual
    
class ScaleWiseNetwork(nn.Module):
    def __init__(self,in_channels=3,out_channels=3,num_features=64,num_blocks=8,num_scales=3,scale_factor=2):
        super(ScaleWiseNetwork, self).__init__()
        self.initial_conv=nn.Conv2d(in_channels,num_features,kernel_size=3,padding=1)
        self.blocks=nn.Sequential(*[ResidualBlock(num_features,num_scales) for _ in range(num_blocks)])
        self.final_conv=nn.Conv2d(num_features,out_channels*(scale_factor ** 2),kernel_size=3,padding=1)
        self.pixel_shuffle=nn.PixelShuffle(scale_factor)
    
    def forward(self,x):
        x=self.initial_conv(x)
        x=self.blocks(x)
        x=self.final_conv(x)
        x=self.pixel_shuffle(x)
        return x