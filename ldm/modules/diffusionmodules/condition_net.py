import torch
import torch.nn as nn
import torch.nn.functional as F
from ldm.modules.diffusionmodules.util import FourierEmbedder

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    zero_module,
    normalization,
)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, down, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        if in_c != out_c or sk == False:
            self.in_conv = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.in_conv = None
        
        self.norm = normalization(out_c)
        
        self.body = nn.Sequential(
            conv_nd(2, out_c, out_c, 3, padding=1),
            nn.SiLU(),
            conv_nd(2, out_c, out_c, ksize, padding=0),
        )
        if sk == False:
            self.skep = nn.Conv2d(in_c, out_c, ksize, 1, ps)
        else:
            self.skep = None

        self.down = down
        if self.down == True:
            self.down_opt = Downsample(in_c, use_conv=use_conv)

    def forward(self, x):
        if self.down == True:
            x = self.down_opt(x)
        if self.in_conv is not None:  # edit
            x = self.in_conv(x)

        x = self.norm(x)
        h = self.body(x)
        if self.skep is not None:
            return h + self.skep(x)
        else:
            return h + x


class ImageConditionNet(nn.Module):
    def __init__(self, autoencoder=None, channels=[320, 640, 1280, 1280], nums_rb=3, cin=4, ksize=3, sk=False, use_conv=True):
        super(ImageConditionNet, self).__init__()
        self.autoencoder = autoencoder
        self.set = False
        
        self.channels = channels
        self.nums_rb = nums_rb

        self.in_layers = nn.Sequential(
            nn.InstanceNorm2d(cin, affine=True),
            conv_nd(2, cin, channels[0], 3, padding=1),
        )
        
        self.body = []
        self.out_convs = []
        for i in range(len(channels)):
            for j in range(nums_rb):
                if (i != 0) and (j == 0):
                    self.body.append(
                        ResnetBlock(channels[i - 1], channels[i], down=True, ksize=ksize, sk=sk, use_conv=use_conv))
                else:
                    self.body.append(
                        ResnetBlock(channels[i], channels[i], down=False, ksize=ksize, sk=sk, use_conv=use_conv))
            self.out_convs.append(
                zero_module(
                    conv_nd(2, channels[i], channels[i], 1, padding=0, bias=False)
                )
            )
    
        self.body = nn.ModuleList(self.body)
        self.out_convs = nn.ModuleList(self.out_convs)

    def forward(self, input_dict, h=None):
        assert self.set
        
        x, masks = input_dict['values'], input_dict['masks']
        
        with torch.no_grad():
            x = self.autoencoder.encode(x)

        # extract features
        features = []

        x = self.in_layers(x)

        for i in range(len(self.channels)):
            for j in range(self.nums_rb):
                idx = i * self.nums_rb + j
                x = self.body[idx](x)
            x = self.out_convs[i](x)
            features.append(x)


        return features


class AutoencoderKLWrapper(nn.Module):
    def __init__(self, autoencoder=None):
        super().__init__()
        self.autoencoder = autoencoder
        self.set = False

    def forward(self, input_dict):
        assert self.set
        x = input_dict['values']
        with torch.no_grad():
            x = self.autoencoder.encode(x)
        return x
        

class NSPVectorConditionNet(nn.Module):
    def __init__(self, in_dim, out_dim, norm=False, fourier_freqs=0, temperature=100, scale=1.0):
        super().__init__()
        self.in_dim = in_dim  
        self.out_dim = out_dim
        self.norm = norm
        self.fourier_freqs = fourier_freqs

        if fourier_freqs > 0:
            self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs, temperature=temperature)
            self.in_dim *= (fourier_freqs * 2)
        if self.norm:
            self.linears = nn.Sequential(
                                nn.Linear(self.in_dim, 512),
                                nn.LayerNorm(512),
                                nn.SiLU(),
                                nn.Linear(512, 512),
                                nn.LayerNorm(512),
                                nn.SiLU(),
                                nn.Linear(512, out_dim),
                                )
        else: 
            self.linears = nn.Sequential(
                                nn.Linear(self.in_dim, 512),
                                nn.SiLU(),
                                nn.Linear(512, 512),
                                nn.SiLU(),
                                nn.Linear(512, out_dim),
                                )

        self.null_features = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.scale = scale

    def forward(self, input_dict):
        vectors, masks = input_dict['values'], input_dict['masks']  # vectors: B*C, masks: B*1
        if self.fourier_freqs > 0:
            vectors = self.fourier_embedder(vectors * self.scale) 
        objs = masks * vectors + (1-masks) * self.null_features.view(1,-1)
        return self.linears(objs).unsqueeze(1)  # B*1*C


class BoxConditionNet(nn.Module):
    def __init__(self, in_dim, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*4 # 2 is sin&cos, 4 is xyxy 

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.null_text_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, input_dict):
        boxes, masks, text_embeddings = input_dict['values'], input_dict['masks'], input_dict['text_embeddings']

        B, N, _ = boxes.shape 
        masks = masks.unsqueeze(-1)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes) # B*N*4 --> B*N*C

        # learnable null embedding 
        text_null = self.null_text_feature.view(1,1,-1)
        xyxy_null =  self.null_position_feature.view(1,1,-1)

        # replace padding with learnable null embedding 
        text_embeddings = text_embeddings*masks + (1-masks)*text_null
        xyxy_embedding = xyxy_embedding*masks + (1-masks)*xyxy_null

        objs = self.linears(  torch.cat([text_embeddings, xyxy_embedding], dim=-1)  )
        assert objs.shape == torch.Size([B,N,self.out_dim])        
        return objs


class KeypointConditionNet(nn.Module):
    def __init__(self, max_persons_per_image, out_dim, fourier_freqs=8):
        super().__init__()
        self.max_persons_per_image = max_persons_per_image
        self.out_dim = out_dim

        self.person_embeddings   = torch.nn.Parameter(torch.zeros([max_persons_per_image,out_dim]))
        self.keypoint_embeddings = torch.nn.Parameter(torch.zeros([17,out_dim]))
         

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*2 # 2 is sin&cos, 2 is xy 

        self.linears = nn.Sequential(
            nn.Linear(self.out_dim + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )
        
        self.null_person_feature = torch.nn.Parameter(torch.zeros([self.out_dim]))
        self.null_xy_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
  

    def forward(self, input_dict):
        points, masks = input_dict['values'], input_dict['masks']
        
        masks = masks.unsqueeze(-1)
        N = points.shape[0]

        person_embeddings = self.person_embeddings.unsqueeze(1).repeat(1,17,1).reshape(self.max_persons_per_image*17, self.out_dim)
        keypoint_embeddings = torch.cat([self.keypoint_embeddings]*self.max_persons_per_image, dim=0)
        person_embeddings = person_embeddings + keypoint_embeddings # (num_person*17) * C 
        person_embeddings = person_embeddings.unsqueeze(0).repeat(N,1,1)

        # embedding position (it may includes padding as placeholder)
        xy_embedding = self.fourier_embedder(points) # B*N*2 --> B*N*C

        
        # learnable null embedding 
        person_null = self.null_person_feature.view(1,1,-1)
        xy_null =  self.null_xy_feature.view(1,1,-1)

        # replace padding with learnable null embedding 
        person_embeddings = person_embeddings*masks + (1-masks)*person_null
        xy_embedding = xy_embedding*masks + (1-masks)*xy_null

        objs = self.linears(  torch.cat([person_embeddings, xy_embedding], dim=-1)  )
      
        return objs

