import torch
import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
from utils.SuperPoint import SuperPointFrontend
from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import transform_points_DVF
from pytorch_model_summary import summary

image_size = 256
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from networks import affine_network_attention as an
from utils.SuperPoint import SuperPointFrontend
from utils.utils1 import transform_points_DVF

class MultiheadAttention(nn.Module):
    def __init__(self, model_params):
        super(MultiheadAttention, self).__init__()
        


class SP_Attention(nn.Module):
    def __init__(self, model_params):
        super(SP_Attention, self).__init__()

        self.conv1f = 64
        self.conv2f = 128
        self.conv3f = 256
        self.conv4f = 512
        self.conv1 = nn.Conv2d(1, self.conv1f, 3, padding=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(self.conv1f, self.conv2f, 3, padding=1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(self.conv2f, self.conv3f, 3, padding=1, padding_mode='zeros')
        self.conv4 = nn.Conv2d(self.conv3f, self.conv4f, 3, padding=1, padding_mode='zeros')

        self.conv1s = nn.Conv2d(self.conv1f, self.conv1f, 2, stride=2, padding_mode='zeros')
        self.conv2s = nn.Conv2d(self.conv2f, self.conv2f, 2, stride=2, padding_mode='zeros')
        self.conv3s = nn.Conv2d(self.conv3f, self.conv3f, 2, stride=2, padding_mode='zeros')



        self.dropout = nn.Dropout(p=0.7)
        self.aPooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.ReLU = nn.PReLU()
        self.ReLU = nn.LeakyReLU()
        self.Act1 = nn.GroupNorm(int(self.conv1f/2), self.conv1f, eps=1e-05, affine=True)
        self.Act2 = nn.GroupNorm(int(self.conv2f/2), self.conv2f, eps=1e-05, affine=True)
        self.Act3 = nn.GroupNorm(int(self.conv3f/2), self.conv3f, eps=1e-05, affine=True)
        self.Act4 = nn.GroupNorm(int(self.conv4f/2), self.conv4f, eps=1e-05, affine=True)

    def forward(self, source_image, target_image, points):
        # if self.model_params.points == 1:
        affine_params = self.affineNet(source_image, target_image)
        
        # elif self.model_params.heatmaps == 1:
        #     print("This part is not yet implemented.")
            # affine_params = self.affineNet(source_image, target_image, heatmap1, heatmap2)
        transformed_source_image = tensor_affine_transform(source_image, affine_params)
        transformed_points = points.clone()
        transformed_points = transform_points_DVF(transformed_points[0].cpu().detach().T, 
            affine_params.cpu().detach(), transformed_source_image.cpu().detach())

        return transformed_source_image, affine_params, transformed_points.T
    