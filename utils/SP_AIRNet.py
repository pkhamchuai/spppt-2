from torchir.networks.globalnet import AIRNet

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

from networks import affine_network_simple_test_size as an
from utils.SuperPoint import SuperPointFrontend

class SP_AIRNet(nn.Module):
    def __init__(self):
        super(SP_AIRNet, self).__init__()
        self.affineNet = AIRNet()
        # self.model_params = model_params

        inputs = torch.rand((1, 1, image_size, image_size)), torch.rand((1, 1, image_size, image_size))
        summary(self.affineNet, *inputs, show_input=True, show_hierarchical=True, print_summary=True)

    def forward(self, source_image, target_image, points):
        # if self.model_params.points == 1:
        translation, rotation, scale, shear = self.affineNet(source_image, target_image)
        
        # create affine transformation matrix from the parameters
        affine_params = create_affine_params(translation, rotation, scale, shear, device)
        # print('affine_params:', affine_params.shape, affine_params)
        # elif self.model_params.heatmaps == 1:
        #     print("This part is not yet implemented.")
            # affine_params = self.affineNet(source_image, target_image, heatmap1, heatmap2)
        transformed_source_image = tensor_affine_transform(source_image, affine_params)
        transformed_points = points.clone()
        # print('transformed_points:', transformed_points.shape)  
        transformed_points = transform_points_DVF(transformed_points[0].cpu().detach().T, 
            affine_params.cpu().detach(), transformed_source_image.cpu().detach())

        return transformed_source_image, affine_params, transformed_points.T
    
def create_affine_params(translation, rotation, scale, shear, device):
    # Create affine transformation matrix from the parameters
    # translation: translation parameters
    # rotation: rotation parameters
    # scale: scale parameters
    # shear: shear parameters
    # device: device to be used
    # returns: affine transformation matrix
    # print('create_affine_params:', translation, rotation, scale, shear, device)
    batch_size = translation.shape[0]
    affine_params = torch.zeros((batch_size, 2, 3), device=device)
    for i in range(batch_size):
        affine_params[i, 0, 0] = scale[i, 0] * torch.cos(rotation[i, 0])
        affine_params[i, 0, 1] = scale[i, 0] * -torch.sin(rotation[i, 0])
        affine_params[i, 0, 2] = translation[i, 0]
        affine_params[i, 1, 0] = scale[i, 1] * torch.sin(rotation[i, 0])
        affine_params[i, 1, 1] = scale[i, 1] * torch.cos(rotation[i, 0])
        affine_params[i, 1, 2] = translation[i, 1]
        # add sheer
        affine_params[i, 0, 1] += shear[i, 1]
        affine_params[i, 1, 0] += shear[i, 0]

    return affine_params