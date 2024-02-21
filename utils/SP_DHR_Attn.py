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



# below is not real attention model
# class SP_DHR_Net(nn.Module):
#     def __init__(self, model_params):
#         super(SP_DHR_Net, self).__init__()
#         self.affineNet = an.load_network(device)
#         self.model_params = model_params
#         print("\nRunning new version (not run SP on source image)")

#         # inputs = torch.rand((1, 1, image_size, image_size)), torch.rand((1, 1, image_size, image_size))
#         # summary(self.affineNet, *inputs, show_input=True, show_hierarchical=True, print_summary=True)

#     def forward(self, source_image, target_image, points):
#         # if self.model_params.points == 1:
#         affine_params = self.affineNet(source_image, target_image)
        
#         # elif self.model_params.heatmaps == 1:
#         #     print("This part is not yet implemented.")
#             # affine_params = self.affineNet(source_image, target_image, heatmap1, heatmap2)
#         transformed_source_image = tensor_affine_transform(source_image, affine_params)
#         transformed_points = points.clone()
#         transformed_points = transform_points_DVF(transformed_points[0].cpu().detach().T, 
#             affine_params.cpu().detach(), transformed_source_image.cpu().detach())

#         return transformed_source_image, affine_params, transformed_points.T
    