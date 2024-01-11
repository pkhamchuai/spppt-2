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

# from networks import affine_network_simple as an
from utils.SuperPoint import SuperPointFrontend
from utils.utils1 import transform_points_DVF

class Affine_Network(nn.Module):
    def __init__(self, device):
        super(Affine_Network, self).__init__()
        self.device = device

        self.feature_extractor = Feature_Extractor(self.device)
        self.regression_network = Regression_Network()

    def forward(self, source, target):
        x = self.feature_extractor(torch.cat((source, target), dim=1))
        x = x.view(1, -1)
        x = self.regression_network(x)
        return x

# class Regression_Network(nn.Module):
#     def __init__(self):
#         super(Regression_Network, self).__init__()

#         self.fc = nn.Sequential(
#             nn.Linear(256, 6),
#         )

#     def forward(self, x):
#         x = self.fc(x)
#         return x.view(-1, 2, 3)
    
class Regression_Network(nn.Module):
    def __init__(self):
        super(Regression_Network, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.fc(x)
        return x.view(2)

class Forward_Layer(nn.Module):
    def __init__(self, channels, pool=False):
        super(Forward_Layer, self).__init__()
        self.pool = pool
        if self.pool:
            self.pool_layer = nn.Sequential(
                nn.Conv2d(channels, 2*channels, 3, stride=2, padding=3)
            )
            self.layer = nn.Sequential(
                nn.Conv2d(channels, 2*channels, 3, stride=2, padding=3),
                nn.GroupNorm(2*channels, 2*channels),
                nn.PReLU(),
                nn.Conv2d(2*channels, 2*channels, 3, stride=1, padding=1),
                nn.GroupNorm(2*channels, 2*channels),
                nn.PReLU(),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                nn.GroupNorm(channels, channels),
                nn.PReLU(),
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                nn.GroupNorm(channels, channels),
                nn.PReLU(),
            )

    def forward(self, x):
        if self.pool:
            return self.pool_layer(x) + self.layer(x)
        else:
            return x + self.layer(x)

class Feature_Extractor(nn.Module):
    def __init__(self, device):
        super(Feature_Extractor, self).__init__()
        self.device = device
        self.input_layer = nn.Sequential(
            nn.Conv2d(2, 64, 7, stride=2, padding=3),
        )
        self.layer_1 = Forward_Layer(64, pool=True)
        self.layer_2 = Forward_Layer(128, pool=False)
        self.layer_3 = Forward_Layer(128, pool=True)
        self.layer_4 = Forward_Layer(256, pool=False)
        self.layer_5 = Forward_Layer(256, pool=True)
        self.layer_6 = Forward_Layer(512, pool=True)

        self.last_layer = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=2, padding=1),
            nn.GroupNorm(512, 512),
            nn.PReLU(),
            nn.Conv2d(512, 256, 3, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.last_layer(x)
        return x

def load_network(device, path=None):
    model = Affine_Network(device)
    model = model.to(device)
    # summary(model, [(1, 1, 256, 256), (1, 1, 256, 256)])
    if path is not None:
        model.load_state_dict(torch.load(path))
        model.eval()
    return model

def transform_points_rigid(points, t):
    # input points and subtract t[0] from x and t[1] from y
    points = points.T
    points_x = torch.subtract(points[:, 0], 256*t[0]) # either add or subtract
    points_y = torch.subtract(points[:, 1], 256*t[1])
    # print(points_x.shape, points_y.shape)
    results = torch.stack((points_x, points_y), dim=1)
    # print(results.shape)
    return results


class SP_DHR_Net(nn.Module):
    def __init__(self, model_params):
        super(SP_DHR_Net, self).__init__()
        self.affineNet = load_network(device)
        self.model_params = model_params
        print("\nRunning new version (not run SP on source image)")

        # inputs = torch.rand((1, 1, image_size, image_size)), torch.rand((1, 1, image_size, image_size))
        # summary(self.affineNet, *inputs, show_input=True, show_hierarchical=True, print_summary=True)

    def forward(self, source_image, target_image, points):
        # if self.model_params.points == 1:
        affine_params = self.affineNet(source_image, target_image).to(device)
        # elif self.model_params.heatmaps == 1:
        #     print("This part is not yet implemented.")
            # affine_params = self.affineNet(source_image, target_image, heatmap1, heatmap2)
        
        identity = torch.eye(2).to(device)
        # concat affine parameters to identity matrix
        M = torch.cat((identity, affine_params.view(2, 1)), dim=1)
        # unsqueeze M to get a batch size of 1
        M = M.unsqueeze(0).cpu()

        transformed_source_image = tensor_affine_transform(source_image.detach().cpu(), M)
        # transformed_points = transform_points_DVF(points[0].cpu().detach().T, 
        #     affine_params.cpu().detach(), transformed_source_image.cpu().detach())

        transformed_points = transform_points_rigid(points[0].T.to(device), affine_params)
        transformed_points = transformed_points.requires_grad_(True)

        return transformed_source_image, affine_params, transformed_points.T
    