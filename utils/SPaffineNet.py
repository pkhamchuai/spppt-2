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

# define model
class SP_AffineNet(nn.Module):
    def __init__(self, model_params):
        super(SP_AffineNet, self).__init__()
        # self.superpoint = SuperPointFrontend('utils/superpoint_v1.pth', nms_dist=4,
        #                   conf_thresh=0.015, nn_thresh=0.7, cuda=True)
        self.affineNet = AffineNet()
        # self.nn_thresh = 0.7
        self.model_params = model_params
        print("\nRunning new version (not run SP on source image)")

        inputs = torch.rand((1, 1, image_size, image_size)), torch.rand((1, 1, image_size, image_size))
        summary(self.affineNet, *inputs, show_input=True, show_hierarchical=True, print_summary=True)

    def forward(self, source_image, target_image, points):
        # source_image = source_image.to(device)
        # target_image = target_image.to(device)

        if self.model_params.heatmaps == 0:
            affine_params = self.affineNet(source_image, target_image)
        elif self.model_params.heatmaps == 1:
            print("This part is not yet implemented.")
            # affine_params = self.affineNet(source_image, target_image, heatmap1, heatmap2)

        # transform the source image using the affine parameters
        # using F.affine_grid and F.grid_sample
        transformed_source_image = tensor_affine_transform(source_image, affine_params)
        transformed_points = points.clone()
        transformed_points = transform_points_DVF(transformed_points.cpu().detach().T, 
            affine_params.cpu().detach(), transformed_source_image.cpu().detach())

        return transformed_source_image, affine_params, transformed_points.T

class AffineNet(nn.Module):
    def __init__(self):
        super(AffineNet, self).__init__()
        self.conv1f = 64
        self.conv2f = 128
        self.conv3f = 256
        self.conv4f = 512
        self.conv5f = 512
        self.conv1 = nn.Conv2d(1, self.conv1f, 3, padding=1, padding_mode='zeros')
        self.conv2 = nn.Conv2d(self.conv1f, self.conv2f, 3, padding=1, padding_mode='zeros')
        self.conv3 = nn.Conv2d(self.conv2f, self.conv3f, 3, padding=1, padding_mode='zeros')
        self.conv4 = nn.Conv2d(self.conv3f, self.conv4f, 3, padding=1, padding_mode='zeros')
        self.conv5 = nn.Conv2d(self.conv4f, self.conv5f, 3, padding=1, padding_mode='zeros')

        self.conv1s = nn.Conv2d(self.conv1f, self.conv1f, 2, stride=2, padding_mode='zeros')
        self.conv2s = nn.Conv2d(self.conv2f, self.conv2f, 2, stride=2, padding_mode='zeros')
        self.conv3s = nn.Conv2d(self.conv3f, self.conv3f, 2, stride=2, padding_mode='zeros')
        self.conv4s = nn.Conv2d(self.conv4f, self.conv4f, 2, stride=2, padding_mode='zeros')
        self.fc1 = nn.Linear(self.conv5f*2, 128)
        self.fc2 = nn.Linear(128, 6)

        self.aPooling = nn.AdaptiveAvgPool2d((1, 1))
        # self.ReLU = nn.PReLU()
        self.ReLU = nn.LeakyReLU()
        self.Act1 = nn.GroupNorm(int(self.conv1f/2), self.conv1f, eps=1e-05, affine=True)
        self.Act2 = nn.GroupNorm(int(self.conv2f/2), self.conv2f, eps=1e-05, affine=True)
        self.Act3 = nn.GroupNorm(int(self.conv3f/2), self.conv3f, eps=1e-05, affine=True)
        self.Act4 = nn.GroupNorm(int(self.conv4f/2), self.conv4f, eps=1e-05, affine=True)
        self.Act5 = nn.GroupNorm(int(self.conv5f/2), self.conv5f, eps=1e-05, affine=True)
        '''self.Act1 = nn.BatchNorm2d(self.conv1f)
        self.Act2 = nn.BatchNorm2d(self.conv2f)
        self.Act3 = nn.BatchNorm2d(self.conv3f)
        self.Act4 = nn.BatchNorm2d(self.conv4f)
        self.Act5 = nn.BatchNorm2d(self.conv5f)'''
        '''
        self.pooling = nn.AvgPool2d(2, ceil_mode=False)
        self.elu = nn.ELU()
        self.GN = nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)'''

        # self.flow = torch.nn.Parameter(torch.zeros(1, 2, image_size, image_size).to(device), requires_grad=True)
        # self.img = torch.nn.Parameter(torch.zeros(1, 1, image_size, image_size).to(device), requires_grad=True)

    def forward(self, source_image, target_image):

        # print(source_image.size(), target_image.size())
        # x = self.conv1(source_image)
        x = self.Act1(self.ReLU(self.conv1s(self.Act1(self.ReLU(self.conv1(source_image))))))
        y = self.Act1(self.ReLU(self.conv1s(self.Act1(self.ReLU(self.conv1(target_image))))))
        # print(x.shape, y.shape)
        x = self.Act2(self.ReLU(self.conv2s(self.Act2(self.ReLU(self.conv2(x))))))
        y = self.Act2(self.ReLU(self.conv2s(self.Act2(self.ReLU(self.conv2(y))))))
        # print(x.shape, y.shape)
        x = self.Act3(self.ReLU(self.conv3s(self.Act3(self.ReLU(self.conv3(x))))))
        y = self.Act3(self.ReLU(self.conv3s(self.Act3(self.ReLU(self.conv3(y))))))
        # print(x.shape, y.shape)
        x = self.Act4(self.ReLU(self.conv4s(self.Act4(self.ReLU(self.conv4(x))))))
        y = self.Act4(self.ReLU(self.conv4s(self.Act4(self.ReLU(self.conv4(y))))))
        # print(x.shape, y.shape)
        x = self.aPooling(self.Act5(self.ReLU(self.conv5(x))))
        y = self.aPooling(self.Act5(self.ReLU(self.conv5(y))))
        # print(x.shape, y.shape)
        t = torch.cat((x, y), dim=1)
        # print(t.shape)
        t = self.ReLU(self.fc1(t.flatten()))
        # print(t.shape)
        t = self.fc2(t)
        t = t.view(-1, 2, 3)
        # print(t.shape)
        return t

    