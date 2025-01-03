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
from utils.utils1 import transform_points_DVF

class SP_DHR_siamese(nn.Module):
    def __init__(self, model_params):
        super(SP_DHR_siamese, self).__init__()
        self.superpoint = SuperPointFrontend('utils/superpoint_v1.pth', nms_dist=4,
                          conf_thresh=0.015, nn_thresh=0.7, cuda=True)
        self.affineNet = an.load_network(device)
        self.nn_thresh = 0.7
        self.model_params = model_params
        print("\nRunning new version (not run SP on source image)")

        # inputs = torch.rand((1, 1, image_size, image_size)), torch.rand((1, 1, image_size, image_size))
        # summary(self.affineNet, *inputs, show_input=True, show_hierarchical=True, print_summary=True)

    def forward(self, source_image, target_image):
        points1, desc1, heatmap1 = self.superpoint(source_image[0, 0, :, :].cpu().numpy())
        points2, desc2, heatmap2 = self.superpoint(target_image[0, 0, :, :].cpu().numpy())

        if self.model_params.heatmaps == 0:
            affine_params = self.affineNet(source_image, target_image)
        elif self.model_params.heatmaps == 1:
            print("This part is not yet implemented.")
            # affine_params = self.affineNet(source_image, target_image, heatmap1, heatmap2)

        # transform the source image using the affine parameters
        # using F.affine_grid and F.grid_sample
        transformed_source_affine = tensor_affine_transform(source_image, affine_params)
        points1_2, desc1_2, heatmap1_2 = self.superpoint(transformed_source_affine[0, 0, :, :].detach().cpu().numpy())

        # match the points between the two images
        tracker = PointTracker(5, nn_thresh=0.7)
        try:
            matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
        except:
            # print('No matches found')
            # TODO: find a better way to do this
            try:
                while matches.shape[1] < 3 and self.nn_thresh > 0.1:
                    self.nn_thresh = self.nn_thresh - 0.1
                    matches = tracker.nn_match_two_way(desc1, desc2, nn_thresh=self.nn_thresh)
            except:
                return transformed_source_affine, affine_params, [], [], [], [], [], [], []

        # take the elements from points1 and points2 using the matches as indices
        matches1 = np.array(points1[:2, matches[0, :].astype(int)])
        matches2 = np.array(points2[:2, matches[1, :].astype(int)])

        # try:
        #     matches1_2 = points1_2[:2, matches[0, :].astype(int)]
        # except:
        # print(affine_params.cpu().detach().shape, transformed_source_affine.shape)
        matches1_2 = transform_points_DVF(torch.tensor(matches1), 
                        affine_params.cpu().detach(), transformed_source_affine)

        # transform the points using the affine parameters
        # matches1_transformed = transform_points(matches1.T[None, :, :], affine_params.cpu().detach())
        return transformed_source_affine, affine_params, matches1, matches2, matches1_2, \
            desc1_2, desc2, heatmap1_2, heatmap2
    