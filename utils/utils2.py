import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt

from datetime import datetime

import torch
import torch.functional as F
torch.manual_seed(9793047918980052389)
print('Seed:', torch.seed())

from utils.utils0 import *
from utils.utils1 import *

# Convert 1x2x3 parameters to 3x3 affine transformation matrices
def params_to_matrix(params):
    params = params.view(-1, 6)[0]
    # print(params)
    return torch.tensor([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [0,         0,         1]
    ], dtype=torch.float32)

# Convert the combined 3x3 matrix back to 1x6 affine parameters
def matrix_to_params(matrix):
    matrix = torch.tensor([
        matrix[0, 0], matrix[0, 1], matrix[0, 2],
        matrix[1, 0], matrix[1, 1], matrix[1, 2]
        ], dtype=torch.float32)
    return matrix.view(1, 2, 3)

# combine 2 affine matrices
def combine_matrices(matrix1, matrix2):
    # print(matrix1, matrix2)
    matrix1 = params_to_matrix(matrix1)
    matrix2 = params_to_matrix(matrix2)
    return matrix_to_params(torch.matmul(matrix1, matrix2))

# Transform a tensor image using an affine transformation matrix
def tensor_affine_transform0(image, matrix):
    # Get the image size
    _, _, H, W = image.size()
    # Generate a grid of (x,y) coordinates
    grid = F.affine_grid(matrix, [1, 1, H, W], align_corners=False)
    # Sample the input image at the grid points
    
    # if image and matrix are on the same device
    if image.device == matrix.device:
        transformed_image = F.grid_sample(image, grid, align_corners=False)
    else:
        transformed_image = F.grid_sample(image.to(matrix.device), grid, align_corners=False)
    return transformed_image

def transform_points_DVF0(points_, M, image, reverse=False):
    # DVF.shape = (B, 2, H, W)
    # points.shape = (2, N, B)
    
    if points_.shape[0] == 2 and points_.shape[1] == 2:
        points_ = points_.T
    elif points_.shape[0] != 2:
        points_ = points_.reshape(2, -1, points_.shape[0])
    # print(points_.shape)

    # print(M.shape)
    # print(image.shape)
    _, N, B = points_.shape
    # print("points_ shape:", points_.shape, "M shape:", M.shape, "image shape:", image.shape)
    # points = [transform_points_DVF_unbatched(points_[:, :, b], M[b, :, :], image[b, :, :, :]) for b in range(B)]
    for b in range(B):
        # print("points_ shape:", points_[:, :, b].shape, "M shape:", M[b].shape, "image shape:", image[b].shape)
        points_[:, :, b] = \
            transform_points_DVF_unbatched0(points_[:, :, b].view(2, N, 1), \
            M[b].view(1, 2, 3), image[b].view(1, 1, image[b].size(-1), image[b].size(-1)), 
            reverse=reverse)
    # print(points_.shape)
    return points_

def transform_points_DVF_unbatched0(points_, M, image, reverse=False):
    # transform points using displacement field
    # DVF.shape = (2, H, W)
    # points.shape = (2, N, 1)
    # print("points_ shape:", points_.shape, "M shape:", M.shape, "image shape:", image.shape)

    # if points_ has 3 dimensions, remove the last dimension
    if len(points_.shape) == 3:
        points_ = points_.squeeze(-1)
    # print("points_ shape:", points_.shape)
    displacement_field = torch.zeros(image.shape[-1], image.shape[-1]).to(M.device)
    DVF = transform_to_displacement_field(
        displacement_field.view(1, 1, displacement_field.size(0), displacement_field.size(1)), 
        M.clone().view(1, 2, 3), device=M.device)
    
    if reverse:
        DVF = -DVF

    if isinstance(DVF, torch.Tensor):
        DVF = DVF.cpu().detach().numpy()

    # loop through each point and apply the transformation
    points = points_.clone()
    points = points.detach().numpy()#.copy()

    for i in range(points.shape[1]):
        try:
            # print(points[:, i], DVF[:, int(points[1, i]), int(points[0, i])])
            points[:, i] = points[:, i] - DVF[:, int(points[1, i]), int(points[0, i])]
            # print(points[:, i])
        except IndexError:
            # if both points are outside the image
            # find the nearest point inside the image and apply the transformation
            if int(points[1, i]) > 255 and int(points[0, i]) > 255:
                points[:, i] = points[:, i] - DVF[:, 255, 255]
            elif int(points[1, i]) < 0 and int(points[0, i]) < 0:
                points[:, i] = points[:, i] - DVF[:, 0, 0]
            elif int(points[1, i]) > 255 and int(points[0, i]) < 0:
                points[:, i] = points[:, i] - DVF[:, 255, 0]
            elif int(points[1, i]) < 0 and int(points[0, i]) > 255:
                points[:, i] = points[:, i] - DVF[:, 0, 255]
            # if only one point is outside the image
            elif int(points[1, i]) > 255:
                points[:, i] = points[:, i] - DVF[:, 255, int(points[0, i])]
            elif int(points[1, i]) < 0:
                points[:, i] = points[:, i] - DVF[:, 0, int(points[0, i])]
            elif int(points[0, i]) > 255:
                points[:, i] = points[:, i] - DVF[:, int(points[1, i]), 255]
            elif int(points[0, i]) < 0:
                points[:, i] = points[:, i] - DVF[:, int(points[1, i]), 0]
            # print("points:", points[0, i], points[1, i])
                
                        # change int(points[1, i]), int(points[0, i]) to 256 if it is 257
            '''if int(points[1, i]) > 255:
                points[1, i] = 255
            elif int(points[1, i]) < 0:
                points[1, i] = 0
            if int(points[0, i]) > 255:
                points[0, i] = 255
            elif int(points[0, i]) < 0:
                points[0, i] = 0
            points[:, i] = points[:, i] - DVF[:, int(points[1, i]), int(points[0, i])]'''

        # print("points shape:", points.shape)
    return torch.tensor(points)