import argparse
import glob
import numpy as np
import os
import time
import cv2
import torch
import matplotlib.pyplot as plt
import random
import math
from skimage.metrics import structural_similarity as ssim
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, AffineTransform
# Suppress the specific warning
import warnings
import csv
import sys
# from IPython.utils.capture import capture_output
from datetime import datetime
# install from here https://pypi.org/project/SimpleITK-SimpleElastix/
# pip install SimpleITK-SimpleElastix
import SimpleITK as sitk 

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

# Jet colormap for visualization.
myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

from utils.SuperPoint import SuperPointFrontend, PointTracker, load_image
from utils.datagen import datagen
from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import ModelParams, print_summary, DL_affine_plot

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

image_size = 256

def process_image(image):
    # squeeze dimensions 0 and 1
    image = image.squeeze(0).squeeze(0)
    # convert to numpy array
    image = image.cpu().numpy()
    # normalize image to range 0 to 1
    image = (image/np.max(image)).astype('float32')
    return image

def get_affine_matrix_from_elastix(transform_params, center_of_rotation, inverse=True):
    """
    Convert SimpleElastix transformation parameters to a 3x3 affine matrix.
    
    Args:
        transform_params (list): 6 parameters [a, b, c, d, tx, ty] from Elastix
        center_of_rotation (list): [x, y] center point used in the registration
    
    Returns:
        numpy.ndarray: 3x3 affine transformation matrix
    """
    # Extract parameters
    a, b, c, d, tx, ty = transform_params
    cx, cy = center_of_rotation
    
    # Create rotation/scaling matrix
    R = np.array([[a, b],
                  [c, d]])
    
    # Calculate the true translation
    t = np.array([tx, ty]) + np.array([cx, cy]) - np.dot(R, np.array([cx, cy]))
    
    # Create full 3x3 affine matrix
    matrix = np.eye(3)
    matrix[:2, :2] = R
    matrix[:2, 2] = t

    if inverse:
        return np.linalg.inv(matrix)
    return matrix

def transform_points(points, affine_matrix):
    """
    Transform a set of 2D points using an affine transformation matrix.
    
    Args:
        points (numpy.ndarray): Nx2 array of points
        affine_matrix (numpy.ndarray): 3x3 affine transformation matrix
    
    Returns:
        numpy.ndarray: Transformed points as Nx2 array
    """
    # print(f"points: {points.shape}")
    points = points.view(-1, 2)
    # Convert to homogeneous coordinates
    homogeneous_points = np.hstack([points, np.ones((len(points), 1))])
    
    # Apply transformation
    transformed_points = np.dot(homogeneous_points, affine_matrix.T)
    
    # Convert back to 2D coordinates
    return transformed_points[:, :2]

def apply_elastix_transform(points, transform_params, center_of_rotation):
    """
    Apply SimpleElastix transformation to a set of points.
    
    Args:
        points (numpy.ndarray): Nx2 array of points to transform
        transform_params (list): 6 parameters from Elastix
        center_of_rotation (list): [x, y] center point used in the registration
    
    Returns:
        numpy.ndarray: Transformed points
    """
    # Get the full affine matrix
    affine_matrix = get_affine_matrix_from_elastix(transform_params, center_of_rotation)
    
    # Transform the points
    return transform_points(points, affine_matrix).T

def run(model_params, method='affine', plot=1, num_iter=0):
    # Initialize SuperPointFrontend
    superpoint = SuperPointFrontend('utils/superpoint_v1.pth', nms_dist=4,
                          conf_thresh=0.015,
                          nn_thresh=0.7, cuda=True)
    
    test_dataset = datagen(model_params.dataset, False, model_params.sup)

    # Create output directory
    output_dir = f"output/{args.model}_{model_params.get_model_code()}_{timestamp}_elastix_{num_iter}_test"
    os.makedirs(output_dir, exist_ok=True)

    metrics = []
    # create a csv file to store the metrics
    csv_file = f"{output_dir}/metrics-{timestamp}.csv"

    testbar = tqdm(test_dataset, desc=f'Testing:')

    parameterMap = sitk.GetDefaultParameterMap(f'{method}')
    # Use a non-rigid transform instead of a translation transform
    # parameterMap['Transform'] = ['BSplineTransform']

    # Because of the increased complexity of the b-spline transform,
    # it is a good idea to run the registration a little longer to
    # ensure convergence
    if num_iter > 0:
        parameterMap['MaximumNumberOfIterations'] = [f'{int(num_iter)}']

    # print(f"\nparameterMap: {parameterMap}")

    for i, data in enumerate(testbar, 0):
        # Get images and affine parameters
        source_image, target_image, affine_params_true, points1, points2, points1_2_true = data
            
        # process images
        source_image = process_image(source_image)
        target_image = process_image(target_image)

        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.LogToConsoleOff()
        elastixImageFilter.SetFixedImage(sitk.GetImageFromArray(target_image))
        elastixImageFilter.SetMovingImage(sitk.GetImageFromArray(source_image))
        # elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
        elastixImageFilter.SetParameterMap(parameterMap)
        elastixImageFilter.Execute()
        transformed_source_affine = elastixImageFilter.GetResultImage()
        transformed_source_affine = sitk.GetArrayFromImage(transformed_source_affine)

        # get parameters to transform points
        affine_transform1 = elastixImageFilter.GetTransformParameterMap()
        affine_transform1 = elastixImageFilter.GetTransformParameterMap()[0]["TransformParameters"]
        affine_transform1 = np.array([float(i) for i in affine_transform1])
        affine_transform1 = torch.tensor(affine_transform1, dtype=torch.float32).view(1, 2, 3)
        # print(f"\naffine_transform1: {affine_transform1}")

        # reshape points1
        points1_ = points1.view(2, -1, 1)
        # print(f"points1: {points1.shape}")
        # points1_transformed = transform_points_DVF_unbatched(points1_, affine_transform1, source_image)
        # print(affine_transform1.shape)
        points1_transformed = apply_elastix_transform(points1_, affine_transform1[0].view(-1).cpu().numpy(), [image_size/2, image_size/2])
        # print(f"points1: {points1.shape}")
        # print(f"points2: {points2.shape}")
        # print(f"points1_transformed: {points1_transformed.shape}")

        if i < 100 and plot == 1:
            plot_ = 1
        elif i < 100 and plot == 2:
            plot_ = 2
        else:
            plot_ = 0
            
        points1 = points1[0].T.cpu().numpy()
        points2 = points2[0].T.cpu().numpy()
        # points1_transformed = torch.tensor(points1_transformed)

        # print(f"points1_transformed: {points1_transformed.shape}")

        results = DL_affine_plot(f"test", output_dir,
                f"{i:03d}", "SE", 
                source_image, target_image,
                transformed_source_affine,
                points1, points2, points1_transformed, None, None, 
                # points1, points2, None, None, None,
                affine_params_true=affine_params_true,
                affine_params_predict=affine_transform1,
                heatmap1=None, heatmap2=None, plot=plot_)


        # calculate metrics
        # matches1_transformed = results[0]
        mse_before = results[1]
        mse12 = results[2]
        tre_before = tre(points1, points2)
        tre12 = tre(points1, points1_transformed)
        # print(f"mse_before: {mse_before}, mse12: {mse12}, tre_before: {tre_before}, tre12: {tre12}")
        mse12_image_before = results[5]
        mse12_image = results[6]
        ssim12_image_before = results[7]
        ssim12_image = results[8]

        # append metrics to metrics list
        metrics.append([i, mse_before, mse12, tre_before, tre12, mse12_image_before, 
            mse12_image, ssim12_image_before, ssim12_image, np.max(points1.shape)])
        
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "mse_before", "mse12", "tre_before", "tre12", "mse12_image_before", "mse12_image", "ssim12_image_before", "ssim12_image", "num_points"])
        for i in range(len(metrics)):
            writer.writerow(metrics[i])
        # write the average and std of the metrics
        metrics = np.array(metrics)
        nan_mask = np.isnan(metrics).any(axis=1)
        metrics = metrics[~nan_mask]
        avg = ["average", np.mean(metrics[:, 1]), np.mean(metrics[:, 2]), np.mean(metrics[:, 3]), np.mean(metrics[:, 4]), 
            np.mean(metrics[:, 5]), np.mean(metrics[:, 6]), np.mean(metrics[:, 7]), np.mean(metrics[:, 8])]
        std = ["std", np.std(metrics[:, 1]), np.std(metrics[:, 2]), np.std(metrics[:, 3]), np.std(metrics[:, 4]), 
            np.std(metrics[:, 5]), np.std(metrics[:, 6]), np.std(metrics[:, 7]), np.std(metrics[:, 8])]
        writer.writerow(avg)
        writer.writerow(std)

    print(f"The test results are saved in {csv_file}")

    # delete all txt files in output_dir
    # for file in os.listdir(output_dir):
    #     if file.endswith(".txt"):
    #         os.remove(os.path.join(output_dir, file))

    # print_summary(args.model, None, model_params, None, timestamp, True)


# if main
# from utils.utils1 import transform_points_DVF
if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')    
    parser.add_argument('--dataset', type=int, default=10, help='dataset number')
    # parser.add_argument('--sup', type=int, default=0, help='supervised learning (1) or unsupervised learning (0)')
    # parser.add_argument('--image', type=int, default=1, help='image used for training')
    # parser.add_argument('--heatmaps', type=int, default=0, help='use heatmaps (1) or not (0)')
    # parser.add_argument('--loss_image', type=int, default=0, help='loss function for image registration')
    parser.add_argument('--num_iter', type=int, default=1, help='number of iterations')
    # parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    # parser.add_argument('--decay_rate', type=float, default=0.96, help='decay rate')
    parser.add_argument('--model', type=str, default='SE', help='which model to use')
    # parser.add_argument('--model_path', type=str, default=None, help='path to model to load')
    parser.add_argument('--plot', type=int, default=1, help='plot the results')
    parser.add_argument('--method', type=str, default='affine')
    args = parser.parse_args()

    model_params = ModelParams(dataset=args.dataset, 
                               )
    model_params.print_explanation()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run(model_params, method=args.method, plot=args.plot, num_iter=args.num_iter)
