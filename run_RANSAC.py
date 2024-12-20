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
from IPython.utils.capture import capture_output
from datetime import datetime

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

def run(model_params, timestamp, plot):
    # Initialize SuperPointFrontend
    # superpoint = SuperPointFrontend('utils/superpoint_v1.pth', nms_dist=4,
    #                       conf_thresh=0.015,
    #                       nn_thresh=0.7, cuda=True)
    
    test_dataset = datagen(model_params.dataset, False, model_params.sup)

    # Create output directory
    output_dir = f"output/{args.model}_{model_params.get_model_code()}_{timestamp}_test"
    os.makedirs(output_dir, exist_ok=True)

    metrics = []
    # create a csv file to store the metrics
    csv_file = f"{output_dir}/metrics.csv"

    testbar = tqdm(test_dataset, desc=f'Testing:')
    for i, data in enumerate(testbar, 0):
        # Get images and affine parameters
        source_image, target_image, affine_params_true, points1, points2, points1_2_true = data

        # process images
        source_image = process_image(source_image)
        target_image = process_image(target_image)
        points1 = points1[0].cpu().numpy()
        points2 = points2[0].cpu().numpy()


        try:
            if args.model == 'RANSAC':
                method = cv2.RANSAC
            elif args.model == 'LMEDS':
                method = cv2.LMEDS
            affine_transform1, inliers = cv2.estimateAffinePartial2D(points1, points2, method=method)
            points1_reshaped = points1.reshape(-1, 1, 2)
            matches1_transformed = cv2.transform(points1_reshaped, affine_transform1)
            matches1_transformed = matches1_transformed.reshape(-1, 2)
            # transform image 1 and 2 using the affine transform matrix
            transformed_source_affine = cv2.warpAffine(source_image, affine_transform1, (256, 256))
        except cv2.error:
            print(f"Error: {i}")
            # set affine_transform1 to identity affine matrix
            affine_transform1 = np.array([[[1., 0., 0.], [0., 1., 0.]]])
            affine_transform1 = torch.tensor(affine_transform1).view(1, 1, 2, 3).to(device)
            matches1_transformed = points1
            transformed_source_affine = source_image

        # mse12 = np.mean((matches1_transformed - matches2_RANSAC)**2)
        # tre12 = np.mean(np.sqrt(np.sum((matches1_transformed - matches2_RANSAC)**2, axis=0)))

        if i < 100 and plot == 1:
            plot_ = True
        elif i < 100 and plot == 0:
            plot_ = False

        
        affine_transform1 = torch.tensor(affine_transform1).view(1, 1, 2, 3).to(device)
        # print(affine_transform1.shape, affine_params_true.shape)
        results = DL_affine_plot(f"test", output_dir,
                f"{i:03d}", str(args.model),
                source_image, target_image,
                transformed_source_affine,
                points1.T, points2.T, matches1_transformed.T, 
                None, None,
                affine_params_true=affine_params_true,
                affine_params_predict=affine_transform1,
                heatmap1=None, heatmap2=None, plot=plot_)

        # calculate metrics
        # matches1_transformed = results[0]
        mse_before = results[1]
        mse12 = results[2]
        tre_before = results[3]
        tre12 = results[4]
        mse12_image_before = results[5]
        mse12_image = results[6]
        ssim12_image_before = results[7]
        ssim12_image = results[8]

        # append metrics to metrics list
        metrics.append([i, mse_before, mse12, tre_before, tre12, mse12_image_before, 
            mse12_image, ssim12_image_before, ssim12_image, matches1_transformed.shape[0]])
        
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

    # print_summary(args.model, None, model_params, 
    #               None, timestamp, True)


# if main
# from utils.utils1 import transform_points_DVF
if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')    
    parser.add_argument('--dataset', type=int, default=1, help='dataset number')
    parser.add_argument('--sup', type=int, default=1, help='supervised learning (1) or unsupervised learning (0)')
    parser.add_argument('--image', type=int, default=1, help='image used for training')
    # parser.add_argument('--heatmaps', type=int, default=0, help='use heatmaps (1) or not (0)')
    parser.add_argument('--loss_image', type=int, default=0, help='loss function for image registration')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='decay rate')
    parser.add_argument('--model', type=str, default='RANSAC', help='which model to use')
    parser.add_argument('--model_path', type=str, default=None, help='path to model to load')
    parser.add_argument('--plot', type=int, default=0, help='plot the results')
    args = parser.parse_args()

    model_params = ModelParams(dataset=args.dataset, sup=args.sup, image=args.image, # heatmaps=args.heatmaps, 
                               loss_image=args.loss_image, num_epochs=args.num_epochs, 
                               learning_rate=args.learning_rate, decay_rate=args.decay_rate)
    model_params.print_explanation()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run(model_params, timestamp, args.plot)