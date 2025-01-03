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
# from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, AffineTransform
# Suppress the specific warning
import warnings
import csv
import sys
# from IPython.utils.capture import capture_output
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

from utils.SuperPoint import SuperPointFrontend, PointTracker, load_image, ransac
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
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    return image

def run(model_params, method1='BFMatcher', method2='RANSAC', plot=1):
    
    test_dataset = datagen(model_params.dataset, False, model_params.sup)

    # Create output directory
    output_dir = f"output/{args.model}_{model_params.get_model_code()}_{timestamp}_{method1}_{method2}_test"
    os.makedirs(output_dir, exist_ok=True)

    metrics = []
    # create a csv file to store the metrics
    csv_file = f"{output_dir}/metrics.csv"

    num_failed = 0

    testbar = tqdm(test_dataset, desc=f'Testing:')
    for i, data in enumerate(testbar, 0):
        # Get images and affine parameters
        source_image, target_image, affine_params_true, points1, points2, points1_2_true = data
        points1 = points1.squeeze(0).cpu().numpy()
        points2 = points2.squeeze(0).cpu().numpy()
        
        # keep the original images
        source_image_orig = source_image.squeeze(0).squeeze(0).cpu().numpy()
        target_image_orig = target_image.squeeze(0).squeeze(0).cpu().numpy()

        # process images
        source_image = process_image(source_image)
        target_image = process_image(target_image)
        # print(f"source_image: {source_image.shape}")
        # print(f"target_image: {target_image.shape}")

        # Extract keypoints and descriptors using SIFT
        sift = cv2.SIFT_create()
        kp1, desc1 = sift.detectAndCompute(source_image, None)
        kp2, desc2 = sift.detectAndCompute(target_image, None)

        # print(desc1.shape)
        # print(desc2.shape)

        # pad the smaller desc to the same size with zeros
        # if desc1.shape[0] < desc2.shape[0]:
        #     desc1 = np.pad(desc1, ((0, desc2.shape[0] - desc1.shape[0]), (0, 0)), mode='constant')
        #     kp1 = kp1 + tuple([cv2.KeyPoint(x=kp1[0].pt[0], y=kp1[0].pt[1], size=0)])
        # elif desc1.shape[0] > desc2.shape[0]:
        #     desc2 = np.pad(desc2, ((0, desc1.shape[0] - desc2.shape[0]), (0, 0)), mode='constant')
        #     kp2 = kp2 + tuple([cv2.KeyPoint(x=kp2[0].pt[0], y=kp2[0].pt[1], size=0)])

        if method1 == 'BFMatcher':
            # Match keypoints using nearest neighbor search
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc1, desc2, k=2)

            good = []
            try:
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        good.append([m])
            except ValueError:
                good = []

            # img3 = cv2.drawMatchesKnn(source_image, kp1, target_image, kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            # plt.imshow(img3), plt.show()

            matches = np.array([m for m in matches])
            matches1 = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1, 2)
            matches2 = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1, 2)
            # print(f"matches1: {matches1.shape}")
            # print(f"matches2: {matches2.shape}")

        elif method1 == 'FLANN':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            matches = flann.knnMatch(desc1,desc2,k=2)

            # Apply ratio test to filter out ambiguous matches
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # Apply RANSAC to filter out outliers
            matches1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            matches2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

            # # Need to draw only good matches, so create a mask
            # matchesMask = [[0,0] for i in range(len(matches))]
            
            # # ratio test as per Lowe's paper
            # for i,(m,n) in enumerate(matches):
            #     if m.distance < 0.*n.distance:
            #         matchesMask[i]=[1,0]

            # draw_params = dict(matchColor = (0,255,0),
            #     singlePointColor = (255,0,0),
            #     matchesMask = matchesMask,
            #     flags = cv2.DrawMatchesFlags_DEFAULT)
            
            # img3 = cv2.drawMatchesKnn(source_image, kp1, target_image, kp2, matches, None, **draw_params)
            # plt.imshow(img3), plt.show()
        
        try:
            if method2 == 'RANSAC':
                affine_transform1, _ = cv2.estimateAffine2D(matches1, matches2, method=cv2.RANSAC)
            elif method2 == 'LMEDS':
                affine_transform1, _ = cv2.estimateAffine2D(matches1, matches2, method=cv2.LMEDS)

            matches1_transformed = cv2.transform(matches1[None, :, :], affine_transform1[:2, :])[0]
            points1_transformed = cv2.transform(points1[None, :, :], affine_transform1[:2, :])[0]
            transformed_source_affine = cv2.warpAffine(source_image_orig, affine_transform1, (image_size, image_size))
            text = "success"

        # except when affine_transform1 is None
        except:
            affine_transform1 = np.array([[[1, 0, 0], [0, 1, 0]]])
            matches1_transformed = matches1
            points1_transformed = points1
            transformed_source_affine = source_image_orig
            text = "failed"

        if i < 100 and plot == 1:
            plot_ = 1
        elif i < 100 and plot == 2:
            plot_ = 2
            # do the bitwise not operation to get the inverse of the image
            # source_image = cv2.bitwise_not(source_image)
            # target_image = cv2.bitwise_not(target_image)
            # transformed_source_affine = cv2.bitwise_not(transformed_source_affine)
        else:
            plot_ = 0

        # print(f"matcches1: {matches1.shape}, matches2: {matches2.shape}, matches1_transformed: {matches1_transformed.shape}")
        # print(f"points1: {points1.shape}, points2: {points2.shape}, points1_transformed: {points1_transformed.shape}")
        affine_transform1 = torch.tensor(affine_transform1, dtype=torch.float32).unsqueeze(0).to(device)

        # print(affine_transform1.shape)
        _ = DL_affine_plot(f"test", output_dir,
            f"{i:03d}_SIFT", f"{text}", source_image_orig, target_image_orig,
            transformed_source_affine,
            matches1.T, matches2.T, matches1_transformed.T,
            None, None,
            affine_params_true=affine_params_true,
            affine_params_predict=affine_transform1, 
            heatmap1=None, heatmap2=None, plot=plot_)
        
        results = DL_affine_plot(f"test", output_dir,
            f"{i:03d}_metrics", f"{text}", source_image_orig, target_image_orig,
            transformed_source_affine,
            points1.T, points2.T, points1_transformed.T, 
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

        # if the metrics are nan, set them to before values
        if np.isnan(mse12):
            mse12 = mse_before
        if np.isnan(tre12):
            tre12 = tre_before

        # except:
        #     num_failed += 1
        #     # mse_before = mse12 = tre_before = tre12 = mse12_image_before = mse12_image = ssim12_image_before = ssim12_image = 0
        #     # matches1 = matches2 = matches1_2 = []
        #     text = "failed"
        #     plot_ = plot

        # append metrics to metrics list
        try:
            metrics.append([i, mse_before, mse12, tre_before, tre12, mse12_image_before, 
                mse12_image, ssim12_image_before, ssim12_image, np.max(matches1.shape)])
        except AttributeError:
            metrics.append([i, mse_before, mse12, tre_before, tre12, mse12_image_before, 
                mse12_image, ssim12_image_before, ssim12_image, 0])        
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["index", "mse_before", "mse12", "tre_before", "tre12", "mse12_image_before", "mse12_image", "ssim12_image_before", "ssim12_image", "num_points"])
        for i in range(len(metrics)):
            writer.writerow(metrics[i])
        # write the average and std of the metrics
        metrics = np.array(metrics)
        # nan_mask = np.isnan(metrics).any(axis=1)
        # metrics = metrics[~nan_mask]
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

    extra_text = f"Failed: {num_failed} out of {len(test_dataset)}"
    print(extra_text)
    print_summary("SIFT", None, model_params, None, timestamp, 
                  output_dir=output_dir, test=True, extra=extra_text)

# if main
# from utils.utils1 import transform_points_DVF
if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')    
    parser.add_argument('--dataset', type=int, default=1, help='dataset number')
    parser.add_argument('--sup', type=int, default=0, help='supervised learning (1) or unsupervised learning (0)')
    parser.add_argument('--image', type=int, default=1, help='image used for training')
    parser.add_argument('--heatmaps', type=int, default=0, help='use heatmaps (1) or not (0)')
    parser.add_argument('--loss_image', type=int, default=0, help='loss function for image registration')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='decay rate')
    parser.add_argument('--model', type=str, default='SIFT', help='which model to use')
    parser.add_argument('--model_path', type=str, default=None, help='path to model to load')
    parser.add_argument('--plot', type=int, default=1, help='plot the results')
    parser.add_argument('--method1', type=str, default='FLANN', help='method to use for matching keypoints')
    parser.add_argument('--method2', type=str, default='RANSAC', help='method to use for estimating affine transformation')
    args = parser.parse_args()

    model_params = ModelParams(dataset=args.dataset, sup=args.sup, image=args.image, #heatmaps=args.heatmaps, 
                               loss_image=args.loss_image, num_epochs=args.num_epochs, 
                               learning_rate=args.learning_rate, decay_rate=args.decay_rate)
    model_params.print_explanation()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    run(model_params, method1=args.method1, method2=args.method2, plot=args.plot)