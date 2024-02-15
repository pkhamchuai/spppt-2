import sys
import argparse
import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from skimage.measure import ransac
# from skimage.transform import FundamentalMatrixTransform, AffineTransform

import csv
from datetime import datetime
from tqdm import tqdm

import torch
# from torchvision import transforms
from torch import nn, optim
# import torch.nn.functional as F
# from torch.utils import data
from pytorch_model_summary import summary

# print('Seed:', torch.seed())
torch.manual_seed(9793047918980052389)

from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import ModelParams, DL_affine_plot, print_summary
from utils.datagen import datagen

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

image_size = 256

from networks import affine_network_simple_test_size as an
from utils.SuperPoint import SuperPointFrontend
from utils.utils1 import transform_points_DVF
from utils.train import train
from utils.test import test

if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')    
    parser.add_argument('--dataset', type=int, default=1, help='dataset number')
    parser.add_argument('--sup', type=int, default=1, help='supervised learning (1) or unsupervised learning (0)')
    parser.add_argument('--image', type=int, default=1, help='image used for training')
    parser.add_argument('--heatmaps', type=int, default=0, help='use heatmaps (1) or not (0)')
    parser.add_argument('--loss_image', type=int, default=0, help='loss function for image registration')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='decay rate')
    parser.add_argument('--model_path', type=str, default=None, help='path to model to load')
    args = parser.parse_args()


    model_params = ModelParams(dataset=args.dataset, sup=args.sup, image=args.image, heatmaps=args.heatmaps, 
                               loss_image=args.loss_image, num_epochs=args.num_epochs, 
                               learning_rate=args.learning_rate, decay_rate=args.decay_rate)
    model_params.print_explanation()
    train_dataset = datagen(model_params.dataset, True, model_params.sup)
    test_dataset = datagen(model_params.dataset, False, model_params.sup)

    # Get sample batch
    print('Train set: ', [x.shape for x in next(iter(train_dataset))])
    print('Test set: ', [x.shape for x in next(iter(test_dataset))])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model, loss_list = train(model_params, timestamp)

    # save the output of print_explanation() and loss_list to a txt file
    print_summary(model_params, loss_list, timestamp)

    print("\nTesting the trained model +++++++++++++++++++++++")

    # model = SPmodel = SP_AffineNet().to(device)
    # print(model)

    # parameters = model.parameters()
    # optimizer = optim.Adam(parameters, model_params.learning_rate)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: model_params.decay_rate ** epoch)

    # model.load_state_dict(torch.load(model_name_to_save))

    test(model, model_params, timestamp)
    print("Test model finished +++++++++++++++++++++++++++++")
