# import sys
import argparse
import cv2
import torch
# from skimage.metrics import structural_similarity as ssim
# from skimage.measure import ransac
# from skimage.transform import FundamentalMatrixTransform, AffineTransform

from datetime import datetime

import torch
# from torchvision import transforms
# import torch.nn.functional as F
# from torch.utils import data
# from torchsummary import summary
# from pytorch_model_summary import summary

from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import ModelParams
from utils.train import train
# from utils.test import test
from test_points import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

image_size = 256

# from utils.SuperPoint import SuperPointFrontend
# from utils.utils1 import transform_points_DVF

if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')    
    parser.add_argument('--dataset', type=int, default=1, help='dataset number')
    parser.add_argument('--sup', type=int, default=1, help='supervised learning (1) or unsupervised learning (0)')
    parser.add_argument('--image', type=int, default=1, help='loss image used for training')
    parser.add_argument('--points', type=int, default=0, help='use loss points (1) or not (0)')
    parser.add_argument('--loss_image', type=int, default=0, help='loss function for image registration')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate')
    parser.add_argument('--model', type=str, default="DHR", help='which model to use')
    parser.add_argument('--model_path', type=str, default=None, help='path to model to load')
    parser.add_argument('--timestamp', type=str, default=None, help='timestamp')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    args = parser.parse_args()

    if args.model_path is None:
      model_path = None
    else:
      model_path = 'trained_models/' + args.model_path
    model_params = ModelParams(dataset=args.dataset, sup=args.sup, image=args.image, points=args.points,
                              loss_image=args.loss_image, num_epochs=args.num_epochs, 
                              learning_rate=args.learning_rate, decay_rate=args.decay_rate,
                              model=args.model, batch_size=args.batch_size)
    
    if args.timestamp is not None:
      timestamp = args.timestamp
    else:
      timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
      # print(f'Timestamp: {timestamp}')
      model_params.timestamp = timestamp
    # model_params.print_explanation()
      
    trained_model, loss_list = train(model_params.model, # model_params.model is a model name
                             model_path, model_params, model_params.timestamp)

    print("\nTesting the trained model +++++++++++++++++++++++")
    test(model_params.model, trained_model, model_params, model_params.timestamp)
    
    print("Test model finished +++++++++++++++++++++++++++++")
