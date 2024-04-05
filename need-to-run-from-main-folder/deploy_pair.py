# import sys
import argparse
import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt
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

torch.manual_seed(9793047918980052389)
print('Seed:', torch.seed())

from utils.utils0 import *
from utils.utils1 import *
# from utils.utils1 import ModelParams, print_summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

image_size = 256

# from utils.SuperPoint import SuperPointFrontend
# from utils.utils1 import transform_points_DVF

# TODO: create its own ModelParams, print_summary
class ModelParams:
    def __init__(self, dataset, sup, model_name, model_path, source_image, target_image, idx):
        self.dataset = dataset
        self.sup = sup
        self.model_name = model_name
        self.model_path = model_path
        self.source_image = source_image
        self.target_image = target_image
        self.idx = idx

    def get_model_code(self):
        if self.model_name == 'SuperPoint':
            return 'SP'
        elif self.model_name == 'SuperPoint+Superglue':
            return 'SPSG'

        
    def print_explanation(self):
        print(f"Dataset: {self.dataset}")
        print(f"Supervised: {self.sup}")
        print(f"Model name: {self.model_name}")
        print(f"Model path: {self.model_path}")
        print(f"Source image: {self.source_image}")
        print(f"Target image: {self.target_image}")
        print(f"Index: {self.idx}")

# class ModelParams:
#     def __init__(self, name=None, model_path=None, dataset=0, sup=0, image=1, 
#                  heatmaps=0, loss_image=0, learning_rate=0.001, decay_rate = 0.96, 
#                  num_epochs=10, batch_size=1, source_image=None, target_image=None, 
#                  idx=None):
#         # dataset: dataset used
#         # dataset=0: actual eye
#         # dataset=1: synthetic eye easy
#         # dataset=2: synthetic eye medium
#         # dataset=3: synthetic eye hard
#         # dataset=4: synthetic shape
#         # sup: supervised or unsupervised model
#         # sup=0: unsupervised model
#         # sup=1: supervised model
#         # image: image type
#         # image=0: image not used
#         # image=1: image used
#         # heatmaps: heatmaps used
#         # heatmaps=0: heatmaps not used
#         # heatmaps=1: heatmaps used
#         # heatmaps=2: enhanced heatmaps used
#         # loss_image: loss function for image
#         # loss_image=0: MSE
#         # loss_image=1: NCC
#         # loss_image=2: MSE + SSIM
#         # loss_image=3: MSE + NCC
#         # loss_image=4: MSE + SSIM + NCC
#         # loss_affine is depending on sup
#         # loss_affine: loss function for affine
#         # loss_affine=0: loss_extra
#         # loss_affine=1: loss_affine

#         self.dataset = dataset
#         if self.dataset == 0:
#             self.sup = 0
#         else:
#             self.sup = sup
#         self.image = image
#         self.heatmaps = heatmaps

#         self.learning_rate = learning_rate
#         self.decay_rate = decay_rate
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         # print("Learning rate: ", learning_rate)
#         # print("Number of epochs: ", num_epochs)
#         # print("Batch size: ", batch_size)
#         # print("Loss image: ", 'MSE' if self.loss_image == 0 else 
#         #                      'NCC' if self.loss_image == 1 else 
#         #                      'MSE + SSIM' if self.loss_image == 2 else 
#         #                      'MSE + NCC' if self.loss_image == 3 else 
#         #                      'MSE + SSIM + NCC')
    
#         if loss_image == 0:
#             self.loss_image_case = 0
#             self.loss_image = nn.MSELoss()
#         elif loss_image == 1:
#             self.loss_image_case = 1
#             self.loss_image = NCC()
#         elif loss_image == 2:
#             self.loss_image_case = 2
#             self.loss_image = MSE_SSIM()
#         elif loss_image == 3:
#             self.loss_image_case = 3
#             self.loss_image = MSE_NCC()
#         elif loss_image == 4:
#             self.loss_image_case = 4
#             self.loss_image = MSE_SSIM_NCC()
        
#         if sup == 1:
#             self.loss_affine = loss_affine()
#         elif sup == 0:
#             # self.loss_affine = loss_extra()
#             self.loss_affine = None

#         self.start_epoch = 0
#         self.name = name
#         self.model_path = model_path
#         self.model_name = self.get_model_name()
#         self.model_code = self.get_model_code()
#         print('Model name: ', self.model_name)
#         print('Model code: ', self.model_code)
#         print('Model params: ', self.to_dict())

#     def get_model_name(self):
#         # model name code
#         model_name = str(self.name) + '_dataset' + str(self.dataset) + '_sup' + str(self.sup) + '_image' + str(self.image) + \
#             '_heatmaps' + str(self.heatmaps) + '_loss_image' + str(self.loss_image_case)
#         return model_name
    
#     def get_model_code(self):
#         # model code
#         model_code = str(self.name) + '_' + str(self.dataset) + str(self.sup) + str(self.image) + \
#             str(self.heatmaps) + str(self.loss_image_case) + \
#             '_' + str(self.learning_rate) + '_' + str(self.start_epoch) + '_' + \
#                 str(self.num_epochs) + '_' + str(self.batch_size)
#         return model_code

#     def to_dict(self):
#         return {
#             'name': self.name,
#             'model_path': self.model_path,
#             'dataset': self.dataset,
#             'sup': self.sup,
#             'image': self.image,
#             'heatmaps': self.heatmaps,
#             'loss_image_case': self.loss_image_case,
#             'loss_image': self.loss_image,
#             'loss_affine': self.loss_affine,
#             'learning_rate': self.learning_rate,
#             'decay_rate': self.decay_rate,
#             'start_epoch': self.start_epoch, 
#             'num_epochs': self.num_epochs,
#             'batch_size': self.batch_size,
#             'model_name': self.model_name,
#             'model_code': self.model_code,
#             'source_image': self.source_image,
#             'target_image': self.target_image,
#             'idx': self.idx
#         }

#     # @classmethod
#     # def from_model_name(cls, model_name):
#     #     sup = int(model_name.split('_')[0][3:]) if model_name.split('_')[0][3:] else 0
#     #     dataset = int(model_name.split('_')[1][7:]) if model_name.split('_')[1][7:] else 0
#     #     image = int(model_name.split('_')[2][5:]) if model_name.split('_')[2][5:] else 0
#     #     heatmaps = int(model_name.split('_')[3][8:]) if model_name.split('_')[3][8:] else 0
#     #     loss_image_case = int(model_name.split('_')[4][10:]) if model_name.split('_')[4][10:] else 0
#     #     return cls(sup, dataset, image, heatmaps, loss_image_case)

#     # @classmethod
#     # def model_code_from_model_path(cls, model_code):
#     #     # print(model_code.split('/')[-1].split('_')[0:-1])
#     #     split_string = model_code.split('/')[-1].split('_')[0:-1]
#     #     dataset = int(split_string[0][0])
#     #     sup = int(split_string[0][1])
#     #     image = int(split_string[0][2])
#     #     heatmaps = int(split_string[0][3])
#     #     loss_image_case = int(split_string[0][4])
#     #     learning_rate = float(split_string[1])
#     #     if len(split_string) == 5:
#     #         start_epoch = int(split_string[2])
#     #         num_epochs = int(split_string[3])
#     #         batch_size = int(split_string[4])
#     #     else:
#     #         start_epoch = 0
#     #         num_epochs = int(split_string[2])
#     #         batch_size = int(split_string[3])
#     #     # decay_rate = 0.96
#     #     return cls(dataset, sup, image, heatmaps, loss_image_case, learning_rate, start_epoch, num_epochs, batch_size)
    
#     # @classmethod
#     # def from_dict(cls, model_dict):
#     #     return cls(model_dict['dataset'], model_dict['sup'], model_dict['image'], \
#     #                model_dict['heatmaps'], model_dict['loss_image'])
    
#     def __str__(self):
#         return self.model_name
    
#     def print_explanation(self):
#         print('\nModel name: ', self.model_name)
#         print('Model: ', self.name)
#         print('Model path: ', self.model_path)
#         print('Model code: ', self.model_code)
#         print('Dataset used: ', 'Actual eye' if self.dataset == 0 else \
#                 'Synthetic eye easy' if self.dataset == 1 else \
#                 'Synthetic eye medium' if self.dataset == 2 else \
#                 'Synthetic eye hard' if self.dataset == 3 else \
#                 'Synthetic shape')
#         print('Supervised or unsupervised model: ', 'Supervised' if self.sup else 'Unsupervised')
#         print('Image type: ', 'Image not used' if self.image == 0 else \
#                 'Image used')
#         print('Heatmaps used: ', 'Heatmaps not used' if self.heatmaps == 0 else \
#                 'Heatmaps used' if self.heatmaps == 1 else \
#                 'Enhanced heatmaps used')
#         print('Loss function case: ', self.loss_image_case)
#         print('Loss function for image: ', self.loss_image)
#         print('Loss function for affine: ', self.loss_affine)
#         print('Learning rate: ', self.learning_rate)
#         print('Decay rate: ', self.decay_rate)
#         print('Start epoch: ', self.start_epoch)
#         print('Number of epochs: ', self.num_epochs)
#         print('Batch size: ', self.batch_size)
#         # print('Model params: ', self.to_dict())
#         print('Source image: ', self.source_image)
#         print('Target image: ', self.target_image)
#         print('Index: ', self.idx)
#         print('\n')

#     def __repr__(self):
#         return self.model_name


def run_deploy(model_params, timestamp):
   # model_name: name of the model
    # model: model to be tested
    # model_params: model parameters
    # timestamp: timestamp of the model
    print('Deploy input:', model_params, timestamp)

    test_dataset = datagen(model_params.dataset, False, model_params.sup)

    # if model is a string, load the model
    # if model is a loaded model, use the model
    if isinstance(model_params.model_name, str):
        model = model_loader(model_params.model_name, model_params)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(model))
        print(f'Loaded model from {model}')

    # Set model to training mode
    model.eval()

    # Create output directory
    output_dir = f"output/{model_params.get_model_code()}_{timestamp}_deploy"
    os.makedirs(output_dir, exist_ok=True)

    # Validate model
    # validation_loss = 0.0

    metrics = []
    # create a csv file to store the metrics
    csv_file = f"{output_dir}/metrics.csv"

    with torch.no_grad():
        testbar = tqdm(test_dataset, desc=f'Deploying:')
        for i, data in enumerate(testbar, 0):
            # Get images and affine parameters
            if model_params.sup:
                source_image, target_image, affine_params_true = data
            else:
                source_image, target_image = data
                affine_params_true = None
            source_image = source_image.to(device)
            target_image = target_image.to(device)

            # Forward pass
            outputs = model(source_image, target_image)
            # for i in range(len(outputs)):
            #     print(i, outputs[i].shape)
            transformed_source_affine = outputs[0]
            affine_params_predicted = outputs[1]
            points1 = np.array(outputs[2])
            points2 = np.array(outputs[3])
            points1_transformed = np.array(outputs[4])

            try:
                points1_affine = points1_transformed.reshape(points1_transformed.shape[2], points1_transformed.shape[1])
            except:
                pass

            desc1_2 = outputs[5]
            desc2 = outputs[6]
            heatmap1 = outputs[7]
            heatmap2 = outputs[8]

            if i < 100:
                plot_ = True
            else:
                plot_ = False

            results = DL_affine_plot(f"{i+1}", output_dir,
                f"{i}", "_", source_image[0, 0, :, :].cpu().numpy(), target_image[0, 0, :, :].cpu().numpy(), \
                transformed_source_affine[0, 0, :, :].cpu().numpy(), \
                points1, points2, points1_transformed, desc1_2, desc2, affine_params_true=affine_params_true,
                affine_params_predict=affine_params_predicted, heatmap1=heatmap1, heatmap2=heatmap2, plot=plot_)


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
            metrics.append([i, mse_before, mse12, tre_before, tre12, mse12_image_before, mse12_image, ssim12_image_before, ssim12_image, points2.shape[-1]])

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



if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')    
    
    parser.add_argument('--dataset', type=int, default=1, help='dataset number')
    parser.add_argument('--sup', type=int, default=1, help='supervised learning (1) or unsupervised learning (0)')
    parser.add_argument('--model_name', type=str, default=None, help='which model to use')
    parser.add_argument('--model_path', type=str, default=None, help='path to model to load')
    parser.add_argument('--source_image', type=int, default=None, help='path to source image')
    parser.add_argument('--target_image', type=int, default=None, help='path to target image')
    parser.add_argument('--idx', type=int, default=None, help='index of image pair')
    args = parser.parse_args()

    model_path = 'trained_models/' + args.model_path
    deploy_params = ModelParams(dataset=args.dataset, sup=args.sup, model_name=args.model_name, 
                                 model_path=model_path, source_image=args.source_image, 
                                 target_image=args.target_image, idx=args.idx)
    deploy_params.print_explanation()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # save the output of print_explanation() and loss_list to a txt file
    print_summary(deploy_params.model_name, model_path, 
                  deploy_params, None, timestamp, True)

    run_deploy(deploy_params, timestamp)
    print("Test model finished +++++++++++++++++++++++++++++")
