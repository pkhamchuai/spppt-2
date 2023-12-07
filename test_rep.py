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
from utils.utils1 import ModelParams, model_loader, print_summary#, test_repeat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

image_size = 256

# from utils.SuperPoint import SuperPointFrontend
# from utils.utils1 import transform_points_DVF


def test_repeat(model_name, model_, model_params, timestamp):
    # model_name: name of the model
    # model: model to be tested
    # model_params: model parameters
    # timestamp: timestamp of the model
    print('Test function input:', model_name, model_, model_params, timestamp)

    test_dataset = datagen(model_params.dataset, False, model_params.sup)

    # if model is a string, load the model
    # if model is a loaded model, use the model
    if isinstance(model_, str):
        model = model_loader(model_name, model_params)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(model_))
        print(f'Loaded model from {model_}')
    elif isinstance(model_, nn.Module):
        print(f'Using model {model_name}')
        model = model_

    # Set model to training mode
    model.eval()

    # Create output directory
    output_dir = f"output/{model_name}_{model_params.get_model_code()}_{timestamp}_test"
    os.makedirs(output_dir, exist_ok=True)

    # Validate model
    # validation_loss = 0.0

    metrics = []
    # create a csv file to store the metrics
    csv_file = f"{output_dir}/metrics.csv"

    with torch.no_grad():
        testbar = tqdm(test_dataset, desc=f'Testing:')
        for i, data in enumerate(testbar, 0):
            # Get images and affine parameters
            if model_params.sup:
                source_image, target_image, affine_params_true = data
            else:
                source_image, target_image = data
                affine_params_true = None
            source_image = source_image.to(device)
            target_image = target_image.to(device)

            # TODO: how to repeat the test?
            # 1. until the affine parameters are not change anymore
            # 2. until the mse is not change anymore
            # 3. until the mse is not change anymore and the affine parameters are not change anymore

            # use for loop with a large number of iterations 
            # check TRE of points1 and points2
            # if TRE grows larger than the last iteration, stop the loop
            TRE_last = 1e10
            MSE_last = 1e10
            for j in range(10):
                # Forward pass
                outputs = model(source_image, target_image)
                # for i in range(len(outputs)):
                #     print(i, outputs[i].shape)
                # transformed_source_affine = outputs[0]
                affine_params_predicted = outputs[1]

                if j < 0:
                    # reduce the values of affine_params_predicted
                    identity = torch.tensor([[1, 0, 0], [0, 1, 0]]).to(device)
                    affine_params_predicted = (affine_params_predicted + (j)*identity)/(j+1)
                # apply affine transformation to source_image
                transformed_source_affine = tensor_affine_transform(source_image, affine_params_predicted)

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

                if i < 50:
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

                # check if the mse and affine parameters are not change anymore
                # print(np.linalg.norm(transformed_source_affine.cpu().numpy() - source_image.cpu().numpy()),
                #       np.linalg.norm(affine_params_predicted[0].cpu().numpy() - identity))
                if tre12 > TRE_last or mse12 > MSE_last:
                    break
                
                # append metrics to metrics list
                metrics.append([i, mse_before, mse12, tre_before, tre12, mse12_image_before, \
                                mse12_image, ssim12_image_before, ssim12_image, points2.shape[-1]])
                TRE_last = tre12
                MSE_last = mse12
                source_image = transformed_source_affine # update the source image
            
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

    print_summary(model_name, None, model_params, 
                  None, timestamp, True)

if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')    
    parser.add_argument('--dataset', type=int, default=1, help='dataset number')
    parser.add_argument('--sup', type=int, default=1, help='supervised learning (1) or unsupervised learning (0)')
    parser.add_argument('--image', type=int, default=1, help='image used for training')
    parser.add_argument('--heatmaps', type=int, default=0, help='use heatmaps (1) or not (0)')
    parser.add_argument('--loss_image', type=int, default=0, help='loss function for image registration')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='decay rate')
    parser.add_argument('--model', type=str, default=None, help='which model to use')
    parser.add_argument('--model_path', type=str, default=None, help='path to model to load')
    args = parser.parse_args()

    model_path = 'trained_models/' + args.model_path
    model_params = ModelParams(dataset=args.dataset, sup=args.sup, image=args.image, heatmaps=args.heatmaps, 
                               loss_image=args.loss_image, num_epochs=args.num_epochs, 
                               learning_rate=args.learning_rate, decay_rate=args.decay_rate)
    model_params.print_explanation()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # save the output of print_explanation() and loss_list to a txt file
    print_summary(args.model, model_path, model_params, None, timestamp, True)

    print(f"\nTesting the trained model: {args.model} +++++++++++++++++++++++")

    test_repeat(args.model, model_path, model_params, timestamp)
    print("Test model finished +++++++++++++++++++++++++++++")
