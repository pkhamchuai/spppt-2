# import sys
import argparse
import numpy as np
import os
import cv2
import torch
import matplotlib.pyplot as plt

from datetime import datetime

import torch
torch.manual_seed(9793047918980052389)
print('Seed:', torch.seed())

from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import ModelParams, print_summary
from utils import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

image_size = 256

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
    grid = F.affine_grid(matrix, [1, 1, H, W])
    # Sample the input image at the grid points
    
    # if image and matrix are on the same device
    if image.device == matrix.device:
        transformed_image = F.grid_sample(image, grid)
    else:
        transformed_image = F.grid_sample(image.to(matrix.device), grid)
    return transformed_image

# from utils.SuperPoint import SuperPointFrontend
# from utils.utils1 import transform_points_DVF
def test(model_name, models, model_params, timestamp):
    # model_name: name of the model
    # model: model to be tested
    # model_params: model parameters
    # timestamp: timestamp of the model
    print('Test 1way with reverse registration in every steps')
    print(f"Function input: {len(model_name)} models, {models}, {model_params}, {timestamp}")

    test_dataset = datagen(model_params.dataset, False, model_params.sup)

    model = [0]*len(models)
    # load the models one-by-one
    for i in range(len(models)):
        # if model is a string, load the model
        # if model is a loaded model, use the model
        if isinstance(models[i], str):
            print(f"\nLoading model: {models[i]}")
            model[i] = model_loader(model_name, model_params)
            buffer = io.BytesIO()
            torch.save(model[i].state_dict(), buffer)
            buffer.seek(0)
            model[i].load_state_dict(torch.load(models[i]))
            # print(f'Loaded model from {model[i]}')
        elif isinstance(models[i], nn.Module):
            print(f'Using model {model_name}')
            model[i] = models[i]

        # Set model to training mode
        model[i].eval()

    # Create output directory
    output_dir = f"output/{model_name}_{model_params.get_model_code()}_{timestamp}_ensemble_1way_reverse_test"
    os.makedirs(output_dir, exist_ok=True)

    # Validate model
    # validation_loss = 0.0

    metrics = []
    # create a csv file to store the metrics
    csv_file = f"{output_dir}/metrics.csv"

    with torch.no_grad():
        testbar = tqdm(test_dataset, desc=f'Testing:')
        for i, data in enumerate(testbar, 0):

            # source -> target +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            
            # Get images and affine parameters
            source_image, target_image, affine_params_true, points1, points2, points1_2_true = data

            source_image = source_image.requires_grad_(True).to(device)
            target_image = target_image.requires_grad_(True).to(device)
            # add gradient to the matches
            points1 = points1.requires_grad_(True).to(device)
            points2 = points2.requires_grad_(True).to(device)
            points1_2_predicted = points1.clone().to(device)
            # points2_1_predicted = points2.clone().to(device)
            
            # TODO: how to repeat the test?
            # 1. until the affine parameters are not change anymore
            # 2. until the mse is not change anymore
            # 3. until the mse is not change anymore and the affine parameters are not change anymore

            # use for loop with a large number of iterations 
            # check TRE of points1 and points2
            # if TRE grows larger than the last iteration, stop the loop
            metrics_points_best, mse_images_best = np.inf, np.inf
            mse12 = 0
            tre12 = 0

            mse_before_first, tre_before_first, mse12_image_before_first, \
                ssim12_image_before_first = 0, 0, 0, 0
            mse_before, tre_before, mse12_image, ssim12_image = 0, 0, 0, 0

            rep = 10
            votes = [np.inf] * rep  # Initialize a list to store the votes for each model
            metrics_points = [np.inf] * 5 * 2
            mse_images = [np.inf] * 5 * 2
            no_improve = 0
            # accepted parameters
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            M = torch.from_numpy(M).unsqueeze(0)#.to(device)

            for j in range(rep):
                for k in range(len(models)):
                    # Forward + backward + optimize
                    # print input devices
                    outputs = model[k](source_image, target_image, points1)
                    transformed_source_affine = outputs[0]
                    affine_params_predicted = outputs[1]
                    points1_2_predicted = outputs[2]

                    if i < 10 and model_params.plot == 0:
                        plot_ = True
                    else:
                        plot_ = False

                    # print points shape
                    if points1.shape[0] != 1 and points1.shape[-1] != 2:
                        points1 = points1.clone().view(1, -1, 2)
                    # print(points1.shape, points2.shape, points1_2_predicted.shape)
                    results = DL_affine_plot(f"test_{i}", output_dir,
                        f"{i+1}", f"fw_rep{j:02d}_{k}", source_image[0, 0, :, :].cpu().numpy(), 
                        target_image[0, 0, :, :].cpu().numpy(), 
                        transformed_source_affine[0, 0, :, :].cpu().numpy(),
                        points1[0].clone().cpu().detach().numpy().T, 
                        points2[0].cpu().detach().numpy().T, 
                        points1_2_predicted[0].cpu().detach().numpy().T, None, None, 
                        affine_params_true=affine_params_true,
                        affine_params_predict=affine_params_predicted, 
                        heatmap1=None, heatmap2=None, plot=plot_)

                    mse_before = results[1]
                    tre_before = results[3]
                    mse12_image_before = results[5]
                    ssim12_image_before = results[7]

                    # mse12 = results[2]
                    tre12 = results[4]
                    mse12_image = results[6]
                    # ssim12_image = results[8]

                    metrics_points[2*k] = tre12
                    mse_images[2*k] = mse12_image

                    if j == 0 and k == 0:
                        mse_before_first, tre_before_first, mse12_image_before_first, \
                            ssim12_image_before_first = mse_before, tre_before, mse12_image_before, ssim12_image_before
                        # print(mse_before_first, tre_before_first, mse12_image_before_first, ssim12_image_before_first)
                
                    # apply the registration in the reverse direction
                    outputs = model[k](target_image, source_image, points2)
                    # transformed_source_affine_rv = outputs[0]
                    affine_params_predicted_rv = outputs[1]
                    # points2_1_predicted_ = outputs[2]

                    affine_params_predicted = matrix_to_params(
                        torch.inverse(params_to_matrix(affine_params_predicted_rv)))
                    transformed_source_affine = tensor_affine_transform0(source_image,
                                        affine_params_predicted.to(device))
                    points1_2_predicted = transform_points_DVF(points1.clone().view(2, -1, 1).cpu().detach(),
                        affine_params_predicted, source_image)

                    if points1.shape[0] != 1:
                        points1 = points1.clone().view(1, -1, 2)
                    if points1_2_predicted.shape[0] != 1:
                        points1_2_predicted = points1_2_predicted.view(1, -1, 2)
                    # print(points1.shape, points2.shape, points2_1_predicted.shape)
                    results = DL_affine_plot(f"test_{i}", output_dir,
                        f"{i+1}", f"rv_rep{j:02d}_{k}", source_image[0, 0, :, :].cpu().numpy(), 
                        target_image[0, 0, :, :].cpu().numpy(), 
                        transformed_source_affine[0, 0, :, :].cpu().numpy(),
                        points1[0].clone().cpu().detach().numpy().T, 
                        points2[0].cpu().detach().numpy().T, 
                        points1_2_predicted[0].cpu().detach().numpy().T, None, None, 
                        affine_params_true=affine_params_true,
                        affine_params_predict=affine_params_predicted, 
                        heatmap1=None, heatmap2=None, plot=plot_)
                    
                    # mse_before = results[1]
                    # tre_before = results[3]
                    # mse21_image_before = results[5]
                    # ssim21_image_before = results[7]

                    mse12 = results[2]
                    tre12 = results[4]
                    mse12_image = results[6]
                    # ssim21_image = results[8]

                    metrics_points[2*k+1] = tre12
                    mse_images[2*k+1] = mse12_image

                # print(f"Pair {i}, Rep {j}: {metrics_points}, {mse_images}")
                min_metrics_points = np.min([metrics_points])
                min_mse_images = np.min([mse_images])
                best_mse_images = np.argmin([mse_images])
                best_metrics_points = np.argmin([metrics_points])
                
                # if any element in tre_list is nan, use the model with the lowest mse
                if np.isnan(min_metrics_points) or np.isinf(min_metrics_points):
                    best_index = best_mse_images
                else:
                    best_index = best_metrics_points

                print(f"Pair {i}, Rep {j}, best model index: {best_index}")
                best_model = best_index//2

                if best_index % 2 == 0:
                    outputs = model[best_model](source_image, target_image, points1)
                    # transformed_source_affine = outputs[0]
                    affine_params_predicted = outputs[1]
                    # points1_2_predicted = outputs[2]
                    # print(f"best_index: {best_index}, {best_index//2}, {points1_2_predicted.shape}")
                    
                else:
                    outputs = model[best_model](target_image, source_image, points2)
                    # transformed_source_affine = outputs[0]
                    affine_params_predicted_rv = outputs[1]
                    # points2_1_predicted = outputs[2]
                    # points1 = points2_1_predicted.clone()
                    # source_image = transformed_source_affine.clone()

                    # inverse of the affine matrix
                    affine_params_predicted = matrix_to_params(
                        torch.inverse(params_to_matrix(affine_params_predicted_rv)))
                    # print(affine_params_predicted_rv, affine_params_predicted)

                # print input device
                # print(points1.view(2, -1, 1).shape, affine_params_predicted.shape, source_image.shape)
                if points1.shape[0] != 1 and points1.shape[-1] != 2:
                    points1 = points1.clone().view(1, -1, 2)
                points1_2_predicted = transform_points_DVF(points1.clone().view(2, -1, 1).cpu().detach(),
                    affine_params_predicted, source_image)
                transformed_source_affine = tensor_affine_transform0(source_image,
                                        affine_params_predicted.to(device))
                # print(f"best_index: {best_index}, {best_index//2}, {points1_2_predicted.shape}")
                
                if model_params.plot == 1 and i < 50:
                    plot_ = True
                else:
                    plot_ = False

                best_model_text = f"rep{j:02d}_{best_index//2}_fw" if best_index % 2 == 0 else f"rep{j:02d}_{best_index//2}_rv"

                if points1.shape[0] != 1 and points1.shape[-1] != 2:
                    points1 = points1.clone().view(1, -1, 2)
                if points1_2_predicted.shape[0] != 1 and points1_2_predicted.shape[-1] != 2:
                    points1_2_predicted = points1_2_predicted.view(1, -1, 2)

                # print(points1.shape, points2.shape, points1_2_predicted.shape)
                results = DL_affine_plot(f"test_{i}", output_dir,
                    f"{i+1}", best_model_text, source_image[0, 0, :, :].cpu().numpy(),
                    target_image[0, 0, :, :].cpu().numpy(),
                    transformed_source_affine[0, 0, :, :].cpu().numpy(),
                    points1[0].clone().cpu().detach().numpy().T,
                    points2[0].cpu().detach().numpy().T,
                    points1_2_predicted[0].cpu().detach().numpy().T, None, None,
                    affine_params_true=affine_params_true,
                    affine_params_predict=affine_params_predicted,
                    heatmap1=None, heatmap2=None, plot=plot_)
                    
                # mse_before = results[1]
                # tre_before = results[3]
                # mse12_image_before = results[5]
                # ssim12_image_before = results[7]

                mse12 = results[2]
                tre12 = results[4]
                mse12_image = results[6]
                ssim12_image = results[8]

                # points1 = points1_2_predicted.T.clone().to(device)
                # source_image = transformed_source_affine.clone().to(device)

                if min_metrics_points < metrics_points_best or min_mse_images < mse_images_best: # TODO: add SSIM    
                    metrics_points_best = min_metrics_points
                    mse_images_best = min_mse_images
                    votes[j] = best_index

                    # update M
                    # print(affine_params_predicted.shape)
                    M = combine_matrices(M, affine_params_predicted).to(device)
                    # print(M)
                    points1 = points1_2_predicted.clone()
                    source_image = tensor_affine_transform0(data[0].to(device), M)

                    if no_improve > 0: 
                        no_improve -= 1

                else:
                    print(f"No improvement for {no_improve+1} reps")
                    no_improve += 1
                    votes[j] = -1
                    points1 = data[3].clone().to(device)   

                # if there is no improvement for 2 reps, stop the iteration
                if no_improve > 2:
                    break

            # print(f'\nEnd register pair {i}')
            # print(f'Votes: {votes}\n')

            # append metrics to metrics list
            new_entry = [i, mse_before_first, mse12, tre_before_first, tre12, mse12_image_before_first, mse12_image, \
                            ssim12_image_before_first, ssim12_image, np.max(points1_2_predicted.shape), votes]
            metrics.append(new_entry)
            # print(f"Pair {i}: {new_entry}")
            if i > 3:
                break

    with open(csv_file, 'w', newline='') as file:

        writer = csv.writer(file)
        writer.writerow(["index", "mse_before", "mse12", "tre_before", "tre12", "mse12_image_before", "mse12_image", 
                         "ssim12_image_before", "ssim12_image", "num_points", "votes"])
        for i in range(len(metrics)):
            writer.writerow(metrics[i])
        
        # drop the last column of the array 'metrics'
        metrics = [metrics[i][0:-1] for i in range(len(metrics))]
        metrics = np.array(metrics)

        # metrics = metrics[:, :8]
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

    # extra_text = f"Test model {model_name} at {model_} with dataset {model_params.dataset}. "
    # print_summary(model_name, model_, model_params, 
    #               None, timestamp, test=True)

if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')    
    parser.add_argument('--dataset', type=int, default=0, help='dataset number')
    parser.add_argument('--sup', type=int, default=1, help='supervised learning (1) or unsupervised learning (0)')
    parser.add_argument('--image', type=int, default=1, help='image used for training')
    parser.add_argument('--heatmaps', type=int, default=0, help='use heatmaps (1) or not (0)')
    parser.add_argument('--loss_image', type=int, default=0, help='loss function for image registration')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='decay rate')
    parser.add_argument('--model', type=str, default='DHR', help='which model to use')
    parser.add_argument('--model_path', type=str, default=None, help='path to model to load')
    parser.add_argument('--plot', type=int, default=1, help='plot the results')
    args = parser.parse_args()

    # model_path = 'trained_models/' + args.model_path
    # model_path = ['DHR_11100_0.001_0_5_100_20240509-155916.pth', 'DHR_21100_0.001_0_5_100_20240509-160207.pth',
    #           'DHR_31100_0.001_0_10_100_20240508-120807.pth', 'DHR_41100_0.001_0_5_100_20240509-133824.pth',
    #           'DHR_51100_0.001_0_5_100_20240509-140837.pth']
    # add 'trained_models/' in front of each element of model_path

    # take only string between '[' and ']', then between "'" and "'"
    args.model_path = args.model_path[1:-1]
    # take only string between "'" and "'"
    for char in ["'", " "]:
        args.model_path = args.model_path.replace(char, "")
    
    # split the string into a list
    model_path = args.model_path.split(',')
    # print(model_path)

    model_path = ['trained_models/' + path for path in model_path]
    
    model_params = ModelParams(dataset=args.dataset, sup=args.sup, image=args.image, 
                               loss_image=args.loss_image, num_epochs=args.num_epochs, 
                               learning_rate=args.learning_rate, decay_rate=args.decay_rate, plot=args.plot)
    model_params.print_explanation()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # save the output of print_explanation() and loss_list to a txt file
    # print_summary(args.model, model_path, model_params, None, timestamp, True)

    print(f"\nTesting the trained model: {args.model} +++++++++++++++++++++++")

    test(args.model, model_path, model_params, timestamp)
    print("Test model finished +++++++++++++++++++++++++++++")
