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
from utils.utils2 import *
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
    grid = F.affine_grid(matrix, [1, 1, H, W], align_corners=False)
    # Sample the input image at the grid points
    transformed_image = F.grid_sample(image, grid, align_corners=False)
    return transformed_image

# def transform_points(points, H, center=None):
#     """
#     Transform a single point using a homography matrix.
#     """
#     # if points are not in homogeneous form, convert them
#     if len(points) == 2:
#         points = np.array([points[0], points[1], np.ones_like(points[0])], dtype=np.float32)
    
#     # print(points.shape, H.shape)

#     if center is not None:
#         # Translate points to the origin
#         translation_matrix = np.array([
#             [1, 0, -center[0]],
#             [0, 1, -center[1]],
#             [0, 0, 1]
#         ], dtype=np.float32)
#         points = np.dot(translation_matrix, points)
    
#     transformed_point = np.dot(H, points)
#     transformed_point = np.array([transformed_point[0], transformed_point[1], np.ones_like(transformed_point[0])], dtype=np.float32)
#     # print(transformed_point.shape)
    
#     if center is not None:
#         # Translate points back from the origin
#         translation_matrix = np.array([
#             [1, 0, center[0]],
#             [0, 1, center[1]],
#             [0, 0, 1]
#         ], dtype=np.float32)
#         transformed_point = np.dot(translation_matrix, transformed_point)
    
#     return transformed_point[:2]


# from utils.SuperPoint import SuperPointFrontend
# from utils.utils1 import transform_points_DVF
def test(model_name, models, model_params, timestamp, verbose=False, plot=1, beam=1):
    # model_name: name of the model
    # model: model to be tested
    # model_params: model parameters
    # timestamp: timestamp of the model

    def reg(model, source_image, target_image, i, j, b, k, output_dir, 
            points1=None, points2=None, plot_=False):
        # Get the predicted affine parameters and transformed source image
        outputs = model(source_image, target_image, points=points1)
        transformed_source_affine = outputs[0]
        affine_params_predicted = outputs[1]
        points1_2_predicted = outputs[2]
        # print(points1.shape)

        if points1.shape[0] == 2:
            points1 = points1.T

        # # check type of points1, if it's a tensor, convert it to numpy
        # if isinstance(points1, torch.Tensor):
        #     points1 = points1.cpu().detach().numpy()
        # print(points1.shape)
        # points1_2_predicted = transform_points(points1.T, affine_params_predicted[0].cpu().detach().numpy(), 
        #                                        center=[image_size/2, image_size/2]).T
        # print(points1_2_predicted.shape, points2.shape, points1.shape)

        if points1 is not None and points2 is not None:
            # print(points1.shape, points2.shape, points1_2_predicted.shape)
            results = DL_affine_plot(f"test_{i}", output_dir,
                f"{i+1}", f"rep{j:02d}_beam{b}_branch_{k}", 
                source_image[0, 0, :, :].cpu().numpy(),
                target_image[0, 0, :, :].cpu().numpy(), 
                transformed_source_affine[0, 0, :, :].cpu().numpy(),
                points1[0].cpu().detach().numpy().T,
                points2[0].cpu().detach().numpy().T,
                points1_2_predicted[0].cpu().detach().numpy().T,
                None, None, 
                affine_params_true=affine_params_true,
                affine_params_predict=affine_params_predicted, 
                heatmap1=None, heatmap2=None, plot=plot_)
            return results, affine_params_predicted
        else:
            points1_2_predicted = None
            results = DL_affine_plot_image(f"test_{i}", output_dir,
                i+1, f"rep{j:02d}_beam{b}_branch_{k}", 
                source_image[0, 0, :, :].cpu().numpy(),
                target_image[0, 0, :, :].cpu().numpy(),
                transformed_source_affine[0, 0, :, :].cpu().numpy(),
                None, None, None,
                None, None,
                affine_params_true=affine_params_true,
                affine_params_predict=affine_params_predicted,
                heatmap1=None, heatmap2=None, plot=plot_)
            return results

    print('Test 1-way BCS image with reverse registration')
    print(f"Function input:', {model_name},\n{model_params},\n{timestamp}")
    print(f"Plot: {plot}, Beam: {beam}")

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
    output_dir = f"output/{model_name}_{model_params.get_model_code()}_{timestamp}_BCS_1way_beam{beam}_image_reverse_test"
    os.makedirs(output_dir, exist_ok=True)

    # Validate model
    # validation_loss = 0.0

    metrics = []
    # create a csv file to store the metrics
    csv_file = f"{output_dir}/metrics.csv"

    with torch.no_grad():
        testbar = tqdm(test_dataset, desc=f'Testing:')
        for i, data in enumerate(testbar, 0):
            # if i > 3:
            #     break

            # Get images and affine parameters
            source_image, target_image, affine_params_true, points1_0, points2, _ = data

            source_image0 = source_image.requires_grad_(True).to(device)
            target_image = target_image.requires_grad_(True).to(device)
            # add gradient to the matches
            points1_0 = points1_0.requires_grad_(True).to(device)
            points2 = points2.requires_grad_(True).to(device)
            # print(points1_0.shape)
            # if isinstance(points1_0, torch.Tensor):
            #     points1_0 = points1_0[0].cpu().detach().numpy()
            #     points2 = points2[0].cpu().detach().numpy()
            # print(points1_0.shape, points2.shape)

            # TODO: how to repeat the test?
            # 1. until the affine parameters are not change anymore
            # 2. until the mse is not change anymore
            # 3. until the mse is not change anymore and 
            #    the affine parameters are not change anymore

            # use for loop with a large number of iterations 
            # check TRE of points1 and points2
            # if TRE grows larger than the last iteration, stop the loop
            metric_best = np.inf
            mse12 = np.inf
            tre12 = np.inf

            mse_before_first, tre_before_first, mse12_image_before_first, \
                ssim12_image_before_first = np.inf, np.inf, np.inf, np.inf
            # mse_before, tre_before, mse12_image, ssim12_image = 0, 0, 0, 0

            rep = 10 # Number of repetitions
            votes = []
            
            no_improve = 0
            # not using intermediate source images anymore
            # re-create the resulting image in every iteration
            
            # Active beams will track which beams are still active
            # Defined as a dictionary with the beam index as 
            # the key and the number of votes as the value
            active_beams_index = list(b for b in range(beam))
            active_beams = [0]*beam

            for j in range(rep):
                if verbose:
                    print(f"\nPair {i}, Rep {j}, active beams {active_beams}")

                if j == 0:
                    # beam_info = []
                    beam_info = [[m] for m in range(2*len(models))]
                    # print(beam_info)
                else:
                    # make b copies of beam_info
                    beam_info = []
                    for b in range(len(active_beams)):
                        for k in range(2*len(models)):
                            # append values of active beams
                            temp = active_beams[b].copy()
                            temp.append(k)
                            beam_info.append(temp)

                if verbose:
                    print(f"\nRep {j}: beam_info {beam_info}")
                    print(f"Lenght of beam_info: {len(beam_info)}")

                metrics_points_forward = []
                metrics_images_forward = []

                for b in beam_info:
                    # print(b)
                    if verbose:
                        print(f"\nRep {j}, Beam {b}: doing registration")
                        
                    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
                    M = torch.from_numpy(M).unsqueeze(0)#.to(device)

                    for k in range(len(b)):
                        if k == 0:
                            points1 = points1_0.clone().to(device)
                            source_image = source_image0.clone().to(device)
                        # print(f"{k}, Pair {i}, Rep {j}, Beam {b}, Model {[b[k]]}")
                        # print(points1.shape, points2.shape)

                        if plot == 0 and i < 10 and k == len(b)-1:
                            plot_ = True
                        else:
                            plot_ = False

                        if b[k] < len(models):
                            _, affine_params_predicted = reg(model[b[k]], source_image, target_image, 
                                i, j, b, k, output_dir, points1=points1, points2=points2, plot_=plot_)
                        else:
                            # offset model number by the number of models
                            model_number = b[k]-len(models)
                            _, affine_params_predicted_rv = reg(model[model_number], target_image, source_image, 
                                i, j, b, k, output_dir, points1=points2, points2=points1, plot_=plot_)                            
                            # inverse the affine parameters
                            affine_params_predicted = matrix_to_params(
                                torch.inverse(params_to_matrix(affine_params_predicted_rv))).to(device)

                        M = combine_matrices(M, affine_params_predicted).to(device)
                        source_image = tensor_affine_transform0(source_image0, M)
                        points1 = transform_points_DVF(points1_0.cpu().detach().T,
                                        M.cpu().detach(), source_image0).T
                        
                        if k == len(b)-1:
                            tre12 = tre(points1.cpu().detach().numpy(), points2.cpu().detach().numpy())
                            mse12_image = mse(source_image[0, 0, :, :].cpu().detach().numpy(), 
                                              target_image[0, 0, :, :].cpu().detach().numpy())
                        
                            metrics_points_forward.append(tre12)
                            metrics_images_forward.append(mse12_image)

                        if verbose:
                            print(f"Pair {i}, Rep {j}, Beam {b}, Model {[b[k]]}: {tre12}, {mse12_image}")
                    
                    # join the metrics of the forward and reverse directions
                    metrics_points = np.array(metrics_points_forward)
                    metrics_images = np.array(metrics_images_forward)

                if verbose:
                    print(f"Pair {i}, Rep {j}, points: {metrics_points}")
                    print(f"\t\timages: {metrics_images}")
                    print(f"Length of metrics_points: {len(metrics_points)}")
                
                # choose the best 'beam' models
                best_index_points = np.argsort(metrics_points)[:beam]
                best_index_images = np.argsort(metrics_images)[:beam]
                min_metrics_points = np.min([metrics_points])

                
                if verbose:
                    print(f"Pair {i}, Rep {j}, best model: points {best_index_points}, images {best_index_images}")
                
                # if any element in tre_list is nan, use the model with the lowest mse
                # ++++++++++++++++ this part must be changed to be cases later ++++++++++++++++
                if np.isnan(min_metrics_points):
                    best_index = best_index_images
                    metric_this_rep = np.min([metrics_images])
                    if verbose:
                        print(f"using images: {metric_this_rep}, {best_index}")
                else:
                    best_index = best_index_points
                    metric_this_rep = min_metrics_points
                    if verbose:
                        print(f"using points: {metric_this_rep}, {best_index}")

                if verbose:
                    print(f"Pair {i}, Rep {j}, best model index: {best_index}")

                # print(beam_info, best_index)
                for b in range(beam):
                    active_beams_index[b] = best_index[b]
                    active_beams[b] = beam_info[best_index[b]]

                if plot == 1 and i < 50:
                    for b in range(beam):
                        if verbose:
                            print(f"Processing active beam {b}: {active_beams[b]}")

                        # loop through the active beam path to collect all transformations and combine into one
                        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
                        M = torch.from_numpy(M).unsqueeze(0).to(device)

                        for k in range(len(active_beams[b])):
                            if k == 0:
                                points1 = points1_0.clone().to(device)
                                source_image = source_image0.clone().to(device)

                            model_number = active_beams[b][k]
                            if model_number < len(models):
                                outputs = model[model_number](source_image, target_image, points=points1)
                                affine_params_predicted = outputs[1]
                            else:
                                model_number = model_number - len(models)
                                outputs = model[model_number](target_image, source_image, points=points2)
                                affine_params_predicted_rv = outputs[1]
                                affine_params_predicted = matrix_to_params(
                                    torch.inverse(params_to_matrix(affine_params_predicted_rv))).to(device)

                            M = combine_matrices(M, affine_params_predicted).to(device)
                            
                            if k == len(active_beams[b])-1:
                                transformed_source_affine = tensor_affine_transform0(source_image0, M)
                                points1_2_predicted = transform_points_DVF(points1_0.cpu().detach().T,
                                    M.cpu().detach(), source_image0).T
                            else:
                                source_image = tensor_affine_transform0(source_image0, M)
                                points1 = transform_points_DVF(points1_0.cpu().detach().T,
                                            M.cpu().detach(), source_image0).T

                            if k == len(active_beams[b])-1:
                                # plot_ = 1
                                _ = DL_affine_plot(f"test_{i}", output_dir,
                                    i+1, f"b{b}_{active_beams[b]}",
                                    source_image[0, 0, :, :].cpu().numpy(),
                                    target_image[0, 0, :, :].cpu().numpy(),
                                    transformed_source_affine[0, 0, :, :].cpu().numpy(),
                                    points1[0].cpu().detach().numpy().T,
                                    points2[0].cpu().detach().numpy().T,
                                    points1_2_predicted[0].cpu().detach().numpy().T,
                                    None, None,
                                    affine_params_true=affine_params_true,
                                    affine_params_predict=affine_params_predicted,
                                    # affine_params_predict=M,
                                    heatmap1=None, heatmap2=None, plot=True)
                
                if verbose:
                    for b in range(beam):
                        print(f"\nActive beam {b}: {active_beams[b]}")
                        print(f"Active beam index {b}: {active_beams_index[b]}")
                    print('\n')

                # apply the best model to this pair
                # if mse12 < mse_before or mse12_image < mse12_image_before:
                # TODO: if tre12 is not available, use mse12_image instead
                
                if metric_this_rep < metric_best:
                    metric_best = metric_this_rep
                    # mse_images_best = mse12_image
                    if no_improve > 0: 
                        no_improve -= 1
                else:
                    if verbose:
                        print(f"No improvement for {no_improve+1} reps")
                    no_improve += 1

                # if verbose:
                #     print(f"Done: Pair {i}, Rep {j}: search path {active_beams}")

                if no_improve > 2:
                    # do final things
                    if verbose:
                        print(f"Pair {i}, Rep {j}: DONE (no improve) final search path {active_beams}")
                    active_beams = active_beams[0][0:-3] # if no improvement, need to set back 3 steps to get the best result
                    break
                elif j == rep-1:
                    if verbose:
                        print(f"Pair {i}, Rep {j}: DONE (end iter) final search path {active_beams}")
                    active_beams = active_beams[0]
                    break

            # Finalize the results
            # loop through the active beam path to collect all transformations and combine into one
            M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            M = torch.from_numpy(M).unsqueeze(0)#.to(device)
            source_image = source_image0.clone().to(device)
            points1 = points1_0.clone().to(device)

            if verbose:
                print(f"\nFinalizing pair {i}: {active_beams}")

            for k in range(len(active_beams)):
                model_number = active_beams[k]
                if model_number < len(models):
                    outputs = model[model_number](source_image, target_image, points=points1)
                    affine_params_predicted = outputs[1]
                else:
                    model_number = model_number - len(models)
                    outputs = model[model_number](target_image, source_image, points=points2)
                    affine_params_predicted_rv = outputs[1]
                    affine_params_predicted = matrix_to_params(
                        torch.inverse(params_to_matrix(affine_params_predicted_rv))).to(device)
                    
                M = combine_matrices(M, affine_params_predicted).to(device)

                if k == len(active_beams)-1:
                    transformed_source_affine = tensor_affine_transform0(source_image0, M)
                    points1_2_predicted = transform_points_DVF(points1_0.cpu().detach().T,
                                M.cpu().detach(), source_image0).T
                else:
                    source_image = tensor_affine_transform0(source_image0, M)
                    points1 = transform_points_DVF(points1_0.cpu().detach().T,
                                M.cpu().detach(), source_image0).T

            if i < 100 and (plot == 1 or plot == 2):
                plot_ = True
            elif plot == 3:
                plot_ = False

            best_model_text = f"final_{active_beams}"
            _ = DL_affine_plot(f"test_{i}", output_dir,
                i+1, best_model_text, 
                source_image0[0, 0, :, :].cpu().numpy(),
                target_image[0, 0, :, :].cpu().numpy(),
                source_image[0, 0, :, :].cpu().numpy(),
                points1_0[0].cpu().detach().numpy().T,
                points2[0].cpu().detach().numpy().T,
                points1_2_predicted[0].cpu().detach().numpy().T,
                None, None,
                affine_params_true=affine_params_true,
                affine_params_predict=M,
                heatmap1=None, heatmap2=None, plot=plot_)
            
            points1_0 = points1_0.cpu().detach().numpy().T
            points1_2_predicted = points1_2_predicted.cpu().detach().numpy().T
            points2 = points2.cpu().detach().numpy().T

            source_image0 = source_image0[0, 0, :, :].cpu().detach().numpy()
            source_image = source_image[0, 0, :, :].cpu().detach().numpy()
            target_image = target_image[0, 0, :, :].cpu().detach().numpy()

            votes = active_beams
            mse_before_first = mse(points1_0, points2)
            mse12 = mse(points1_2_predicted, points2)
            tre_before_first = tre(points1_0, points2)
            tre12 = tre(points1_2_predicted, points2)
            mse12_image_before_first = mse(source_image0, target_image)
            mse12_image = mse(source_image, target_image)
            ssim12_image_before_first = ssim(source_image0, target_image)
            ssim12_image = ssim(source_image, target_image)

            # append metrics to metrics list
            new_entry = [i, mse_before_first, mse12, tre_before_first, tre12, mse12_image_before_first, mse12_image, \
                            ssim12_image_before_first, ssim12_image, np.max(points1_2_predicted.shape), votes]
            metrics.append(new_entry)

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
    # print_summary(model_name, None, model_params, 
    #               None, timestamp, test=True, output_dir=output_dir)

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
    '''
    plot = 0: plot every steps
    plot = 1: plot only the search path
    plot = 2: plot only the final result
    plot = 3: plot nothing
    '''
    parser.add_argument('--verbose', type=int, default=0, help='verbose output')
    parser.add_argument('--beam', type=int, default=1, help='beam search width')
    parser.add_argument('--rep', type=int, default=10, help='number of repetitions')
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
                               learning_rate=args.learning_rate, decay_rate=args.decay_rate,
                               plot=args.plot, rep=args.rep)
    model_params.print_explanation()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # save the output of print_explanation() and loss_list to a txt file
    # print_summary(args.model, model_path, model_params, None, timestamp, True)

    print(f"\nTesting the trained model: {args.model} +++++++++++++++++++++++")

    args.verbose = int(args.verbose)
    print(f"verbose: {args.verbose}")
    test(args.model, model_path, model_params, timestamp, 
         verbose=args.verbose, plot=args.plot, beam=args.beam)
    print("Test model finished +++++++++++++++++++++++++++++")
