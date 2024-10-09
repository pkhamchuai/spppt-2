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
    grid = F.affine_grid(matrix, [1, 1, H, W], align_corners=False)
    # Sample the input image at the grid points
    transformed_image = F.grid_sample(image, grid, align_corners=False)
    return transformed_image

def transform_points(points, H):
    """
    Transform a single point using a homography matrix.
    """
    # if points are not in homogeneous form, convert them
    # print(points.shape)
    if len(points) == 2:
        points = np.array([points[0], points[1], np.ones_like(points[0])], dtype=np.float32)
    transformed_point = np.dot(H, points)
    # transformed_point /= transformed_point[2]
    # print(transformed_point[:2])
    return transformed_point[:2]


# from utils.SuperPoint import SuperPointFrontend
# from utils.utils1 import transform_points_DVF
def test(model_name, models, model_params, timestamp, verbose=False, plot=1, beam=1):
    # model_name: name of the model
    # model: model to be tested
    # model_params: model parameters
    # timestamp: timestamp of the model

    def reg(model, source_image, target_image, i, j, b, k, output_dir, points1=None, points2=None):
        # Get the predicted affine parameters and transformed source image
        outputs = model(source_image, target_image)
        transformed_source_affine = outputs[0]
        affine_params_predicted = outputs[1]
        # print(points1.shape)

        # if i is an odd number
        if i < 10 and model_params.plot == 0:
            plot_ = True
        else:
            plot_ = False

        results = DL_affine_plot_image(f"test_{i}", output_dir,
            f"{i+1}", f"rep{j:02d}_beam{b}_model{k}", source_image[0, 0, :, :].cpu().numpy(), 
            target_image[0, 0, :, :].cpu().numpy(), 
            transformed_source_affine[0, 0, :, :].cpu().numpy(),
            points1, points2, None,
            None, None, 
            affine_params_true=affine_params_true,
            affine_params_predict=affine_params_predicted, 
            heatmap1=None, heatmap2=None, plot=plot_)

        return results

    print('Test 1-way ensemble model')
    print('Function input:', model_name, models, model_params, timestamp)

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
    output_dir = f"output/{model_name}_{model_params.get_model_code()}_{timestamp}_BCS_1way_image_test"
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
            source_image, target_image, affine_params_true, _, _, _ = data

            source_image0 = source_image.requires_grad_(True).to(device)
            target_image = target_image.requires_grad_(True).to(device)
            # add gradient to the matches
            # points1 = points1_0.requires_grad_(True).to(device)
            # points2 = points2.requires_grad_(True).to(device)
            # print(points1_0.shape)
            # if isinstance(points1_0, torch.Tensor):
            #     points1_0 = points1_0[0].cpu().detach().numpy()
            #     points2 = points2[0].cpu().detach().numpy()
            # print(points1_0.shape, points2.shape)

            # TODO: how to repeat the test?
            # 1. until the affine parameters are not change anymore
            # 2. until the mse is not change anymore
            # 3. until the mse is not change anymore and the affine parameters are not change anymore

            # use for loop with a large number of iterations 
            # check TRE of points1 and points2
            # if TRE grows larger than the last iteration, stop the loop
            mse_points_best, mse_images_best = np.inf, np.inf
            # mse12 = np.inf
            # tre12 = np.inf

            mse_before_first, tre_before_first, mse12_image_before_first, \
                ssim12_image_before_first = np.inf, np.inf, np.inf, np.inf
            # mse_before, tre_before, mse12_image, ssim12_image = 0, 0, 0, 0

            rep = 10 # Number of repetitions
            votes = []
            
            no_improve = 0
            # intermediate source images
            source_beam = [source_image0.clone().to(device)] * beam 
            # points1_beam = [points1_0] * beam
            # print('points1_beam', points1_beam[0].shape)
            
            # Active beams will track which beams are still active
            # Defined as a dictionary with the beam index as the key and the number of votes as the value
            active_beams_index = list(i for i in range(beam))
            active_beams = [0]*beam
            if verbose:
                print(f"Active beams: {active_beams_index}, {active_beams}")

            for j in range(rep):
                if j == 0:
                    beam_info = []
                else:
                    # make b copies of beam_info
                    beam_info = []
                    for active_beam in active_beams:
                        for k in range(len(models)):
                            # append values of active beams
                            beam_info.append(active_beam.copy())

                if verbose:
                    print(f"\nRep {j}: beam_info {beam_info}")

                # metrics_points_forward = []
                metrics_images_forward = []

                for b in range(beam):
                    if j == 0 and b != 0:
                        if verbose:
                            print(f"Rep {j}, Beam {b}")
                            print('Pass')
                        pass
                    else:
                        if verbose:
                            print(f"Rep {j}, Beam {b}: doing registration")
                        
                        # source_image = source_beam[b].clone()
                        # points1 = points1_beam[b].clone()

                        for k in range(len(models)):
                            results = reg(model[k], source_beam[b].clone(), 
                                          target_image, 
                                          i, j, b, k, output_dir)
                            
                            mse12_image_before = results[0]
                            mse12_image = results[1]
                            ssim12_image_before = results[2]
                            ssim12_image = results[3]

                            # beam identifiers
                            # print(beam_info, b, b*len(models)+k, len(beam_info))
                            if j == 0:
                                # pass
                                beam_info.append([k])
                            else:
                                beam_info[b*len(models)+k].append(k)
                                # print(beam_info[b*len(models)+k])
                                # beam_info[b*len(models)+k] = [b, k, mse12, tre12, mse12_image, ssim12_image]

                            # metrics_points_forward.append(tre12)
                            metrics_images_forward.append(mse12_image)
                            if verbose:
                                print(f"Pair {i}, Rep {j}, Beam {b}, Model {k}: {mse12_image}")

                            if j == 0 and k == 0:
                                mse12_image_before_first, \
                                    ssim12_image_before_first = mse12_image_before, ssim12_image_before
                                # print(mse_before_first, tre_before_first, mse12_image_before_first, ssim12_image_before_first)
                    if verbose:
                        print(beam_info, '\n')
                    # join the metrics of the forward and reverse directions
                    # metrics_points = np.array(metrics_points_forward)
                    metrics_images = np.array(metrics_images_forward)
                    
                if verbose:
                    print(f"Pair {i}, Rep {j}: {metrics_images}")
                
                # choose the best 'beam' models
                # best_index_points = np.argsort(metrics_points)[:beam]
                best_index_images = np.argsort(metrics_images)[:beam]
                # min_metrics_points = np.min([metrics_points])
                min_mse_images = np.min([metrics_images])

                if verbose:
                    print(f"Pair {i}, Rep {j}, best model (images): {best_index_images}")
                
                # if any element in tre_list is nan, use the model with the lowest mse
                # ++++++++++++++++ this part must be changed to be cases later ++++++++++++++++
                best_index = best_index_images

                if verbose:
                    print(f"Pair {i}, Rep {j}, best model index: {best_index} (image only)")

                for b in range(beam):
                    active_beams_index[b] = best_index[b]
                    active_beams[b] = beam_info[best_index[b]]
                    if verbose:
                        print(f"Processing active beam {b}: {active_beams[b]}")

                    # loop through the active beam path to collect all transformations and combine into one
                    M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
                    M = torch.from_numpy(M).unsqueeze(0)#.to(device)
                    src = source_image0.clone().to(device)

                    for k in range(len(active_beams[b])):
                        model_number = active_beams[b][k]
                        outputs = model[model_number](src, target_image)
                        affine_params_predicted = outputs[1]
                        M = combine_matrices(M, affine_params_predicted).to(device)
                        src = tensor_affine_transform0(source_image0, M)

                    transformed_source_affine = tensor_affine_transform0(source_image0, M)

                    if model_params.plot == 1 and i < 50:
                        plot_ = True
                    else:
                        plot_ = False
                    results = DL_affine_plot_image(f"test_{i}", output_dir,
                        i+1, f"{b}_{active_beams[b]}",
                        source_beam[b][0, 0, :, :].cpu().numpy(),
                        target_image[0, 0, :, :].cpu().numpy(),
                        transformed_source_affine[0, 0, :, :].cpu().numpy(),
                        None, None, None,
                        None, None,
                        affine_params_true=affine_params_true,
                        affine_params_predict=affine_params_predicted,
                        heatmap1=None, heatmap2=None, plot=plot_)

                    # update the source image and points
                    source_beam[b] = transformed_source_affine.clone()
                    # points1_beam[b] = points1_2_predicted.copy()

                    if b == 0:
                        # get the best results
                        mse12_image_before = results[0]
                        mse12_image = results[1]
                        ssim12_image_before = results[2]
                        ssim12_image = results[3]
                
                if verbose:
                    for b in range(beam):
                        print(f"Active beam {b}: {active_beams[b]}")
                        print(f"Active beam index {b}: {active_beams_index[b]}\n")

                # apply the best model to this pair
                # if mse12 < mse_before or mse12_image < mse12_image_before:
                if mse12_image < mse_images_best:
                    # mse_points_best = mse12
                    mse_images_best = mse12_image
                    # votes[j] = best_index

                    # update M
                    # print(affine_params_predicted.shape)
                    # M = combine_matrices(M, affine_params_predicted).to(device)
                    # # print(M)
                    # points1 = points1_2_predicted.clone()
                    # source_image = tensor_affine_transform0(data[0].to(device), M)

                    if no_improve > 0: 
                        no_improve -= 1

                else:
                    if verbose:
                        print(f"No improvement for {no_improve} reps")
                    no_improve += 1
                    # votes[j] = -1

                if verbose:
                    print(f"Pair {i}, Rep {j}: search path {active_beams}")

                if no_improve > 2:
                    # do final things
                    if verbose:
                        print(f"Pair {i}, Rep {j}: DONE (no improve) final search path {active_beams}")
                    active_beams = active_beams[0][0:-2]
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
            src = source_image0.clone().to(device)
            # points1 = points1_0.clone().to(device)

            if verbose:
                print(f"Finalizing pair {i}: {active_beams}")

            for k in range(len(active_beams)):
                model_number = active_beams[k]
                outputs = model[model_number](src, target_image)
                affine_params_predicted = outputs[1]
                M = combine_matrices(M, affine_params_predicted).to(device)
                src = tensor_affine_transform0(source_image0, M).clone()
                # points1 = transform_points_DVF(points1_0.cpu(), M.cpu(), source_image0)
                
            # get the final results
            transformed_source_affine = tensor_affine_transform0(source_image0, M)
            # points1_2_predicted = transform_points(points1_0[0].cpu().detach().numpy().T, M.cpu().detach().numpy()).T

            plot_ = 1
            best_model_text = f"final_rep{j:02d}_{active_beams}"
            results = DL_affine_plot_image(f"test_{i}", output_dir,
                i+1, best_model_text, source_image0[0, 0, :, :].cpu().numpy(),
                target_image[0, 0, :, :].cpu().numpy(),
                transformed_source_affine[0, 0, :, :].cpu().numpy(),
                None, None, None,
                None, None,
                affine_params_true=affine_params_true,
                affine_params_predict=M,
                heatmap1=None, heatmap2=None, plot=plot_)

            # print(f'\nEnd register pair {i}')
            # print(f'Votes: {votes}\n')
            # break

            mse12_image_before = results[0]
            mse12_image = results[1]
            ssim12_image_before = results[2]
            ssim12_image = results[3]

            # append metrics to metrics list
            new_entry = [i, mse12_image_before_first, mse12_image, \
                            ssim12_image_before_first, ssim12_image, [], active_beams]
            metrics.append(new_entry)
            # print(f"Pair {i}: {new_entry}")
            # print('\n')
            # break

    with open(csv_file, 'w', newline='') as file:

        writer = csv.writer(file)
        writer.writerow(["index", "mse12_image_before", "mse12_image", 
                         "ssim12_image_before", "ssim12_image", "num_points", "votes"])
        for i in range(len(metrics)):
            writer.writerow(metrics[i])
        
        # drop the last column of the array 'metrics'
        metrics = [metrics[i][0:-2] for i in range(len(metrics))]
        # print(metrics)
        metrics = np.array(metrics)

        # metrics = metrics[:, :8]
        nan_mask = np.isnan(metrics).any(axis=1)
        metrics = metrics[~nan_mask]

        avg = ["average", np.mean(metrics[:, 1]), np.mean(metrics[:, 2]), np.mean(metrics[:, 3]), np.mean(metrics[:, 4]), 
            ]
        std = ["std", np.std(metrics[:, 1]), np.std(metrics[:, 2]), np.std(metrics[:, 3]), np.std(metrics[:, 4]),
            ]
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
    parser.add_argument('--verbose', type=int, default=0, help='verbose output')
    parser.add_argument('--beam', type=int, default=1, help='beam search width')
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
                               plot=args.plot)
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
