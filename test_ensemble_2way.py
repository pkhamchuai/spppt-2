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

# from utils.SuperPoint import SuperPointFrontend
# from utils.utils1 import transform_points_DVF
def test(model_name, models, model_params, timestamp):
    # model_name: name of the model
    # model: model to be tested
    # model_params: model parameters
    # timestamp: timestamp of the model
    print('Test function input:', model_name, models, model_params, timestamp)

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
    output_dir = f"output/{model_name}_{model_params.get_model_code()}_{timestamp}_ensemble_test"
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

            # TODO: how to repeat the test?
            # 1. until the affine parameters are not change anymore
            # 2. until the mse is not change anymore
            # 3. until the mse is not change anymore and the affine parameters are not change anymore

            # use for loop with a large number of iterations 
            # check TRE of points1 and points2
            # if TRE grows larger than the last iteration, stop the loop
            TRE_last = np.inf
            MSE_last = np.inf
            mse12 = 0
            tre12 = 0

            mse_before_first, tre_before_first, mse12_image_before_first, \
                ssim12_image_before_first = 0, 0, 0, 0
            mse_before, tre_before, mse12_image, ssim12_image = 0, 0, 0, 0

            rep = 10
            votes = [np.inf] * rep  # Initialize a list to store the votes for each model
            mse_list = [np.inf] * 5
            tre_list = [np.inf] * 5
            no_improve = 0

            for j in range(rep):
                for k in range(len(models)):
                    # Forward + backward + optimize
                    outputs = model[k](source_image, target_image, points1)
                    transformed_source_affine = outputs[0]
                    affine_params_predicted = outputs[1]
                    points1_2_predicted = outputs[2]

                    # if i is an odd number
                    if i % 2 == 1 and i < 10 and model_params.plot == 0:
                        plot_ = True
                    else:
                        plot_ = False

                    results = DL_affine_plot(f"test_{i}", output_dir,
                        f"{i+1}", f"fw_rep{j:02d}_{k}", source_image[0, 0, :, :].cpu().numpy(), 
                        target_image[0, 0, :, :].cpu().numpy(), 
                        transformed_source_affine[0, 0, :, :].cpu().numpy(),
                        points1[0].cpu().detach().numpy().T, 
                        points2[0].cpu().detach().numpy().T, 
                        points1_2_predicted[0].cpu().detach().numpy().T, None, None, 
                        affine_params_true=affine_params_true,
                        affine_params_predict=affine_params_predicted, 
                        heatmap1=None, heatmap2=None, plot=plot_)

                    mse_before = results[1]
                    tre_before = results[3]
                    mse12_image_before = results[5]
                    ssim12_image_before = results[7]

                    mse12 = results[2]
                    tre12 = results[4]
                    mse12_image = results[6]
                    ssim12_image = results[8]

                    mse_list[k] = mse12
                    tre_list[k] = tre12

                    if j == 0 and k == 0:
                        mse_before_first, tre_before_first, mse12_image_before_first, \
                            ssim12_image_before_first = mse_before, tre_before, mse12_image_before, ssim12_image_before
                        # print(mse_before_first, tre_before_first, mse12_image_before_first, ssim12_image_before_first)
                
                # print(f"Pair {i}, Rep {j}: {mse_list}, {tre_list}")
                # the lowset mse12 and tre12 and its index
                mse12, tre12 = np.min(mse_list), np.min(tre_list)
                best_mse = np.argmin([mse_list])  # Find the index of the model with the best results
                best_tre = np.argmin([tre_list])  # Find the index of the model with the best results

                # print(f"Pair {i}, Rep {j}: {mse12}, {tre12}, best model: {best_mse}, {best_tre}")
                
                # if any element in tre_list is nan, use the model with the lowest mse
                if np.isnan(tre12):
                    votes[j] = best_tre
                else:
                    votes[j] = best_mse
                    best_tre = best_mse
                    tre12 = mse12
                    TRE_last = MSE_last

                # print(f"Pair {i}, Rep {j}: {mse12}, {tre12}, best model: {best_tre} {best_mse}")

                outputs = model[best_tre](source_image, target_image, points1)
                transformed_source_affine = outputs[0]
                affine_params_predicted = outputs[1]
                points1_2_predicted = outputs[2]
                
                if model_params.plot == 1 and i < 50:
                    plot_ = True
                else:
                    plot_ = False

                results = DL_affine_plot(f"test_{i}", output_dir,
                    f"{i+1}", f"rep{j:02d}_{best_tre}", source_image[0, 0, :, :].cpu().numpy(),
                    target_image[0, 0, :, :].cpu().numpy(),
                    transformed_source_affine[0, 0, :, :].cpu().numpy(),
                    points1[0].cpu().detach().numpy().T,
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

                points1 = points1_2_predicted.clone()
                source_image = transformed_source_affine.clone()

                # apply the best model to this pair
                # if tre12 < TRE_last and mse12 < MSE_last:
                if mse12 < MSE_last:
                    TRE_last = tre12
                    MSE_last = mse12
                    no_improve -= 1
                
                else:
                    tre12 = TRE_last
                    mse12 = MSE_last
                    no_improve += 1

                # if there is no improvement for 2 reps, stop the iteration
                if no_improve > 2:
                    break

            # print(f'\nEnd register pair {i}')
            # print(f'Votes: {votes}\n')
            # break

            # append metrics to metrics list
            new_entry = [i, mse_before_first, mse12, tre_before_first, tre12, mse12_image_before_first, mse12_image, \
                            ssim12_image_before_first, ssim12_image, np.max(points1_2_predicted.shape), votes]
            metrics.append(new_entry)
            # print(f"Pair {i}: {new_entry}")
            # break

            # target -> source +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            TRE_last = np.inf
            MSE_last = np.inf
            mse12 = 0
            tre12 = 0

            mse_before_first, tre_before_first, mse12_image_before_first, \
                ssim12_image_before_first = 0, 0, 0, 0
            mse_before, tre_before, mse12_image, ssim12_image = 0, 0, 0, 0

            rep = 10
            votes = [np.inf] * rep  # Initialize a list to store the votes for each model
            mse_list = [np.inf] * 5
            tre_list = [np.inf] * 5
            no_improve = 0

            for j in range(rep):
                for k in range(len(models)):
                    # Forward + backward + optimize
                    outputs = model[k](target_image, source_image, points2)
                    transformed_source_affine = outputs[0]
                    affine_params_predicted = outputs[1]
                    points1_2_predicted = outputs[2]

                    # if i is an odd number
                    if i % 2 == 1 and i < 10 and model_params.plot == 0:
                        plot_ = True
                    else:
                        plot_ = False

                    results = DL_affine_plot(f"test_{i}", output_dir,
                        f"{i+1}", f"rv_rep{j:02d}_{k}", target_image[0, 0, :, :].cpu().numpy(), 
                        source_image[0, 0, :, :].cpu().numpy(), 
                        transformed_source_affine[0, 0, :, :].cpu().numpy(),
                        points2[0].cpu().detach().numpy().T, 
                        points1[0].cpu().detach().numpy().T, 
                        points1_2_predicted[0].cpu().detach().numpy().T, None, None, 
                        affine_params_true=affine_params_true,
                        affine_params_predict=affine_params_predicted, 
                        heatmap1=None, heatmap2=None, plot=plot_)

                    mse_before = results[1]
                    tre_before = results[3]
                    mse12_image_before = results[5]
                    ssim12_image_before = results[7]

                    mse12 = results[2]
                    tre12 = results[4]
                    mse12_image = results[6]
                    ssim12_image = results[8]

                    mse_list[k] = mse12
                    tre_list[k] = tre12

                    if j == 0 and k == 0:
                        mse_before_first, tre_before_first, mse12_image_before_first, \
                            ssim12_image_before_first = mse_before, tre_before, mse12_image_before, ssim12_image_before
                        # print(mse_before_first, tre_before_first, mse12_image_before_first, ssim12_image_before_first)
                
                # print(f"Pair {i}, Rep {j}: {mse_list}, {tre_list}")
                # the lowset mse12 and tre12 and its index
                mse12, tre12 = np.min(mse_list), np.min(tre_list)
                best_mse = np.argmin([mse_list])  # Find the index of the model with the best results
                best_tre = np.argmin([tre_list])  # Find the index of the model with the best results

                # print(f"Pair {i}, Rep {j}: {mse12}, {tre12}, best model: {best_mse}, {best_tre}")
                
                # if any element in tre_list is nan, use the model with the lowest mse
                if np.isnan(tre12):
                    votes[j] = best_tre
                else:
                    votes[j] = best_mse
                    best_tre = best_mse
                    tre12 = mse12
                    TRE_last = MSE_last

                # print(f"Pair {i}, Rep {j}: {mse12}, {tre12}, best model: {best_tre} {best_mse}")

                outputs = model[best_tre](target_image, source_image, points2)
                transformed_source_affine = outputs[0]
                affine_params_predicted = outputs[1]
                points1_2_predicted = outputs[2]
                
                if model_params.plot == 1 and i < 50:
                    plot_ = True
                else:
                    plot_ = False

                results = DL_affine_plot(f"test_{i}", output_dir,
                    f"{i+1}", f"rv_rep{j:02d}_{best_tre}", target_image[0, 0, :, :].cpu().numpy(),
                    source_image[0, 0, :, :].cpu().numpy(),
                    transformed_source_affine[0, 0, :, :].cpu().numpy(),
                    points2[0].cpu().detach().numpy().T,
                    points1[0].cpu().detach().numpy().T,
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

                points1 = points1_2_predicted.clone()
                source_image = transformed_source_affine.clone()

                # apply the best model to this pair
                # if tre12 < TRE_last and mse12 < MSE_last:
                if mse12 < MSE_last:
                    TRE_last = tre12
                    MSE_last = mse12
                    no_improve -= 1
                
                else:
                    tre12 = TRE_last
                    mse12 = MSE_last
                    no_improve += 1

                # if there is no improvement for 2 reps, stop the iteration
                if no_improve > 2:
                    break

            # print(f'\nEnd register pair {i}')
            # print(f'Votes: {votes}\n')
            # break

            # append metrics to metrics list
            new_entry = [i, mse_before_first, mse12, tre_before_first, tre12, mse12_image_before_first, mse12_image, \
                            ssim12_image_before_first, ssim12_image, np.max(points1_2_predicted.shape), votes]
            metrics.append(new_entry)
            # print(f"Pair {i}: {new_entry}")
            # break
        

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
