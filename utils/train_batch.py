# import
import os
import sys
import io
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim


from utils.utils0 import *
from utils.utils1 import *

# class for TRE loss
class TRE_loss(nn.Module):
    def __init__(self):
        super(TRE_loss, self).__init__()

    def __call__(self, points1, points2):
        return torch.mean(torch.sqrt(torch.sum((points1 - points2)**2, dim=0)))

# Define training function
def train(model_name, model_path, model_params, timestamp):

    model_params.print_explanation()
    batch_size = model_params.batch_size
    train_dataset = datagen(model_params.dataset, True, model_params.sup, batch_size=batch_size)
    test_dataset = datagen(model_params.dataset, False, model_params.sup)

    # Get sample batch
    print('Train set: ', [x.shape for x in next(iter(train_dataset))])
    print('Test set: ', [x.shape for x in next(iter(test_dataset))])

    # Define loss function based on supervised or unsupervised learning
    criterion = model_params.loss_image
    # extra = loss_extra()
    # criterion_points = nn.MSELoss() # 
    # criterion_points = loss_points()
    criterion_points = TRE_loss()

    if model_params.sup:
        criterion_affine = nn.MSELoss()
        # TODO: add loss for points1_affine and points2, Euclidean distance

    model = model_loader(model_name, model_params)
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, model_params.learning_rate)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: model_params.decay_rate ** epoch)

    # if a model is loaded, the training will continue from the epoch it was saved at
    if model_path is not None:
        # if the model is a string, load the model
        # if the model is a loaded model, use the model
        if isinstance(model_path, nn.Module):
            print(f'Using model {model_name}')
            model = model_path
        elif isinstance(model_path, str):
            # if model_name == 'DHR':
            #     torch.manual_seed(9793047918980052389)
            model = model_loader(model_name, model_params)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            model.load_state_dict(torch.load(model_path))
            print(f'Loaded model from {model_path}')
            # print(model_path.split('/')[-1].split('_'))
            model_params.start_epoch = int(model_path.split('/')[-1].split('_')[4])
        else:
            print('Input a valid model')
            sys.exit()
        
        print(f'Loaded model from {model_path}\nStarting at epoch {model_params.start_epoch}')
        if model_params.start_epoch >= model_params.num_epochs:
            model_params.num_epochs += model_params.start_epoch
    else:
        model_params.start_epoch = 0
        print('No model loaded, starting from scratch')

    # print case
    # print(model_params)
    model_params.print_explanation()

    # Create empty list to store epoch number, train loss and validation loss
    epoch_loss_list = []
    running_loss_list = []
    
    print('Seed:', torch.seed())
 
    # Create output directory
    output_dir = f"output/{model_name}_{model_params.get_model_code()}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    save_plot_name = f"{output_dir}/loss_{model_params.get_model_code()}_epoch{model_params.num_epochs}_{timestamp}.png"

    # Train model
    for epoch in range(model_params.start_epoch, model_params.num_epochs):
        # Set model to training mode
        model.train()
        
        running_loss = 0.0
        loss = 0.0
        train_bar = tqdm(train_dataset, desc=f'Training Epoch {epoch+1}/{model_params.num_epochs}')
        for i, data in enumerate(train_bar):

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Get images and affine parameters
            source_image, target_image, affine_params_true, \
                points1, points2, points1_2_true = data
            source_image = source_image.to(device)
            target_image = target_image.to(device)

            # Forward + backward + optimize
            outputs = model(source_image, target_image, points1)
            # for i in range(len(outputs)):
            #         print(i, outputs[i].shape)
            # 0 torch.Size([1, 1, 256, 256])
            # 1 torch.Size([1, 2, 3])
            # 2 (2, 0)
            # 3 (2, 0)
            # 4 (256, 100)
            # 5 (256, 115)
            # 6 (256, 256)
            # 7 (256, 256)
            transformed_source_affine = outputs[0] # image
            affine_params_predicted = outputs[1] # affine parameters
            # points1 = np.array(outputs[2])
            # points2 = np.array(outputs[3])
            points1_2_predicted = outputs[2]

            # print(f"affine_params_true: {affine_params_true}")
            # print(f"affine_params_predicted: {affine_params_predicted}\n")

            # try:
            #     points1_2_predicted = points1_2_predicted.reshape(
            #         points1_2_predicted.shape[2], points1_2_predicted.shape[1])
            # except:
            #     pass

            if model_params.image:
                loss += criterion(transformed_source_affine, target_image)
                # loss += extra(affine_params_predicted)
                
            if model_params.sup:
                # print(f"affine_params_true: {affine_params_true.shape}")
                loss_affine = criterion_affine(affine_params_true, \
                                               affine_params_predicted.cpu())
                # TODO: add loss for points1_affine and points2, Euclidean distance
                # loss_points = criterion_points(points1_affine, points2)
                loss += loss_affine

            if model_params.points:
                if not isinstance(points1_2_predicted, torch.Tensor):
                        points1_2_predicted = torch.tensor(points1_2_predicted)
                if not isinstance(points1_2_true, torch.Tensor):
                        points1_2_true = torch.tensor(points1_2_true)
                # print(f"points1_2_predicted: {points1_2_predicted.shape}, points1_2_true: {points1_2_true.shape}")
                try:
                    loss += criterion_points(torch.flatten(points1_2_predicted, start_dim=1).cpu().detach(), 
                                    torch.flatten(points1_2_true, start_dim=1).cpu().detach())
                except:
                    pass
                
            # optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # if i % batch_size == 0:
                
            #     if points1_2_predicted.shape[-1] != 2:
            #         points1_2_predicted = points1_2_predicted.T
            #     if points1.shape[-1] != 2:
            #         points1 = points1.T
            #     if points2.shape[-1] != 2:
            #         points2 = points2.T

            #     for j in range(4):
            #         # if points1_2_predicted.shape[-1] != 2:
            #         #     points1_2_predicted = points1_2_predicted.T
            #         # if points1.shape[-1] != 2:
            #         #     points1 = points1.T
            #         # if points2.shape[-1] != 2:
            #         #     points2 = points2.T
            #         # if the points are not torch tensors, convert them to torch tensors
            #         if not isinstance(points1, torch.Tensor):
            #             points1 = torch.tensor(points1)
            #         if not isinstance(points2, torch.Tensor):
            #             points2 = torch.tensor(points2)
            #         if not isinstance(points1_2_predicted, torch.Tensor):
            #             points1_2_predicted = torch.tensor(points1_2_predicted)

            #         print(f"points1_2_predicted: {points1_2_predicted.shape}, points1_2_true: {points1_2_true.shape}")
                        
            #         DL_affine_plot(f"epoch{epoch+1}_train", output_dir, f"{j}", f"{j+1}", 
            #             source_image[j, 0, :, :].cpu().numpy(), 
            #             target_image[j, 0, :, :].cpu().numpy(), 
            #             transformed_source_affine[0, 0, :, :].cpu().detach().numpy(),
            #             points1[j].cpu().detach().numpy().T, 
            #             points2[j].cpu().detach().numpy().T, 
            #             points1_2_predicted[j].T, None, None, 
            #             affine_params_true=affine_params_true[j],
            #             affine_params_predict=affine_params_predicted[j], 
            #             heatmap1=None, heatmap2=None, plot=True)

            # Print statistics
            running_loss += loss.item()
            running_loss_list.append([epoch+((i+1)/len(train_dataset)), loss.item()])
            train_bar.set_postfix({'loss': running_loss / (i+1)})
            loss = 0.0

        print(f'Training Epoch {epoch+1}/{model_params.num_epochs} loss: {running_loss / len(train_dataset)}')
        
        # Validate model
        validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_dataset, 0):
                loss = 0.0
                
                # Get images and affine parameters
                source_image, target_image, affine_params_true, \
                    points1, points2, points1_2_true = data
                source_image = source_image.to(device)
                target_image = target_image.to(device)

                # Forward pass
                outputs = model(source_image, target_image, points1)
                # for i in range(len(outputs)):
                #     print(i, outputs[i].shape)
                transformed_source_affine = outputs[0]
                affine_params_predicted = outputs[1]
                # points1 = np.array(outputs[2])
                # points2 = np.array(outputs[3])
                points1_2_predicted = np.array(outputs[2])

                # try:
                #     points1_2_predicted = points1_2_predicted.reshape(
                #         points1_2_predicted.shape[2], points1_2_predicted.shape[1])
                # except:
                #     pass

                # desc1_2 = outputs[5]
                # desc2 = outputs[6]
                # heatmap1 = outputs[7]
                # heatmap2 = outputs[8]

                if model_params.image:
                    loss += criterion(transformed_source_affine, target_image)
                # loss += extra(affine_params_predicted)

                if model_params.sup:
                    loss_affine = criterion_affine(affine_params_true.view(1, 2, 3), affine_params_predicted.cpu())
                #     # TODO: add loss for points1_affine and points2, Euclidean distance
                #     # loss_points = criterion_points(points1_affine, points2)
                    loss += loss_affine

                if model_params.points:
                    # if the input is not a tensor, convert it to a tensor
                    if not isinstance(points1_2_predicted, torch.Tensor):
                        points1_2_predicted = torch.tensor(points1_2_predicted)
                    if not isinstance(points1_2_true, torch.Tensor):
                        points1_2_true = torch.tensor(points1_2_true)
                    # print(f"points1_2_predicted: {points1_2_predicted}, points1_2_true: {points1_2_true}")
                    try:
                        loss += criterion_points(torch.flatten(points1_2_predicted, start_dim=1).cpu().detach(), 
                                        torch.flatten(points1_2_true, start_dim=1).cpu().detach())
                    except:
                        pass

                # Add to validation loss
                validation_loss += loss.item()

                if i < 10:
                    # if points1_2_predicted.shape[-1] != 2:
                    #     points1_2_predicted = points1_2_predicted.T
                    # if points1.shape[-1] != 2:
                    #     points1 = points1.T
                    # if points2.shape[-1] != 2:
                    #     points2 = points2.T
                    # if the points are not torch tensors, convert them to torch tensors
                    if not isinstance(points1, torch.Tensor):
                        points1 = torch.tensor(points1)
                    if not isinstance(points2, torch.Tensor):
                        points2 = torch.tensor(points2)
                    if not isinstance(points1_2_predicted, torch.Tensor):
                        points1_2_predicted = torch.tensor(points1_2_predicted)

                    # print(f"points1_2_predicted: {points1_2_predicted.shape}, points1_2_true: {points1_2_true.shape}")
                    # print(f"points1: {points1.shape}, points2: {points2.shape}")
                    
                        
                    DL_affine_plot(f"epoch{epoch+1}_valid", output_dir, f"{i}", f"{i+1}", 
                        source_image[0, 0, :, :].cpu().numpy(), 
                        target_image[0, 0, :, :].cpu().numpy(), 
                        transformed_source_affine[0, 0, :, :].cpu().numpy(),
                        points1[0].cpu().detach().numpy().T, 
                        points2[0].cpu().detach().numpy().T, 
                        points1_2_predicted[0].numpy().T, None, None, 
                        affine_params_true=affine_params_true[0],
                        affine_params_predict=affine_params_predicted[0], 
                        heatmap1=None, heatmap2=None, plot=True)

        # Print validation statistics
        validation_loss /= len(test_dataset)
        print(f'Validation Epoch {epoch+1}/{model_params.num_epochs} loss: {validation_loss}')

        # Append epoch number, train loss and validation loss to epoch_loss_list
        epoch_loss_list.append([epoch+1, running_loss / len(train_dataset), validation_loss])

        # Extract epoch number, train loss and validation loss from epoch_loss_list
        epoch = [x[0] for x in epoch_loss_list]
        train_loss = [x[1] for x in epoch_loss_list]
        val_loss = [x[2] for x in epoch_loss_list]
        step = [x[0] for x in running_loss_list]
        running_train_loss = [x[1] for x in running_loss_list]

        # Plot train loss and validation loss against epoch number
        fig = plt.figure(figsize=(12, 5))
        plt.plot(step, running_train_loss, label='Running Train Loss', alpha=0.3)
        plt.plot(epoch, train_loss, label='Train Loss', linewidth=3)
        plt.plot(epoch, val_loss, label='Validation Loss', linewidth=3)
        plt.title('Train and Validation Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.tight_layout()
        signaturebar_gray(fig, f'{model_params.get_model_code()} - batch {model_params.batch_size} - {model_params.num_epochs} epochs - {timestamp}')
        plt.savefig(save_plot_name)
        plt.close(fig)

    print('\nFinished Training')

    # delete all txt files in output_dir
    for file in os.listdir(output_dir):
        if file.endswith(".txt"):
            os.remove(os.path.join(output_dir, file))

    # Save model
    model_save_path = "trained_models/"
    model_name_to_save = model_save_path + f"{model_name}_{model_params.get_model_code()}_{timestamp}.pth"
    torch.save(model.state_dict(), model_name_to_save)
    print(f'Model saved in: {model_name_to_save}')

    # save the output of print_explanation() and loss_list to a txt file
    print_summary(model_name, model_name_to_save, model_params, 
                  epoch_loss_list, timestamp, output_dir=output_dir, test=False)

    # Return epoch_loss_list
    return model, epoch_loss_list