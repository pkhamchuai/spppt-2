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
from train_points_rigid import train
# from test_points import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

# Stub to warn about opencv version.
if int(cv2.__version__[0]) < 3: # pragma: no cover
  print('Warning: OpenCV 3 is not installed')

image_size = 256


# class for TRE loss
class TRE_loss(nn.Module):
    def __init__(self):
        super(TRE_loss, self).__init__()

    def __call__(self, points1, points2):
        return torch.mean(torch.sqrt(torch.sum((points1 - points2)**2, dim=0)))


# from utils.SuperPoint import SuperPointFrontend
# from utils.utils1 import transform_points_DVF
# Define training function
def train(model_name, model_path, model_params, timestamp):
    
    # if model_params.sup:
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
            model = model_loader(model_name, model_params)
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            model.load_state_dict(torch.load(model_path))
            print(f'Loaded model from {model_path}')
            # print(model_path.split('/')[-1].split('_'))
            model_params.start_epoch = int(model_path.split('/')[-1].split('_')[5])
        else:
            print('Input a valid model')
            sys.exit()
        
        print(f'Loaded model from {model_path}\nStarting at epoch {model_params.start_epoch}')
        if model_params.start_epoch >= model_params.num_epochs:
            model_params.num_epochs += model_params.start_epoch
    else:
        model_params.start_epoch = 0
        print('No model loaded, starting from scratch')
    
    datasets = [1, 2, 3, 0]
    epochs = [0, 5, 10, 15, 20]
    sups = [1, 1, 1, 0]

    
    for idx in range(4):
        print(f'\nStage: {idx+1}')

        # Create empty list to store epoch number, train loss and validation loss
        epoch_loss_list = []
        running_loss_list = []

        model_params.dataset = datasets[idx]
        model_params.sup = sups[idx]
        model_params.num_epochs = epochs[idx+1]
        model_params.start_epoch = epochs[idx]

        train_dataset = datagen(datasets[idx], True, sups[idx])
        test_dataset = datagen(datasets[idx], False, sups[idx])

        # Get sample batch
        # print('Train set: ', [x.shape for x in next(iter(train_dataset))])
        # print('Test set: ', [x.shape for x in next(iter(test_dataset))])

        # Define loss function based on supervised or unsupervised learning
        criterion = model_params.loss_image
        # extra = loss_extra()
        # criterion_points = nn.MSELoss() # 
        # criterion_points = loss_points()
        criterion_points = TRE_loss()

        # print case
        # print(model_params)
        model_params.print_explanation()
        
        # print the fixed seed
        # print(f'Fixed seed: {torch.initial_seed()}')
        
        # Create output directory
        output_dir = f"output/{timestamp}_stage{idx+1}_{model_name}_{model_params.get_model_code()}"
        os.makedirs(output_dir, exist_ok=True)
        save_plot_name = f"{output_dir}/loss_{model_params.get_model_code()}_epoch{model_params.num_epochs}_{timestamp}.png"

        # Plot train loss and validation loss against epoch number
        fig = plt.figure(figsize=(12, 8))

        # Train model
        for epoch in range(epochs[idx], epochs[idx+1]):
            # Set model to training mode
            model.train()
            
            # optimizer.zero_grad()
            running_loss = 0.0
            train_bar = tqdm(train_dataset, desc=f'Training Epoch {epoch+1}/{model_params.num_epochs}')
            for i, data in enumerate(train_bar):
                # Zero the parameter gradients
                # optimizer.zero_grad()
                
                loss = 0.0

                # Get images and affine parameters
                source_image, target_image, affine_params_true, \
                    points1, points2, points1_2_true = data
                # print(source_image.shape, target_image.shape, affine_params_true.shape,
                #     points1.shape, points2.shape, points1_2_true.shape)

                source_image = source_image.requires_grad_(True).to(device)
                target_image = target_image.requires_grad_(True).to(device)
                # add gradient to the matches
                points1.requires_grad_(True).to(device)
                points2.requires_grad_(True).to(device)
                points1_2_true.requires_grad_(True).to(device)

                # Forward + backward + optimize
                outputs = model(source_image, target_image, points1)
                # for j in range(len(outputs)):
                #         print(j, outputs[j].shape)
                # 0 torch.Size([1, 1, 256, 256])
                # 1 torch.Size([1, 2, 3])
                # 2 (2, 0)
                # 3 (2, 0)
                # 4 (256, 100)
                # 5 (256, 115)
                # 6 (256, 256)
                # 7 (256, 256)
                affine_params_predicted = outputs[1] # affine parameters
                # print(affine_params_predicted.shape)
                transformed_source_affine = outputs[0] # image
                # translate_params_predicted = outputs[1] # translation parameters
                affine_params_predicted.requires_grad_(True).to(device)
                points1_2_predicted = outputs[2].T
                points1_2_predicted.requires_grad_(True).to(device)

                try:
                    points1_2_predicted = points1_2_predicted.reshape(
                        points1_2_predicted.shape[2], points1_2_predicted.shape[1])
                except:
                    pass

                if model_params.image:
                    loss += criterion(transformed_source_affine, target_image)
                    # loss += extra(affine_params_predicted)
                    
                if model_params.sup:
                    loss_affine = criterion_affine(affine_params_true.view(1, 2, 3), affine_params_predicted.cpu())
                    # TODO: add loss for points1_affine and points2, Euclidean distance
                    # loss_points = criterion_points(points1_affine, points2)
                    loss += loss_affine

                if model_params.points:
                    loss_ += criterion_points(torch.flatten(points1_2_predicted, start_dim=1), 
                                        torch.flatten(points1_2_true[0], start_dim=1).to(device))
                    # loss_ = torch.subtract(points1_2_predicted.cpu().detach(), points1_2_true[0].cpu().detach())
                    loss += torch.sum(torch.square(loss_))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                    
                # print shape of points1_2_predicted, points2, points1
                # print(points1_2_predicted.shape, points2.shape, points1.shape)
                # Plot images if i < 5
                if i % 50 == 0:
                    if points1_2_predicted.shape[-1] != 2:
                        points1_2_predicted = points1_2_predicted.T
                    if points1.shape[-1] != 2:
                        points1 = points1.T
                    if points2.shape[-1] != 2:
                        points2 = points2.T
                    # print(points1_2_predicted.shape, points2.shape, points1.shape)

                    DL_affine_plot(f"epoch{epoch+1}_train", output_dir,
                        #         f"{i}", model_params.get_model_code(), 
                        #         source_image[0, 0, :, :].detach().cpu().numpy(), target_image[0, 0, :, :].detach().cpu().numpy(), 
                        #         transformed_source_affine[0, :, :].detach().cpu().numpy(),
                        #         points1[0].cpu().detach().numpy().T, 
                        #         points2[0].cpu().detach().numpy().T, 
                        #         points1_2_predicted.cpu().detach().numpy().T, None, None, affine_params_true=affine_params_true,
                        #             affine_params_predict=affine_params_predicted, heatmap1=None, heatmap2=None, plot=True)
                        # DL_affine_plot(f"epoch{epoch+1}_valid", output_dir,
                        f"{i}", model_params.get_model_code(), 
                        source_image[0, 0, :, :].detach().cpu().numpy(), 
                        target_image[0, 0, :, :].detach().cpu().numpy(), 
                        transformed_source_affine[0, 0, :, :].detach().cpu().numpy(),
                        points1[0].cpu().detach().numpy().T, 
                        points2[0].cpu().detach().numpy().T, 
                        points1_2_predicted.cpu().detach().numpy().T, None, None, 
                        affine_params_true=affine_params_true,
                        affine_params_predict=affine_params_predicted, 
                        heatmap1=None, heatmap2=None, plot=True)

                # Print statistics
                running_loss += loss.item()
                running_loss_list.append([epoch+((i+1)/len(train_dataset)), loss.item()])
                train_bar.set_postfix({'loss': running_loss / (i+1)})
            print(f'Training Epoch {epoch+1}/{model_params.num_epochs} loss: {running_loss / len(train_dataset)}')
            
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            
            # Validate model
            validation_loss = 0.0
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(test_dataset, 0):
                    loss = 0.0
                    # Get images and affine parameters
                    source_image, target_image, affine_params_true, \
                        points1, points2, points1_2_true = data

                    source_image = source_image.requires_grad_(True).to(device)
                    target_image = target_image.requires_grad_(True).to(device)
                    # add gradient to the matches
                    points1 = points1.requires_grad_(True).to(device)
                    points2 = points2.requires_grad_(True).to(device)
                    points1_2_true = points1_2_true.requires_grad_(True).to(device)

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
                    points1_2_predicted = outputs[2].T

                    try:
                        points1_2_predicted = points1_2_predicted.reshape(
                        points1_2_predicted.shape[2], points1_2_predicted.shape[1])
                    except:
                        pass

                    if model_params.image:
                        loss += criterion(transformed_source_affine, target_image)
                    # loss += extra(affine_params_predicted)

                    if model_params.sup:
                        loss_affine = criterion_affine(affine_params_true.view(1, 2, 3), affine_params_predicted.cpu())
                    #     # TODO: add loss for points1_affine and points2, Euclidean distance
                    #     # loss_points = criterion_points(points1_affine, points2)
                        loss += loss_affine

                    if model_params.points:
                        # print the input's device
                        loss += criterion_points(torch.flatten(points1_2_predicted, start_dim=1).cpu().detach(), 
                                        torch.flatten(points1_2_true[0], start_dim=1).cpu().detach())

                    # Add to validation loss
                    validation_loss += loss.item()

                    # Plot images if i < 5
                    if i % 25 == 0:
                        if points1_2_predicted.shape[-1] != 2:
                            points1_2_predicted = points1_2_predicted.T
                        if points1.shape[-1] != 2:
                            points1 = points1.T
                        if points2.shape[-1] != 2:
                            points2 = points2.T
                        DL_affine_plot(f"epoch{epoch+1}_valid", output_dir,
                            f"{i}", model_params.get_model_code(), 
                            source_image[0, 0, :, :].cpu().numpy(), 
                            target_image[0, 0, :, :].cpu().numpy(), 
                            transformed_source_affine[0, 0, :, :].cpu().numpy(),
                            points1[0].cpu().detach().numpy().T, 
                            points2[0].cpu().detach().numpy().T, 
                            points1_2_predicted.cpu().detach().numpy().T, None, None, 
                            affine_params_true=affine_params_true,
                            affine_params_predict=affine_params_predicted, 
                            heatmap1=None, heatmap2=None, plot=True)
                        
                    # Print validation statistics
                    validation_loss /= len(test_dataset)
                    print(f'Validation Epoch {epoch+1}/{model_params.num_epochs} loss: {validation_loss}')

                    # Append epoch number, train loss and validation loss to epoch_loss_list
                    epoch_loss_list.append([epoch, running_loss / len(train_dataset), validation_loss])

            print(f'\nFinished Training Stage: {idx+1}')

            # Extract epoch number, train loss and validation loss from epoch_loss_list
            epoch = [x[0] for x in epoch_loss_list]
            train_loss = [x[1] for x in epoch_loss_list]
            val_loss = [x[2] for x in epoch_loss_list]
            step = [x[0] for x in running_loss_list]
            running_train_loss = [x[1] for x in running_loss_list]

            # Plot train loss and validation loss against epoch number
            # fig = plt.figure(figsize=(12, 5))
            plt.subplot(2, 2, idx+1)
            plt.plot(step, running_train_loss, label='Running Train', alpha=0.3)
            plt.plot(epoch, train_loss, label='Train', linewidth=3)
            plt.plot(epoch, val_loss, label='Validation', linewidth=3)
            plt.title('Train and Validation Loss')
            plt.legend()
            plt.xlabel(f"(Stage {i + 1}) Epochs")
            if idx == 0 or idx == 2:
                plt.ylabel("Loss")
            plt.grid(True)
            plt.yscale('log')
            plt.tight_layout()

            # test using all datasets
            test(model_name, model, model_params, timestamp)

            # Save model
            model_save_path = "trained_models/"
            model_name_to_save = model_save_path + f"{timestamp}_{model_name}_{model_params.get_model_code()}_{idx}.pth"
            torch.save(model.state_dict(), model_name_to_save)
            print(f'Model saved in: {model_name_to_save}')

        # Save plot
        signaturebar_gray(fig, f'{model_params.get_model_code()} - epoch{model_params.num_epochs} - {timestamp}')
        fig.savefig(save_plot_name)
        # plt.show()
        plt.close(fig)

        # delete all txt files in output_dir
        for file in os.listdir(output_dir):
            if file.endswith(".txt"):
                os.remove(os.path.join(output_dir, file))

        # save the output of print_explanation() and loss_list to a txt file
        print_summary(model_name, model_name_to_save, 
                    model_params, epoch_loss_list, timestamp, False)

    # Return epoch_loss_list
    return model, epoch_loss_list

def test(model_name, model_, model_params, timestamp, stage=0):

    print('Test function input:', model_name, model_, model_params, timestamp)

    # if model is a string, load the model
    # if model is a loaded model, use the model
    # if isinstance(model_, str):
    #     model = model_loader(model_name, model_params)
    #     buffer = io.BytesIO()
    #     torch.save(model.state_dict(), buffer)
    #     buffer.seek(0)
    #     model.load_state_dict(torch.load(model_))
    #     print(f'Loaded model from {model_}')
    # elif isinstance(model_, nn.Module):
    #     print(f'Using model {model_name}')
    model = model_
    datasets = [1, 2, 3, 0]
    epochs = [0, 2, 4, 6, 8]
    # epochs = [0, 5, 10, 15, 20]
    sups = [1, 1, 1, 0]

    for idx in range(4):
        print(f'\nDataset: {idx+1}')

        model_params.dataset = datasets[idx]
        model_params.sup = sups[idx]
        model_params.num_epochs = epochs[idx+1]
        model_params.start_epoch = epochs[idx]

        # Set model to training mode
        model.eval()
        test_dataset = datagen(model_params.dataset, False, model_params.sup)

        # Create output directory
        output_dir = f"output/{timestamp}_stage{stage}_dataset{idx+1}_{model_name}_{model_params.get_model_code()}_test"
        # output_dir = f"output/{model_name}_{model_params.get_model_code()}_{timestamp}_test"
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
                source_image, target_image, affine_params_true, points1, points2, points1_2_true = data

                source_image = source_image.requires_grad_(True).to(device)
                target_image = target_image.requires_grad_(True).to(device)
                # add gradient to the matches
                points1 = points1.requires_grad_(True).to(device)
                points2 = points2.requires_grad_(True).to(device)

                for j in range(1):
                    # Forward + backward + optimize
                    outputs = model(source_image, target_image, points1)
                    # for i in range(len(outputs)):
                    #     print(i, outputs[i].shape)
                    transformed_source_affine = outputs[0]
                    affine_params_predicted = outputs[1]
                    points1_2_predicted = outputs[2]

                    try:
                        points1_2_predicted = points1_2_predicted.reshape(
                        points1_2_predicted.shape[2], points1_2_predicted.shape[1])
                    except:
                        pass

                    if i < 100:
                        plot_ = True
                    else:
                        plot_ = False

                    if points1_2_predicted.shape[-1] != 2:
                        points1_2_predicted = points1_2_predicted.T
                    if points1.shape[-1] != 2:
                        points1 = points1.T
                    if points2.shape[-1] != 2:
                        points2 = points2.T

    # Set model to training mode
    model.eval()
    test_dataset = datagen(model_params.dataset, False, model_params.sup)

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
            source_image, target_image, affine_params_true, points1, points2, points1_2_true = data

            source_image = source_image.requires_grad_(True).to(device)
            target_image = target_image.requires_grad_(True).to(device)
            # add gradient to the matches
            points1 = points1.requires_grad_(True).to(device)
            points2 = points2.requires_grad_(True).to(device)

            for j in range(1):
                # Forward + backward + optimize
                outputs = model(source_image, target_image, points1)
                # for i in range(len(outputs)):
                #     print(i, outputs[i].shape)
                transformed_source_affine = outputs[0]
                affine_params_predicted = outputs[1]
                points1_2_predicted = outputs[2]

                try:
                    points1_2_predicted = points1_2_predicted.reshape(
                    points1_2_predicted.shape[2], points1_2_predicted.shape[1])
                except:
                    pass

                if i < 100:
                    plot_ = True
                else:
                    plot_ = False

                if points1_2_predicted.shape[-1] != 2:
                    points1_2_predicted = points1_2_predicted.T
                if points1.shape[-1] != 2:
                    points1 = points1.T
                if points2.shape[-1] != 2:
                    points2 = points2.T
                # print(points1_2_predicted.shape, points2.shape, points1.shape)

                results = DL_affine_plot(f"test", output_dir,
                    f"{i}", f"{i+1}", source_image[0, 0, :, :].cpu().numpy(), 
                    target_image[0, 0, :, :].cpu().numpy(), 
                    transformed_source_affine[0, 0, :, :].cpu().numpy(),
                    points1[0].cpu().detach().numpy().T, 
                    points2[0].cpu().detach().numpy().T, 
                    points1_2_predicted.cpu().detach().numpy().T, None, None, 
                    affine_params_true=affine_params_true,
                    affine_params_predict=affine_params_predicted, 
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

                # append metrics to metrics list
                metrics.append([i, mse_before, mse12, tre_before, tre12, mse12_image_before, mse12_image, ssim12_image_before, ssim12_image, np.max(points1_2_predicted.shape)])

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

    # extra_text = f"Test model {model_name} at {model_} with dataset {model_params.dataset}. "
    print_summary(model_name, model_, model_params, 
                  None, timestamp, test=True)

if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser(description='Deep Learning for Image Registration')    
    parser.add_argument('--dataset', type=int, default=1, help='dataset number')
    parser.add_argument('--sup', type=int, default=1, help='supervised learning (1) or unsupervised learning (0)')
    parser.add_argument('--image', type=int, default=0, help='loss image used for training')
    parser.add_argument('--points', type=int, default=1, help='use loss points (1) or not (0)')
    parser.add_argument('--loss_image', type=int, default=0, help='loss function for image registration')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.96, help='decay rate')
    parser.add_argument('--model', type=str, default='DHR', help='which model to use')
    # parser.add_argument('--model', type=str, default='SP_Rigid', help='which model to use')
    parser.add_argument('--model_path', type=str, default=None, help='path to model to load')
    args = parser.parse_args()

    if args.model_path is None:
      model_path = None
    else:
      model_path = 'trained_models/' + args.model_path

    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_params = ModelParams(image=args.image, 
                              loss_image=args.loss_image, start_epoch=0,  
                              learning_rate=args.learning_rate, decay_rate=args.decay_rate)
    model_params.print_explanation()
    
    trained_model, loss_list = train(args.model, model_path, model_params, timestamp)
    
    # print("\nTesting the trained model +++++++++++++++++++++++")
    # test(args.model, trained_model, model_params, timestamp)
    # print("Test model finished +++++++++++++++++++++++++++++")
    
    # for i in range(1, 4):
    #   print(datasets[i], sups[i], epochs[i])
    #   model_params = ModelParams(dataset=datasets[i], sup=sups[i], image=args.image, heatmaps=args.heatmaps, 
    #                             loss_image=args.loss_image, start_epoch=epochs[i-1], num_epochs=epochs[i], 
    #                             learning_rate=args.learning_rate, decay_rate=args.decay_rate)
    #   model_params.print_explanation()
      
    #   trained_model, loss_list = train(args.model, trained_model, model_params, timestamp)

    # print("\nTesting the trained model +++++++++++++++++++++++")
    # test(args.model, trained_model, model_params, timestamp)
    # print("Test model finished +++++++++++++++++++++++++++++")
