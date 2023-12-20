from utils.utils0 import *
from utils.utils1 import *
from utils.utils1 import ModelParams, model_loader, print_summary#, test_repeat
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

model_params = ModelParams(dataset=2, sup=1)
train_dataset = datagen(model_params.dataset, True, model_params.sup)
test_dataset = datagen(model_params.dataset, False, model_params.sup)

class AffineTransformationNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(AffineTransformationNetwork, self).__init__()

        # Check if hidden_sizes is empty
        if not hidden_sizes:
            self.hidden_layers = nn.ModuleList([])
            self.fc = nn.Linear(input_size, output_size)
        else:
            layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(), nn.Dropout(0.1)]
            for i in range(1, len(hidden_sizes)):
                layers += [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]), nn.ReLU(), nn.Dropout(0.1)]
            layers += [nn.Linear(hidden_sizes[-1], output_size)]
            self.hidden_layers = nn.ModuleList(layers)

    def forward(self, affine_matrix, point):
        # Flatten and concatenate the input matrices
        input_data = torch.cat((affine_matrix, point))

        # Pass the input through hidden layers
        for layer in self.hidden_layers:
            input_data = layer(input_data)

        # Reshape the output to (2x1)
        transformed_point = input_data#.view(2)

        return transformed_point

# Example: Create an instance of the neural network with 2 hidden layers and 64 neurons in each hidden layer
input_size = 8  # Affine matrix (2x3) + Point (2)
num_hidden_layers = 2
num_perceptrons = 64
hidden_sizes = num_hidden_layers*[num_perceptrons]
output_size = 2  # Transformed point (2)

model = AffineTransformationNetwork(input_size, hidden_sizes, output_size)
print(model)
model.to(device)
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
parameters = list(model.parameters())
print(f'Number of parameters: {len(parameters)}')
# print(f'Parameters: {parameters}')
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1400, gamma=0.01)

train_dataset_new = []
# loop through all items in the train_dataset 
# and modify the input and output as described above
train_bar = tqdm(train_dataset, desc='Training')
# Loop over all training examples
for i, (source_img, target_img, affine_params, \
        matches1, matches2, matches1_2) in enumerate(train_bar):
    for j in range(matches1.shape[1]):
        train_dataset_new.append((torch.cat((affine_params.view(-1), matches1[0][j])), matches2[0][j]))
        if j > 3:
            break   

test_dataset_new = []
# loop through all items in the train_dataset 
# and modify the input and output as described above
test_bar = tqdm(test_dataset, desc='Testing')
# Loop over all training examples
for i, (source_img, target_img, affine_params, \
        matches1, matches2, matches1_2) in enumerate(test_bar):
    for j in range(matches1.shape[1]):
        test_dataset_new.append((torch.cat((affine_params.view(-1), matches1[0][j])), matches2[0][j]))
        if j > 1:
            break   

# Create empty list to store epoch number, train loss and validation loss
epoch_loss_list = []
running_loss_list = []
val_loss_list = []
print('Seed:', torch.seed())

# Training loop
epochs = 2000
for epoch in range(epochs):
    model.train()
    
    # optimizer.zero_grad()
    running_loss = 0.0
    
    train_bar = tqdm(train_dataset_new, desc='Training Epoch %d/%d' % (epoch+1, epochs))
    # Loop over all training examples
    for i, (input, matches1_2) in enumerate(train_bar):
        optimizer.zero_grad()
        # Set the inputs and labels to the training device
        # source_img = source_img.to(device)
        # target_img = target_img.to(device)
        affine_params = input[:6].to(device)
        matches1 = input[6:].to(device)
        # matches2 = matches2.to(device)
        matches1_2 = matches1_2.to(device)

        # optimizer.zero_grad()
        # for j in range(matches1.shape[1]):
        # Forward pass
        predicted_points = model(affine_params, matches1)

        # Compute the loss
        loss = criterion(predicted_points, matches1_2)

        running_loss += loss.item()
        
        train_bar.set_postfix({'loss': running_loss / (i+1)})

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    scheduler.step()
    epoch_loss_list.append(running_loss / len(train_dataset_new))
    # if epoch % 10 == 0:
    #     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_dataset_new)}')

    # Validation loop
    model.eval()

    val_running_loss = 0.0

    test_bar = tqdm(test_dataset_new, desc='Testing Epoch %d/%d' % (epoch+1, epochs))
    # Loop over all validation examples
    for i, (input, matches1_2) in enumerate(test_bar):
        # Set the inputs and labels to the validation device
        # source_img = source_img.to(device)
        # target_img = target_img.to(device)
        affine_params = input[:6].to(device)
        matches1 = input[6:].to(device)
        # matches2 = matches2.to(device)
        matches1_2 = matches1_2.to(device)

        # Forward pass
        predicted_points = model(affine_params, matches1)

        # Compute the loss
        loss = criterion(predicted_points, matches1_2)

        val_running_loss += loss.item()
        test_bar.set_postfix({'loss': val_running_loss / (i+1)})

    val_loss_list.append(val_running_loss / len(test_dataset_new))
    # if epoch % 10 == 0:
    #     print(f'Epoch [{epoch + 1}/{epochs}], Val Loss: {val_running_loss / len(test_dataset_new)}')

    # Save the model if the validation loss is the lowest we've seen so far
    # if val_loss_list[-1] == min(val_loss_list):
    #     torch.save(model.state_dict(), f'trained_models/affine_NN_{epoch:03d}.pth')
    #     print('Model saved')

# save model to file train_model/affine_transformation_network.pt
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
torch.save(model.state_dict(), f'trained_models/affine_NN_{num_hidden_layers}_{num_perceptrons}_{epochs:04d}_{timestamp}.pth')

 # Plot train loss and validation loss against epoch number
fig = plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), epoch_loss_list, label='Running Train Loss', linewidth=3)
plt.plot(range(1, epochs+1), val_loss_list, label='Running Validation Loss', linewidth=3)
plt.title('Train Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
# plt.yscale('log')
plt.tight_layout()

# Save plot
# signaturebar_gray(fig, f'{model_params.get_model_code()} - epoch{model_params.num_epochs} - {timestamp}')
fig.savefig(f'output/affine_NN_{num_hidden_layers}_{num_perceptrons}_{epochs:04d}_{timestamp}.png', dpi=300)

model.eval()

test_loss_list = []
test_bar = tqdm(test_dataset_new, desc='Testing')
for i, (input, matches1_2) in enumerate(test_bar):
    # Set the inputs and labels to the training device
    # source_img = source_img.to(device)
    # target_img = target_img.to(device)
    affine_params = input[:6].to(device)
    matches1 = input[6:].to(device)
    # matches2 = matches2.to(device)
    matches1_2 = matches1_2.to(device)

    # optimizer.zero_grad()
    # for j in range(matches1.shape[1]):
    # Forward pass
    predicted_points = model(affine_params, matches1)

    # Compute the loss
    loss = criterion(predicted_points, matches1_2)

    test_loss_list.append(loss.item())

    test_bar.set_postfix({'loss': loss.item() / (i+1)})
    print(f'True: {matches1_2.detach().cpu().numpy()}, Predicted: {predicted_points.detach().cpu().numpy()}')
    if i > 10:
        break

# mean and std of the test loss
mean_test_loss = np.mean(test_loss_list)
std_test_loss = np.std(test_loss_list)
print(f'Mean test loss: {mean_test_loss}, Std test loss: {std_test_loss}')
