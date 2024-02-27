import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom dataset class
class MyCustomDataset(Dataset):
    def __init__(self, image_paths, points_sets):
        self.image_paths = image_paths
        self.points_sets = points_sets

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        points = self.points_sets[index]
        # Load images and preprocess them (you should implement your own image loading logic)
        # create a random image for testing
        image1 = np.random.rand(256, 256, 3)
        image1 = torch.tensor(image1, dtype=torch.float32)
        image2 = np.random.rand(256, 256, 3)
        image2 = torch.tensor(image2, dtype=torch.float32)
        return image1, image2, points

# Custom collate function
def my_collate(batch):
    # Separate the elements of the batch
    images1, images2, points_sets_list = zip(*batch)
    
    # Pad points within each batch
    max_num_points = max(len(points) for points_set in points_sets_list for points in points_set)
    padded_points_sets_list = []
    for points_set in points_sets_list:
        padded_points_set = []
        for points in points_set:
            num_points_to_pad = max_num_points - len(points)
            padded_points = np.pad(points, ((0, num_points_to_pad), (0, 0)), mode='constant', constant_values=0)
            padded_points_set.append(padded_points)
        padded_points_sets_list.append(padded_points_set)

    # Convert lists to tensors
    images1 = torch.stack(images1)
    images2 = torch.stack(images2)
    points_sets_list = [torch.tensor(padded_points_sets, dtype=torch.float32) for padded_points_sets in padded_points_sets_list]

    return images1, images2, points_sets_list

# Example data
image_paths = ['image1.jpg', 'image2.jpg']
points_sets_list = [
    [np.array([[x, x] for x in range(5)]), np.array([[x, x] for x in range(3)]), np.array([[x, x] for x in range(4)])],
    [np.array([[x, x] for x in range(4)]), np.array([[x, x] for x in range(6)]), np.array([[x, x] for x in range(2)])]
]
for i, points_sets in enumerate(points_sets_list):
    print(f"Points sets {i+1}:", len(points_sets), "sets of points")
    for j, points in enumerate(points_sets):
        print(f"Points set {i+1}-{j+1}:", points.shape)

# Create custom dataset instance
custom_dataset = MyCustomDataset(image_paths, points_sets_list)

# Create a DataLoader with custom collate function
custom_dataloader = DataLoader(custom_dataset, batch_size=2, collate_fn=my_collate)

# Iterate through the DataLoader
for images1, images2, points_sets_list in custom_dataloader:
    # Process each batch
    print("Batch images1:", images1.shape)
    print("Batch images2:", images2.shape)
    for i, points_sets in enumerate(points_sets_list):
        print(f"Batch points_sets {i+1}:", points_sets.shape)