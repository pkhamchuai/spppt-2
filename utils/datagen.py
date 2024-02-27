import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# import MNIST dataset
# from torchvision.datasets import MNIST
from torchvision import transforms
import cv2
import numpy as np

img_size = 256

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, df, is_train, sup, transform=None):
        self.dataset_path = dataset_path
        self.df = df
        self.is_train = is_train
        self.sup = sup
        if transform is not None:
            self.transform = transform
        elif transform is None and self.is_train == False: 
            # if test, no random transformation
            self.transform = transforms.ToTensor()
        elif transform is None and self.sup == True and self.is_train == True:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))
            ])
        elif transform is None and self.sup == False and self.is_train == True: 
            # if unsupervised, apply random transformation too
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source_path = row['source']
        target_path = row['target']
        # print(idx, source_path, target_path)
        source_img = cv2.imread(source_path, 0)
        target_img = cv2.imread(target_path, 0)
        source_img = cv2.resize(source_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        target_img = cv2.resize(target_img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # convert images to float32
        source_img = (source_img/np.max(source_img)).astype(np.float32)
        target_img = (target_img/np.max(target_img)).astype(np.float32)
        source_img = self.transform(source_img)
        target_img = self.transform(target_img)

        keypoints_path = row['keypoints']
        # import keypoints, which is a csv file
        keypoints = pd.read_csv(keypoints_path)
        keypoints = keypoints.to_numpy()
        keypoints = keypoints.astype(np.float32)

        if self.sup == True:
            affine_params = np.array([[row['M00'], row['M01'], row['M02']], [row['M10'], row['M11'], row['M12']]]).astype(np.float32)
            # if supervised, keypoints have 2 more columns
            matches1 = np.array(keypoints[:, 0:2]).astype(np.float32)
            matches2 = np.array(keypoints[:, 2:4]).astype(np.float32)
            matches1_2 = np.array(keypoints[:, 4:6]).astype(np.float32)
            return source_img, target_img, affine_params, matches1, matches2, matches1_2
        elif self.sup == False:
            matches1 = np.array(keypoints[:, 0:2]).astype(np.float32)
            matches2 = np.array(keypoints[:, 2:4]).astype(np.float32)
            return source_img, target_img, np.array([]), matches1, matches2, np.array([])

class MyDataset_batch(torch.utils.data.Dataset):
    def __init__(self, dataset_path, df, is_train, sup, transform=None):
        self.dataset_path = dataset_path
        self.df = df
        self.is_train = is_train
        self.sup = sup
        if transform is not None:
            self.transform = transform
        elif transform is None and self.is_train == False: 
            # if test, no random transformation
            self.transform = transforms.ToTensor()
        elif transform is None and self.sup == True and self.is_train == True:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x))
            ])
        elif transform is None and self.sup == False and self.is_train == True: 
            # if unsupervised, apply random transformation too
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + 0.01 * torch.randn_like(x)),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        source_path = row['source']
        target_path = row['target']
        # print(idx, source_path, target_path)
        source_img = cv2.imread(source_path, 0)
        target_img = cv2.imread(target_path, 0)
        source_img = cv2.resize(source_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        target_img = cv2.resize(target_img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        # convert images to float32
        source_img = (source_img/np.max(source_img)).astype(np.float32)
        target_img = (target_img/np.max(target_img)).astype(np.float32)
        source_img = self.transform(source_img)
        target_img = self.transform(target_img)

        keypoints_path = row['keypoints']
        # import keypoints, which is a csv file
        keypoints = pd.read_csv(keypoints_path)
        keypoints = keypoints.to_numpy()
        keypoints = keypoints.astype(np.float32)

        if self.sup == True:
            affine_params = np.array([[row['M00'], row['M01'], row['M02']], [row['M10'], row['M11'], row['M12']]]).astype(np.float32)
            # if supervised, keypoints have 2 more columns
            matches1 = np.array(keypoints[:, 0:2]).astype(np.float32)
            matches2 = np.array(keypoints[:, 2:4]).astype(np.float32)
            matches1_2 = np.array(keypoints[:, 4:6]).astype(np.float32)
            # for i in [matches1, matches2, matches1_2]:
            #     print(np.array(i).shape)
            return source_img, target_img, affine_params, matches1, matches2, matches1_2
        elif self.sup == False:
            matches1 = np.array(keypoints[:, 0:2]).astype(np.float32)
            matches2 = np.array(keypoints[:, 2:4]).astype(np.float32)
            return source_img, target_img, np.array([]), matches1, matches2, np.array([])
        
    # def get_points(self, idx):
    #     row = self.df.iloc[idx]
    #     keypoints_path = row['keypoints']
    #     keypoints = pd.read_csv(keypoints_path)
    #     keypoints = keypoints.to_numpy()
    #     keypoints = keypoints.astype(np.float32)

    #     if self.sup == True:
    #         matches1 = np.array(keypoints[:, 0:2]).astype(np.float32)
    #         matches2 = np.array(keypoints[:, 2:4]).astype(np.float32)
    #         matches1_2 = np.array(keypoints[:, 4:6]).astype(np.float32)
    #         return matches1, matches2, matches1_2
    #     elif self.sup == False:
    #         matches1 = np.array(keypoints[:, 0:2]).astype(np.float32)
    #         matches2 = np.array(keypoints[:, 2:4]).astype(np.float32)
    #         return matches1, matches2, np.array([])
        
# Custom collate function
def my_collate(batch):
    # Separate the elements of the batch
    images1, images2, affine_params, matches1, matches2, matches1_2 = zip(*batch)
    points_sets_list = [matches1, matches2, matches1_2]

    # Pad points within each batch
    max_num_points = max(len(points) for points_set in points_sets_list for points in points_set)
    padded_points_sets_list = []
    for points_set in points_sets_list:
        padded_points_set = []
        for points in points_set:
            num_points_to_pad = max_num_points - len(points)
            padded_points = np.pad(points, ((0, num_points_to_pad), (0, 0)), mode='constant', constant_values=0) if num_points_to_pad > 0 else points
            padded_points = np.array(padded_points)
            padded_points_set.append(padded_points)
        padded_points_sets_list.append(padded_points_set)
    # print('padded_points_sets_list: ', len(padded_points_sets_list))
    # for i in padded_points_sets_list:
    #     print('padded_points_sets_list[i]: ', len(i))
    #     for j in i:
    #         print('padded_points_sets_list[i][j]: ', j.shape)

    # Convert lists to tensors
    images1 = torch.stack(images1)
    images2 = torch.stack(images2)
    points_sets_list = [torch.tensor(padded_points_sets, dtype=torch.float32) for padded_points_sets in padded_points_sets_list]
    matches1 = points_sets_list[0]
    matches2 = points_sets_list[1]
    matches1_2 = points_sets_list[2]
    affine_params = torch.tensor(affine_params, dtype=torch.float32)

    return images1, images2, affine_params, matches1, matches2, matches1_2
        

def datagen(dataset, is_train, sup, batch_size=1):
    if dataset == 0:
        # actual eye dataset
        dataset_path = 'Dataset/Dataset-processed'
        if is_train:
            df = pd.read_csv('Dataset/dataset_eye_actual_keypoints.csv')
            # count number of rows that df['training'] == 1
            print('Training eye dataset')
            print('Number of training data: ', len(df[df['training'] == 1]))
            df = df[df['training'] == 1]
        else:
            df = pd.read_csv('Dataset/dataset_eye_actual_keypoints.csv')
            # count number of rows that df['training'] == 0
            print('Test eye dataset')
            print('Number of testing data: ', len(df[df['training'] == 0]))
            df = df[df['training'] == 0]

    elif dataset == 1:
        if is_train:
            # synthetic eye dataset
            dataset_path = 'Dataset/synth_eye_easy_train'
            df = pd.read_csv('Dataset/synth_eye_easy_train.csv')
        else:
            # synthetic eye dataset
            dataset_path = 'Dataset/synthetic_eye_easy_test'
            df = pd.read_csv('Dataset/synth_eye_easy_test.csv')
            
    elif dataset == 2:
        if is_train:
            # synthetic eye dataset medium
            dataset_path = 'Dataset/synthetic_eye_medium_train'
            df = pd.read_csv('Dataset/synth_eye_medium_train.csv')
        else:
            # synthetic eye dataset medium
            dataset_path = 'Dataset/synthetic_eye_medium_test'
            df = pd.read_csv('Dataset/synth_eye_medium_test.csv')

    elif dataset == 3:
        if is_train:
            # synthetic eye dataset hard
            dataset_path = 'Dataset/synthetic_eye_hard_train'
            df = pd.read_csv('Dataset/synth_eye_hard_train.csv')
        else:
            # synthetic eye dataset hard
            dataset_path = 'Dataset/synthetic_eye_hard_test'
            df = pd.read_csv('Dataset/synth_eye_hard_test.csv')

    elif dataset == 4:
        # new easy dataset
        if is_train:
            # synthetic eye dataset
            dataset_path = 'Dataset/synth_eye_easy_2000_train'
            df = pd.read_csv('Dataset/synth_eye_easy_2000_train.csv')
        else:
            # synthetic eye dataset
            dataset_path = 'Dataset/synthetic_eye_easy_2000_test'
            df = pd.read_csv('Dataset/synth_eye_easy_2000_test.csv')

    # elif dataset == 4:
    #     # synthetic shape dataset
    #     dataset_path = 'Dataset/synthetic_shape_dataset'
    #     if is_train:
    #         df = pd.read_csv('Dataset/dataset_shape_synth_train.csv')
    #     else:  
    #         df = pd.read_csv('Dataset/dataset_shape_synth_test.csv')

    # elif dataset == 5:
    #     # MNIST dataset
    #     if is_train:
    #         dataset_path = 'Dataset/MNIST'
    #         dataset = MNIST(dataset_path, train=True, transform=transforms.ToTensor(), download=True)
    #     else:
    #         dataset_path = 'Dataset/MNIST'
    #         dataset = MNIST(dataset_path, train=False, transform=transforms.ToTensor(), download=True)
    #     dataloader = DataLoader(dataset, batch_size=1, shuffle=is_train)
    #     return dataloader

    else:
        raise ValueError('Input dataset parameter 0-4')

    if batch_size == 1:
        dataset = MyDataset(dataset_path, df, is_train, sup)
        dataloader = DataLoader(dataset, shuffle=is_train, batch_size=1)
        return dataloader
    else:
        dataset = MyDataset_batch(dataset_path, df, is_train, sup)
        dataloader = DataLoader(dataset, shuffle=is_train, batch_size=batch_size, collate_fn=my_collate)
        return dataloader #, df, dataset_path

