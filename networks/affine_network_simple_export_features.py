import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
# from pytorch_model_summary import summary


class Affine_Network(nn.Module):
    def __init__(self, device):
        super(Affine_Network, self).__init__()
        self.device = device

        self.feature_extractor = Feature_Extractor(self.device)
        self.regression_network = Regression_Network()

    def forward(self, source, target):
        print("class Affine_Network")
        print(torch.cat((source, target), dim=1).shape)
        x = self.feature_extractor(torch.cat((source, target), dim=1))
        print("After self.feature_extractor:", x.shape)
        # x = x.view(1, -1)
        # squeeze to [batch_size, 256]
        x = x.squeeze()
        print("After squeeze:", x.shape)
        x = self.regression_network(x)
        print("After regression, x shape:", x.shape)
        return x

class Regression_Network(nn.Module):
    def __init__(self):
        super(Regression_Network, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(256, 6),
        )

    def forward(self, x):
        # reshape to [batch_size, 256]
        # x = x.view(-1, 256)
        x = self.fc(x)
        print(f"Regression Network: {x.shape}")
        return x.view(-1, 2, 3)

class Forward_Layer(nn.Module):
    def __init__(self, channels, pool=False):
        super(Forward_Layer, self).__init__()
        self.pool = pool
        if self.pool:
            self.pool_layer = nn.Sequential(
                nn.Conv2d(channels, 2*channels, 3, stride=2, padding=3)
            )
            self.layer = nn.Sequential(
                nn.Conv2d(channels, 2*channels, 3, stride=2, padding=3),
                nn.GroupNorm(2*channels, 2*channels),
                nn.PReLU(),
                nn.Conv2d(2*channels, 2*channels, 3, stride=1, padding=1),
                nn.GroupNorm(2*channels, 2*channels),
                nn.PReLU(),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                nn.GroupNorm(channels, channels),
                nn.PReLU(),
                nn.Conv2d(channels, channels, 3, stride=1, padding=1),
                nn.GroupNorm(channels, channels),
                nn.PReLU(),
            )

    def forward(self, x):
        if self.pool:
            return self.pool_layer(x) + self.layer(x)
        else:
            return x + self.layer(x)

class Feature_Extractor(nn.Module):
    def __init__(self, device):
        super(Feature_Extractor, self).__init__()
        self.device = device
        self.input_layer = nn.Sequential(
            nn.Conv2d(2, 64, 7, stride=2, padding=3),
        )
        self.layer_1 = Forward_Layer(64, pool=True)
        self.layer_2 = Forward_Layer(128, pool=False)
        self.layer_3 = Forward_Layer(128, pool=True)
        self.layer_4 = Forward_Layer(256, pool=False)
        self.layer_5 = Forward_Layer(256, pool=True)
        self.layer_6 = Forward_Layer(512, pool=True)

        self.last_layer = nn.Sequential(
            nn.Conv2d(1024, 512, 3, stride=2, padding=1),
            nn.GroupNorm(512, 512),
            nn.PReLU(),
            nn.Conv2d(512, 256, 3, stride=2, padding=1),
            nn.GroupNorm(256, 256),
            nn.PReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        print("class Feature_Extractor")
        print("Input shape:", x.shape)
        x = self.input_layer(x)
        print("After input_layer shape:", x.shape)
        # subplot the feature maps
        
        x = self.layer_1(x)
        print("After layer_1 shape:", x.shape)
        x = self.layer_2(x)
        print("After layer_2 shape:", x.shape)
        x = self.layer_3(x)
        print("After layer_3 shape:", x.shape)
        x = self.layer_4(x)
        print("After layer_4 shape:", x.shape)
        x = self.layer_5(x)
        print("After layer_5 shape:", x.shape)
        x = self.layer_6(x)
        print("After layer_6 shape:", x.shape)
        x = self.last_layer(x)
        print("After last_layer shape:", x.shape)
        return x

def load_network(device, path=None):
    model = Affine_Network(device)
    model = model.to(device)
    # summary(model, [(1, 256, 256), (1, 256, 256)])
    if path is not None:
        model.load_state_dict(torch.load(path))
        model.eval()
    return model

def test_forward_pass():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_network(device)
    y_size = 256
    x_size = 256
    no_channels = 1
    # summary(model, [(no_channels, y_size, x_size), (no_channels, y_size, x_size)])
    # summary(model, [(no_channels, y_size, x_size), (no_channels, y_size, x_size)])


    batch_size = 4
    example_source = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)
    example_target = torch.rand((batch_size, no_channels, y_size, x_size)).to(device)

    # example_source[:, :, 200:500, 50:450] = 1
    # example_target[:, :, 100:600, 200:400] = 1

    result = model(example_source, example_target)
    print(result.size())


def run():
    test_forward_pass()

if __name__ == "__main__":
    run()

