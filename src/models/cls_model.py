import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, padding=1):
        super(SingleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, padding = padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ClassifierNet(nn.Module):
    def __init__(self):

        super(ClassifierNet, self).__init__()

        self.filters = [8,16,16,32,64]
        self.fc_length = 16
        self.layers = nn.Sequential()
        prev_channels = 3
        for f,next_channels in enumerate(self.filters):
            self.layers.add_module(f"block{f}",SingleBlock(prev_channels, next_channels))
            prev_channels = next_channels

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(self.filters[-1], self.fc_length )
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.fc_length , 2)

    def forward(self, x):
        for li,layer in enumerate(self.layers):
            x = layer(x)

        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
