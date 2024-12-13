import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size, stride=1, padding=1)
        self.elu = nn.ELU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.elu(x)
        x = self.conv2(x)
        x = self.elu(x)
        x = self.maxpool(x)
        return x

class EEGCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EEGCNN, self).__init__()
        self.blockcv1 = Block(input_channels, 32, 3)
        self.blockcv2 = Block(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.blockcv1(x)
        x = self.blockcv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)
        return x