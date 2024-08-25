import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGCNN(nn.Module):
    def __init__(self):
        super(EEGCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1,256), bias=False),
            nn.BatchNorm2d(3),
            nn.Tanh(),
        )
        self.maxpool = nn.MaxPool2d([1,11],stride=[1,11],padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=873, out_features=8, bias=False)
        self.initialize_weights()
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


