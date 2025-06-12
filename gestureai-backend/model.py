# model.py
import torch.nn as nn
import torch.nn.functional as F

class Gesture3DCNN(nn.Module):
    def __init__(self):
        super(Gesture3DCNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        self.fc5 = nn.Linear(512 * 2 * 5 * 5, 512)  # adjust input size
        self.fc5_ln = nn.BatchNorm1d(512)
        self.fc6 = nn.Linear(512, 7)

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = F.adaptive_avg_pool3d(x, (2, 5, 5))  # downsample to match fc5 input
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.fc5_ln(x)
        x = F.relu(x)
        x = self.fc6(x)
        return x
