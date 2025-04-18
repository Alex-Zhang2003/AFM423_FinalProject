import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else "cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.name = 'cnn'

        # 2D convolution to extract spatial features from LOB data
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 40))

        # 1D convolutions to process feature sequences
        self.conv1d_1 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4)
        self.conv1d_2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv1d_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)

        # Fully connected layer for classification
        self.fc = nn.Linear(32, 3)

    def forward(self, x):

        x = F.relu(self.conv2d(x))
        x = x.squeeze(3)   # remove the last dimension after 2D conv

        x = F.leaky_relu(self.conv1d_1(x), negative_slope=0.01)
        x = F.max_pool1d(x, kernel_size=2)

        x = F.leaky_relu(self.conv1d_2(x), negative_slope=0.01)
        x = F.leaky_relu(self.conv1d_3(x), negative_slope=0.01)
        x = F.max_pool1d(x, kernel_size=2)

        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return F.softmax(x, dim=1)


if __name__ == '__main__':
    model = CNN()
    summary(model, (1, 1, 100, 40))
