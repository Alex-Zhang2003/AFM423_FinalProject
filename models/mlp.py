import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class MLP(nn.Module):

    def __init__(self, input_dim=(100, 40)):
        super(MLP, self).__init__()

        self.input_size = input_dim[0] * input_dim[1]

        self.fc1 = nn.Linear(self.input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.3)

        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)

        self.fc5 = nn.Linear(64, 32)

        self.fc_out = nn.Linear(32, 3)


    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size, -1)

        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout1(x)

        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout2(x)

        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout3(x)

        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.01)

        x = F.leaky_relu(self.fc5(x), negative_slope=0.01)

        x = self.fc_out(x)

        return F.softmax(x, dim=1)


if __name__ == '__main__':
    model = MLP()
    summary(model, (1, 1, 100, 40))
