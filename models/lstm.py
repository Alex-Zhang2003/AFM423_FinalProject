import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class LSTM(nn.Module):

    def __init__(self, input_size=40, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.Linear(hidden_size * 2, 1)

        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)

        self.fc_out = nn.Linear(32, 3)

    def forward(self, x):


        x = x.squeeze(1)

        lstm_out, _ = self.lstm(x)

        attention_weights = F.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        attention_weights = attention_weights.unsqueeze(-1)

        context = torch.sum(lstm_out * attention_weights, dim=1)

        x = F.leaky_relu(self.fc1(context), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)

        x = self.fc_out(x)

        return F.softmax(x, dim=1)


if __name__ == '__main__':
    model = LSTM()
    summary(model, (1, 1, 100, 40))