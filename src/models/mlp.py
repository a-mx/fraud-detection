from torch import nn

class MLP(nn.Module):
    def __init__(self, threshold, input_size, hidden_size, output_size):
        super().__init__()
        self.threshold = threshold
        self.model = nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
