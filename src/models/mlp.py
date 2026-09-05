from torch import nn

class MLP(nn.Module):
    def __init__(self, input_size, params):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = params.hidden_size
        self.output_size = params.output_size
        self.threshold = params.threshold

        self.model_params = {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "threshold": self.threshold,
        }

        self.training_params = {
            "lr": params.lr,
            "batch_size": params.batch_size,
            "epochs": params.epochs,
        }


        self.model = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(self.hidden_size,self.hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
