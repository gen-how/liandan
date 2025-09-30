from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flatten = nn.Linear(in_features, out_features)
        self.flatten.weight.data.normal_(0, 0.01)
        self.flatten.bias.data.fill_(0)

    def forward(self, x):
        return self.flatten(x)
