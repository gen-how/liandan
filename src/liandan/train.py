import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .data.linear_regression import SyntheticData
from .models.linear_regression import LinearRegression


def main():
    true_w = torch.tensor([2, -3.4])
    true_b = torch.tensor(4.2)
    batch_size = 10
    dataset = SyntheticData(true_w, true_b, 1000)
    data_iter = DataLoader(dataset, batch_size, shuffle=True)
    net = LinearRegression(in_features=2, out_features=1)
    loss_fn = nn.MSELoss()
    trainer = optim.SGD(net.parameters(), lr=0.03)
    num_epochs = 3
    for epoch in range(num_epochs):
        for xs, ys in data_iter:
            loss = loss_fn(net(xs), ys)
            trainer.zero_grad()
            loss.backward()
            trainer.step()
        features, labels = dataset[:]
        loss = loss_fn(net(features), labels)
        print(f"epoch {epoch + 1}, {loss=:f}")
    print(f"{true_w=}, {true_b=}")
    print(f"regressed_w={net.flatten.weight.data}, regressed_b={net.flatten.bias.data}")


if __name__ == "__main__":
    main()
