import torch
from torch.utils.data import Dataset


class SyntheticData(Dataset):
    def __init__(self, w: torch.Tensor, b: torch.Tensor, num_examples: int) -> None:
        self.X = torch.normal(0, 1, (num_examples, len(w)))
        self.y = torch.matmul(self.X, w) + b
        self.y += torch.normal(0, 0.01, self.y.shape)
        self.y.unsqueeze_(dim=-1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = torch.tensor(4.2)
    data = SyntheticData(true_w, true_b, 1000)
    print(f"{len(data)=}")
    print(f"{data[0]=}")
