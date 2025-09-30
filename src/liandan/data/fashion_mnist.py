from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

MNIST_TRAIN = FashionMNIST("./datasets", train=True, transform=ToTensor())
MNIST_TEST = FashionMNIST("./datasets", train=False, transform=ToTensor())


def get_data_loader(batch_size=256, num_workers=4):
    train_iter = DataLoader(
        MNIST_TRAIN, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_iter = DataLoader(
        MNIST_TEST, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return (train_iter, test_iter)


if __name__ == "__main__":
    from time import perf_counter

    from torch.utils.data import DataLoader

    train_iter = DataLoader(MNIST_TRAIN, batch_size=256, shuffle=True, num_workers=4)
    s = perf_counter()
    for xs, ys in train_iter:
        continue
    e = perf_counter()
    print(f"Read all samples using {e - s:.2f} s")
