from pathlib import Path

from torchvision import datasets


def main():
    path = Path(__file__).parent / "../datasets"
    mnist_train = datasets.FashionMNIST(path, train=True, download=True)
    mnist_test = datasets.FashionMNIST(path, train=False, download=True)
    print(f"{len(mnist_train)=}, {len(mnist_test)=}")
    print(f"{mnist_train[0]=}")


if __name__ == "__main__":
    main()
