from pathlib import Path

from torchvision import datasets


def main():
    path = Path(__file__).parent / "../datasets"
    train = datasets.MNIST(path, train=True, download=True)
    valid = datasets.MNIST(path, train=False, download=True)
    print(f"{len(train)=}, {len(valid)=}")
    print(f"{train[0]=}")


if __name__ == "__main__":
    main()
