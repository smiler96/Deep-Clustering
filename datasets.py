from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets import MNIST, STL10
from torchvision import transforms


class MNISTDataset(Dataset):
    def __init__(self):
        train = MNIST(root='./data/',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)

        test = MNIST(root='./data/',
                     train=False,
                     transform=transforms.ToTensor())

        self.data = ConcatDataset([train, test])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

class STL10Dataset(Dataset):
    def __init__(self):
        train = STL10(root='./data/',
                      split='train',
                      transform=transforms.ToTensor(),
                      download=True)

        test = STL10(root='./data/',
                     split='test',
                     transform=transforms.ToTensor())

        self.data = ConcatDataset([train, test])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

if __name__ == "__main__":
    dataset = STL10Dataset()
    x, y = dataset[0]
    print()
