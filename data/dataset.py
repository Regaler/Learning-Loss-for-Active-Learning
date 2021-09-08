from torchvision.datasets import CIFAR100, CIFAR10
import torchvision.transforms as T


def get_dataset(name, train=True, download=True, transform="train"):
    assert transform in ["train", "test"]
    if name == "CIFAR10":
        if transform == "train":
            trans = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=32, padding=4),
                T.ToTensor(),
                T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        elif transform == "test":
            trans = T.Compose([
                T.ToTensor(),
                T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
        return CIFAR10('../cifar10', train=train, download=download, transform=trans)
    elif name == "CIFAR100":
        if transform == "train":
            trans = T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=32, padding=4),
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        elif transform == "test":
            trans = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        return CIFAR100('../cifar100', train=train, download=download, transform=trans)
    else:
        print(f"The dataset {name} is not supported. ")