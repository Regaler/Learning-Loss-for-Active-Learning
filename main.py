'''
Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning
(https://arxiv.org/abs/1905.03677)
'''

# Python
from models.al import ALFramework
import random

# Torch
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Torchvison
import torchvision.transforms as T
from torchvision.datasets import CIFAR100, CIFAR10

# Custom
import models.resnet as resnet
import models.lossnet as lossnet
import config as cf
from data.sampler import SubsetSequentialSampler


def get_uncertainty(trainer, model, unlabeled_loader):
    predictions = trainer.predict(model, unlabeled_loader)
    return torch.cat(predictions, dim=0).squeeze().cpu()


if __name__ == '__main__':
    # Seed
    random.seed("Minuk Ma")
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    # Data
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100

    cifar10_train = CIFAR10('../cifar10', train=True,
                            download=True, transform=train_transform)
    cifar10_unlabeled = CIFAR10('../cifar10', train=True,
                                download=True, transform=test_transform)
    cifar10_test = CIFAR10('../cifar10', train=False,
                           download=True, transform=test_transform)

    for trial in range(cf.TRIALS):
        exp_dir = f"./result/cifar10/train/trial_{trial}"
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000
        # data points from the entire dataset.
        indices = list(range(cf.NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:cf.ADDENDUM]
        unlabeled_set = indices[cf.ADDENDUM:]

        train_loader = DataLoader(cifar10_train, batch_size=cf.BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(cifar10_test, batch_size=cf.BATCH)

        resnet18 = resnet.ResNet18(num_classes=10).cuda()
        losspred = lossnet.LossNet().cuda()
        model = ALFramework(resnet18, losspred)
        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(cf.CYCLES):
            checkpoint_callback = ModelCheckpoint(monitor="val_acc",
                                                  save_last=True,
                                                  save_top_k=1,
                                                  mode='max')
            lr_callback = LearningRateMonitor(logging_interval='epoch')
            trainer = pl.Trainer(
                gpus=torch.cuda.device_count(),
                max_epochs=cf.EPOCH,
                accumulate_grad_batches=1,
                sync_batchnorm=True,
                default_root_dir=exp_dir,
                callbacks=[checkpoint_callback,
                           lr_callback]
            )
            trainer.fit(model, train_loader, test_loader)
            trainer.test(model, test_loader)

            # Update the labeled dataset 
            # via loss prediction-based uncertainty measurement
            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:cf.SUBSET]

            # Create unlabeled dataloader for the unlabeled subset
            # more convenient if we maintain the order of subset for Sampler
            unlabeled_loader = DataLoader(cifar10_unlabeled,
                                          batch_size=cf.BATCH,
                                          sampler=SubsetSequentialSampler(subset),
                                          pin_memory=True)

            uncertainty = get_uncertainty(trainer, model, unlabeled_loader)
            # Index in ascending order
            arg = np.argsort(uncertainty)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-cf.ADDENDUM:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][:-cf.ADDENDUM].numpy()) + unlabeled_set[SUBSET:]

            # Create a new dataloader for the updated labeled dataset
            train_loader = DataLoader(cifar10_train, batch_size=cf.BATCH,
                                      sampler=SubsetRandomSampler(labeled_set),
                                      pin_memory=True)

        # Save a checkpoint
        torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': model.backbone.state_dict(),
                    'state_dict_losspred': model.losspred.state_dict()
                },
                f'{exp_dir}/resnet18.pth')
