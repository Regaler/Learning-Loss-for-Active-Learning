'''
Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning
(https://arxiv.org/abs/1905.03677)
'''

# Python
import random
import numpy as np

# Torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchvision.transforms as T

# Custom
import config as cf
from data.sampler import SubsetSequentialSampler
from data.dataset import get_dataset
from models.al import get_model


def get_uncertainty(trainer, model, unlabeled_loader):
    predictions = trainer.predict(model, unlabeled_loader)
    return torch.cat(predictions, dim=0).squeeze().cpu()


def query_unlabeled_samples(labeled_set, unlabeled_set, 
                            unlabeled_dataset, trainer, model):
    """
    Get unlabeled samples to annotate for active learning
    Randomly sample 10000 unlabeled data points, as there can be too many unlabeled
    The get_uncertainty function measures the information of the unlabeled samples
    As a result, the labeled set, unlabeled sets, and train_loader are updated

    Parameters
    ----------
    labeled_set: list[int]
        The indices of the labeled samples in the whole data pool
    unlabeled_set: list[int]
        The indices of the unlabeled samples in the whole data pool
    unlabeled_dataset: torch.utils.data.Dataset
        The actual unlabeled dataset object
        Each sample in this dataset will be measure for uncertainty
    trainer: pytorch_lightning.Trainer
        The trainer to run inference
    model: pytorch_lightning.LightningModule
        The trained model

    Returns
    -------
    labeled_set: list[int]
        The updated indices of the labeled samples in the whole data pool
    unlabeled_set: list[int]
        The updated indices of the unlabeled samples in the whole data pool
    train_loader: torch.utils.data.DataLoader
        The updated train loader with some more annotated data
    """
    random.shuffle(unlabeled_set)
    subset = unlabeled_set[:cf.SUBSET]
    # Create unlabeled dataloader for the unlabeled subset
    # more convenient if we maintain the order of subset for Sampler
    unlabeled_loader = DataLoader(unlabeled_dataset,
                                    batch_size=cf.BATCH,
                                    sampler=SubsetSequentialSampler(subset),
                                    pin_memory=True)
    uncertainty = get_uncertainty(trainer, model, unlabeled_loader)
    # Index in ascending order
    arg = np.argsort(uncertainty)
    # Update the labeled dataset and the unlabeled dataset, respectively
    labeled_set += list(torch.tensor(subset)[arg][-cf.ADDENDUM:].numpy())
    unlabeled_set = list(torch.tensor(subset)[arg][:-cf.ADDENDUM].numpy()) + unlabeled_set[cf.SUBSET:]
    # Create a new dataloader for the updated labeled dataset
    train_loader = DataLoader(cifar10_train, batch_size=cf.BATCH,
                            sampler=SubsetRandomSampler(labeled_set),
                            pin_memory=True)
    return labeled_set, unlabeled_set, train_loader


if __name__ == '__main__':
    # Seed
    random.seed("Minuk Ma")
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    # Data
    cifar10_train = get_dataset(name="CIFAR10", train=True, download=True, transform="train")
    cifar10_unlabeled = get_dataset(name="CIFAR10", train=True, download=True, transform="test")
    cifar10_test = get_dataset(name="CIFAR10", train=False, download=True, transform="test")

    for trial in range(cf.TRIALS):
        exp_dir = f"./result/cifar10/train/trial_{trial}"
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000
        indices = list(range(cf.NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:cf.ADDENDUM]
        unlabeled_set = indices[cf.ADDENDUM:]

        train_loader = DataLoader(cifar10_train, batch_size=cf.BATCH,
                                  sampler=SubsetRandomSampler(labeled_set),
                                  pin_memory=True)
        test_loader = DataLoader(cifar10_test, batch_size=cf.BATCH)

        model = get_model(method="LL4AL", backbone="resnet18", num_classes=10)
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
            
            # Annotate some unlabeled samples for active learning (AL)
            labeled_set, unlabeled_set, train_loader = \
                query_unlabeled_samples(
                    labeled_set=labeled_set,
                    unlabeled_set=unlabeled_set, 
                    unlabeled_dataset=cifar10_unlabeled,
                    trainer=trainer,
                    model=model)