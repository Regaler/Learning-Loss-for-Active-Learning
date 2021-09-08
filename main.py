'''
Active Learning Procedure in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning
(https://arxiv.org/abs/1905.03677)
'''

# Python
import random
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

# Torch
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Custom
# import config as cf
from data.sampler import SubsetSequentialSampler
from data.dataset import get_dataset
from models.al import get_model

# Seed
random.seed("Minuk Ma")
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True


def get_config():
    parser = ArgumentParser()
    # experiment-related
    parser.add_argument("--desc", default="Dummy", type=str,
                        help="The description of this experiment")
    parser.add_argument("--method", default="LL4AL", type=str,
                        choices=["random", "LL4AL"],
                        help="The AL method for the experiment")
    parser.add_argument("--trials", default=3, type=int,
                        help="The num of trials to repeat to reduce variance")
    parser.add_argument("--cycles", default=10, type=int,
                        help="The num of AL cycle to query")
    parser.add_argument("--dataset", default="CIFAR10", type=str,
                        choices=["CIFAR10", "CIFAR100"])
    # model related
    parser.add_argument("--backbone", default="resnet18", type=str,
                        choices=["resnet18", "resnet34"])
    parser.add_argument("--margin", metavar="XI", default=1.0, type=float,
                        help="The margin for the loss prediction module")
    parser.add_argument("--weight", metavar="LAMBDA", default=1.0, type=float,
                        help="loss weight for backbone vs loss prediction")
    # learning related
    parser.add_argument("--epoch", metavar="B", default=200, type=int,
                        help="Num of epoch to run each training")
    parser.add_argument("--batch", metavar="B", default=128, type=int,
                        help="The batch size for training / val / test")
    parser.add_argument('--optimizer', default='SGD', type=str,
                        choices=["SGD", "Adam"])
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float,
                        help="momentum used for SGD optimizer")
    parser.add_argument('--wd', default=5e-4, type=float,
                        help="weight decay for the optimizer")
    parser.add_argument('--milestones', nargs='+', default=[160],
                        help='The parameter for StepLRScheduler')
    parser.add_argument('--epochl', default=120, type=int,
                        help="After this epoch, stop gradient from the loss \
                            prediction module propagated to the target model")
    parser.add_argument("--subset", metavar="M", default=10000, type=int,
                        help="The number of unlabeled to consider at each cycle")
    parser.add_argument("--addendum", metavar="K", default=1000, type=int,
                        help="The number of samples to query at each cycle")

    cf = parser.parse_args()
    cf.milestones = [int(x) for x in cf.milestones]
    if cf.dataset == "CIFAR10":
        cf.num_train = 50000
        cf.num_val = 50000 - cf.num_train
        cf.num_classes = 10
    elif cf.dataset == "CIFAR100":
        cf.num_classes = 100
        raise ValueError(f"Please get this info for {cf.dataset}")
    return cf


def prepare_exp_result_dir(desc, dataset, trial):
    dateTimeObj = datetime.now()
    now = dateTimeObj.strftime("%d-%b-%Y-%H-%M-%S")
    outdir = f"./result/{dataset}/{now}_{desc}_{trial}"
    return outdir


def get_uncertainty(trainer, model, unlabeled_loader):
    predictions = trainer.predict(model, unlabeled_loader)
    return torch.cat(predictions, dim=0).squeeze().cpu()


def query_unlabeled_samples(cf, labeled_set, unlabeled_set,
                            unlabeled_dataset, trainer, model):
    """
    Get unlabeled samples to annotate for AL
    Randomly sample 10K unlabeled data points, to avoid too many unlabeled
    get_uncertainty() measures the info of the unlabeled samples
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
    subset = unlabeled_set[:cf.subset]
    # Create unlabeled dataloader for the unlabeled subset
    # more convenient if we maintain the order of subset for Sampler
    unlabeled_loader = DataLoader(unlabeled_dataset,
                                  batch_size=cf.batch,
                                  sampler=SubsetSequentialSampler(subset),
                                  pin_memory=True)
    uncertainty = get_uncertainty(trainer, model, unlabeled_loader)
    # Index in ascending order
    arg = np.argsort(uncertainty)
    # Update the labeled dataset and the unlabeled dataset, respectively
    labeled_set += list(torch.tensor(subset)[arg][-cf.addendum:].numpy())
    unlabeled_set = list(torch.tensor(subset)[arg][:-cf.addendum].numpy()) + \
        unlabeled_set[cf.subset:]
    # Create a new dataloader for the updated labeled dataset
    train_loader = DataLoader(dataset_train,
                              batch_size=cf.batch,
                              sampler=SubsetRandomSampler(labeled_set),
                              pin_memory=True)
    return labeled_set, unlabeled_set, train_loader


def run_AL_experiment(cf, labeled_set, unlabeled_set,
                      train_loader, test_loader, model):
    """
    Run typical AL experiment that gradually expands the dataset

    Parameters
    ----------
    labeled_set: list[int]
        The initial labeled samples that was selected randomly
    unlabeled_set: list[int]
        The initial unlabeled samples complementary to the labeled set
    train_loader: torch.utils.data.DataLoader
        The data loader with the initial labeled trn set
    test_loader: torch.utils.data.DataLoader
        The data loader with the initial labeled test set
    model: pytorch_lightning.LightningModule
        The model used for AL

    Returns
    -------
    list[float]
        The performance by the active learning cycle
        This is used to draw performance curve
    """
    result = []  # performance by cycle
    for _ in range(cf.cycles):
        checkpoint_callback = ModelCheckpoint(monitor="val_acc",
                                              save_last=True,
                                              save_top_k=1,
                                              mode='max')
        lr_callback = LearningRateMonitor(logging_interval='epoch')
        trainer = pl.Trainer(
            gpus=torch.cuda.device_count(),
            accelerator='dp',
            max_epochs=cf.epoch,
            accumulate_grad_batches=1,
            sync_batchnorm=True,
            default_root_dir=exp_dir,
            callbacks=[checkpoint_callback,
                        lr_callback]
        )
        trainer.fit(model, train_loader, test_loader)
        res = trainer.test(model, test_loader)
        result.append(res[0]["test_acc"])  # append float value

        # Annotate some unlabeled samples for active learning (AL)
        labeled_set, unlabeled_set, train_loader = \
            query_unlabeled_samples(
                cf=cf,
                labeled_set=labeled_set,
                unlabeled_set=unlabeled_set,
                unlabeled_dataset=dataset_unlabeled,
                trainer=trainer,
                model=model)
    return result


if __name__ == '__main__':
    cf = get_config()
    # Data
    dataset_train = get_dataset(
                        name=cf.dataset,
                        train=True,
                        download=True,
                        transform="train")
    dataset_unlabeled = get_dataset(
                            name=cf.dataset,
                            train=True,
                            download=True,
                            transform="test")
    dataset_test = get_dataset(
                        name=cf.dataset,
                        train=False,
                        download=True,
                        transform="test")

    # AL is sensitive to the choice of initial labeled set
    # Therefore, do the experiment many times
    for trial in range(cf.trials):
        exp_dir = prepare_exp_result_dir(cf.desc, cf.dataset, trial)
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000
        indices = list(range(cf.num_train))
        random.shuffle(indices)
        labeled_set = indices[:cf.addendum]
        unlabeled_set = indices[cf.addendum:]
        train_loader = DataLoader(
                            dataset_train,
                            batch_size=cf.batch,
                            sampler=SubsetRandomSampler(labeled_set),
                            pin_memory=True
                       )
        test_loader = DataLoader(
                            dataset_test,
                            batch_size=cf.batch
                      )
        model = get_model(cf, cf.method, cf.backbone)
        torch.backends.cudnn.benchmark = False

        # Run typical AL experiment that gradually expands the dataset
        result = run_AL_experiment(
                    cf,
                    labeled_set,
                    unlabeled_set,
                    train_loader,
                    test_loader,
                    model
        )
        print(f"The result at {trial}-th trial is: {result}")
        with open(f"result_{trial}.txt", 'w') as f:
            f.write("\n".join([str(x) for x in result]))

