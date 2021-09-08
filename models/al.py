# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import pytorch_lightning as pl

# custom
import models.resnet as rn
# import config as cf


def get_model(cf, method, backbone):
    """
    Get the proper model

    Parameters
    ----------
    method: str
        The active learning method
    backbone: str
        The resnet backbone to use

    Returns
    -------
    pytorch_lightning.LightningModule
    """
    # Select the backbone
    if backbone == "resnet18":
        resnet = rn.ResNet18(num_classes=cf.num_classes)
    else:
        raise ValueError(f"The backbone {backbone} is not supported. ")
    # Get the Lightning model
    if method == "LL4AL":
        losspred = LossNet()
        model = LL4AL(cf, resnet, losspred)
    else:
        raise ValueError(f"The policy {method} is not supported. ")
    return model


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    # assert len(input) % 2 == 0, 'the batch size is not even.'
    # assert input.shape == input.flip(0).shape
    if len(input) % 2 == 0:
        pass
    else:
        input = input[:-1]
        target = target[:-1]

    # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    input = (input - input.flip(0))[:len(input)//2]
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()
    # 1 operation which is defined by the authors
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        # Note that the size of input is already halved
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    return loss


class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4],
                 num_channels=[64, 128, 256, 512], interm_dim=128):
        super(LossNet, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out


class LL4AL(pl.LightningModule):
    """
    The Lightning class for LL4AL method

    Parameters
    ----------
    backbone: nn.Module
        The backbone object
    losspred: nn.Module
        The loss prediction module object
    """
    def __init__(self, cf, backbone, losspred):
        super().__init__()
        self.backbone = backbone
        self.losspred = losspred
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.cf = cf

    def forward(self, x):
        # Predict the loss of a sample
        pred_loss = self.losspred(x)
        return pred_loss

    def _get_loss(self, scores, y, features):
        # Calculate the losses. Predict the loss as well.
        pred_loss = self.losspred(features)
        pred_loss = pred_loss.view(pred_loss.size(0))
        target_loss = self.criterion(scores, y)
        backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        losspred_loss = LossPredLoss(pred_loss, target_loss,
                                     margin=self.cf.margin)
        loss = backbone_loss + self.cf.weight * losspred_loss
        return loss

    def _get_acc(self, scores, y):
        _, preds = torch.max(scores.data, 1)
        total = y.size(0)
        correct = (preds == y).sum().item()
        acc = 100 * correct / total
        return acc

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        y = y.cuda()
        scores, features = self.backbone(x.cuda())

        if self.current_epoch > self.cf.epochl:
            # After 120 epochs, stop the gradient from the loss prediction
            # module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        loss = self._get_loss(scores, y, features)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y = y.cuda()
            scores, features = self.backbone(x.cuda())
            acc = self._get_acc(scores, y)
            loss = self._get_loss(scores, y, features)
        return acc, loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            y = y.cuda()
            scores, _ = self.backbone(x.cuda())
            acc = self._get_acc(scores, y)
        return acc

    def training_epoch_end(self, train_step_outputs):
        total_loss = 0
        for loss in train_step_outputs:
            total_loss += sum([x['loss'] for x in loss])
        total_loss = total_loss / len(train_step_outputs)
        self.log("trn_loss", float(total_loss.cpu()))

    def validation_epoch_end(self, validation_step_outputs):
        total_acc = 0
        total_loss = 0
        for acc, loss in validation_step_outputs:
            total_acc += acc
            total_loss += loss
        total_acc = total_acc / len(validation_step_outputs)
        total_loss = total_loss / len(validation_step_outputs)
        self.log("val_acc", total_acc)
        self.log("val_loss", total_loss)

    def test_epoch_end(self, test_step_outputs):
        total_acc = 0
        for acc in test_step_outputs:
            total_acc += acc
        total_acc = total_acc / len(test_step_outputs)
        self.log("test_acc", total_acc)

    def predict_step(self, batch, dataloader_idx):
        x, _ = batch
        _, features = self.backbone(x.cuda())
        pred_loss = self.losspred(features)
        return pred_loss

    def configure_optimizers(self):
        if self.cf.optimizer == "SGD":
            optim_backbone = optim.SGD(self.backbone.parameters(),
                                       lr=self.cf.lr,
                                       momentum=self.cf.momentum,
                                       weight_decay=self.cf.wd)
            optim_losspred = optim.SGD(self.losspred.parameters(),
                                       lr=self.cf.lr,
                                       momentum=self.cf.lr,
                                       weight_decay=self.cf.wd)
            
        elif self.cf.optimizer == "Adam":
            optim_backbone = optim.Adam(self.backbone.parameters(),
                                        lr=self.cf.lr,
                                        weight_decay=self.cf.wd)
            optim_losspred = optim.Adam(self.losspred.parameters(),
                                        lr=self.cf.lr,
                                        weight_decay=self.cf.wd)
        scheduler1 = lr_scheduler.MultiStepLR(
                                        optim_backbone,
                                        milestones=self.cf.milestones,
                                        gamma=0.1)
        scheduler2 = lr_scheduler.MultiStepLR(
                                        optim_losspred,
                                        milestones=self.cf.milestones,
                                        gamma=0.1)
        optimizer = [optim_backbone, optim_losspred]
        scheduler = [scheduler1, scheduler2]
        return optimizer, scheduler
