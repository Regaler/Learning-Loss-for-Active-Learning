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
    elif method == "random":
        model = NormalModel(cf, resnet)
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

class NormalModel(pl.LightningModule):
    """
    The basic model that trains a resnet

    Parameters
    ----------
    backbone: nn.Module
        The backbone object
    """
    def __init__(self, cf, backbone):
        super().__init__()
        self.backbone = backbone  # returns scores and features
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.cf = cf

    def forward(self, x):
        scores, features = self.backbone(x)
        return scores

    def _get_loss(self, scores, y, features):
        return self.criterion(scores, y)

    def _get_acc(self, scores, y):
        _, preds = torch.max(scores.data, 1)
        total = y.size(0)
        correct = (preds == y).sum().item()
        acc = 100 * correct / total
        return acc

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.cuda()
        scores, features = self.backbone(x.cuda())
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
        total_loss = sum([x['loss'] for x in train_step_outputs])
        total_loss = total_loss / len(train_step_outputs)
        self.log("trn_loss", float(total_loss.cpu()))

    def validation_epoch_end(self, validation_step_outputs):
        total_acc, total_loss = map(sum, zip(*validation_step_outputs))
        total_acc = total_acc / len(validation_step_outputs)
        total_loss = total_loss / len(validation_step_outputs)
        self.log("val_acc", total_acc)
        self.log("val_loss", total_loss)

    def test_epoch_end(self, test_step_outputs):
        total_acc = sum([x for x in test_step_outputs])
        total_acc = total_acc / len(test_step_outputs)
        self.log("test_acc", total_acc)

    def predict_step(self, batch, dataloader_idx):
        x, _ = batch
        scores, features = self.backbone(x.cuda())
        return scores

    def configure_optimizers(self):
        if self.cf.optimizer == "SGD":
            optimizer = optim.SGD(self.backbone.parameters(),
                                       lr=self.cf.lr,
                                       momentum=self.cf.momentum,
                                       weight_decay=self.cf.wd)            
        elif self.cf.optimizer == "Adam":
            optimizer = optim.Adam(self.backbone.parameters(),
                                        lr=self.cf.lr,
                                        weight_decay=self.cf.wd)
        scheduler = lr_scheduler.MultiStepLR(
                                        optimizer,
                                        milestones=self.cf.milestones,
                                        gamma=0.1)
        return [optimizer], [scheduler]

class LL4AL(NormalModel):
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
        super().__init__(cf, backbone)
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

    def training_step(self, batch, batch_idx):
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

    def predict_step(self, batch, dataloader_idx):
        x, _ = batch
        _, features = self.backbone(x.cuda())
        pred_loss = self.losspred(features)
        return pred_loss

    def configure_optimizers(self):
        if self.cf.optimizer == "SGD":
            optimizer = optim.SGD(list(self.backbone.parameters()) + list(self.losspred.parameters()),
                                       lr=self.cf.lr,
                                       momentum=self.cf.momentum,
                                       weight_decay=self.cf.wd)
        elif self.cf.optimizer == "Adam":
            optimizer = optim.Adam(list(self.backbone.parameters()) + list(self.losspred.parameters()),
                                        lr=self.cf.lr,
                                        weight_decay=self.cf.wd)
        scheduler = lr_scheduler.MultiStepLR(
                                        optimizer,
                                        milestones=self.cf.milestones,
                                        gamma=0.1)
        return [optimizer], [scheduler]