import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import pytorch_lightning as pl
import config as cf


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
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


class ALFramework(pl.LightningModule):
    def __init__(self, backbone, losspred):
        super().__init__()
        self.backbone = backbone
        self.losspred = losspred
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optim_backbone = optim.SGD(self.backbone.parameters(), 
                                        lr=cf.LR, 
                                        momentum=cf.MOMENTUM, 
                                        weight_decay=cf.WDECAY)
        self.optim_losspred = optim.SGD(self.losspred.parameters(), 
                                        lr=cf.LR, 
                                        momentum=cf.MOMENTUM, 
                                        weight_decay=cf.WDECAY)

    def forward(self, x):
        # Predict the loss of a sample
        pred_loss = self.losspred(x)
        return pred_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        y = y.cuda()
        scores, features = self.backbone(x.cuda())
        
        if cf.EPOCH > cf.EPOCHL:
            # After 120 epochs, stop the gradient from the loss prediction
            # module propagated to the target model.
            features[0] = features[0].detach()
            features[1] = features[1].detach()
            features[2] = features[2].detach()
            features[3] = features[3].detach()

        # Calculate the losses. Predict the loss as well. 
        pred_loss = self.losspred(features)
        pred_loss = pred_loss.view(pred_loss.size(0))
        target_loss = self.criterion(scores, y)
        backbone_loss = torch.sum(target_loss) / target_loss.size(0)
        losspred_loss = LossPredLoss(pred_loss, target_loss, margin=cf.MARGIN)
        loss = backbone_loss + cf.WEIGHT * losspred_loss
        return loss

    def train_epoch_end(self, train_step_outputs):
        total_acc = 0
        total_loss = 0
        for acc, loss in train_step_outputs:
            total_acc += acc
            total_loss += loss
        total_acc = total_acc / len(train_step_outputs)
        total_loss = total_loss / len(train_step_outputs)
        self.log("trn_acc", total_acc)
        self.log("trn_loss", total_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        total = 0
        correct = 0
        with torch.no_grad():
            y = y.cuda()
            scores, features = self.backbone(x.cuda())
            _, preds = torch.max(scores.data, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()
            # Get the losses
            pred_loss = self.losspred(features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            target_loss = self.criterion(scores, y)
            backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            losspred_loss = LossPredLoss(pred_loss, target_loss, margin=cf.MARGIN)
            loss = backbone_loss + cf.WEIGHT * losspred_loss
        acc = 100 * correct / total
        return acc, loss

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

    def predict_step(self, batch, dataloader_idx):
        x, _ = batch        
        _, features = self.backbone(x.cuda())
        pred_loss = self.losspred(features)
        return pred_loss

    def configure_optimizers(self):
        optimizer = [self.optim_backbone, self.optim_losspred]
        scheduler1 = lr_scheduler.MultiStepLR(
                                    self.optim_backbone, 
                                    milestones=cf.MILESTONES, 
                                    gamma=0.1)
        scheduler2 = lr_scheduler.MultiStepLR(
                                    self.optim_losspred, 
                                    milestones=cf.MILESTONES, 
                                    gamma=0.1)
        scheduler = [scheduler1, scheduler2]
        return optimizer, scheduler