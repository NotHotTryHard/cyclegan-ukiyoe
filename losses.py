import torch
import torch.nn as nn


class CycleConsistencyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CycleConsistencyLoss, self).__init__()
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, x, x_rec):
        return self.criterion(x_rec, x)


class AdversarialLossCE(nn.Module):
    def __init__(self, reduction='mean'):
        super(AdversarialLossCE, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def loss_real(self, pred):
        target = torch.ones_like(pred)
        return self.bce(pred, target)

    def loss_fake(self, pred):
        target = torch.zeros_like(pred)
        return self.bce(pred, target)

    def forward(self, real_pred, fake_pred=None):
        if fake_pred is None:
            return self.loss_real(real_pred)
        else:
            loss_real = self.loss_real(real_pred)
            loss_fake = self.loss_fake(fake_pred)
            return (loss_real + loss_fake) / 2.0


class AdversarialLossMSE(nn.Module):
    def __init__(self, reduction='mean'):
        super(AdversarialLossMSE, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def loss_real(self, pred):
        target = torch.ones_like(pred)
        return self.mse(pred, target)

    def loss_fake(self, pred):
        target = torch.zeros_like(pred)
        return self.mse(pred, target)

    def forward(self, real_pred, fake_pred=None):
        if fake_pred is None:
            return self.loss_real(real_pred)
        else:
            loss_real = self.loss_real(real_pred)
            loss_fake = self.loss_fake(fake_pred)
            return (loss_real + loss_fake) / 2.0


class FullDiscriminatorLoss(nn.Module):
    def __init__(self, is_mse=True, reduction='mean'):
        super(FullDiscriminatorLoss, self).__init__()
        self.adversarial_loss_func = AdversarialLossMSE(reduction=reduction) if is_mse else AdversarialLossCE(reduction=reduction)

    def forward(
        self,
        a_real_pred,
        a_fake_pred,
        b_real_pred,
        b_fake_pred,
    ):
        loss_a = self.adversarial_loss_func(a_real_pred, a_fake_pred)
        loss_b = self.adversarial_loss_func(b_real_pred, b_fake_pred)
        return (loss_a + loss_b) / 2.0


class FullGeneratorLoss(nn.Module):
    def __init__(self, lambda_cyc=10., lambda_idt=0., is_mse=True, reduction='mean'):
        super(FullGeneratorLoss, self).__init__()
        self.adversarial_loss_func = AdversarialLossMSE(reduction=reduction) if is_mse else AdversarialLossCE(reduction=reduction)
        self.cycle_loss_func = CycleConsistencyLoss(reduction=reduction)
        self.identity_loss_func = nn.L1Loss(reduction=reduction)
        self.lambda_cyc = lambda_cyc
        self.lambda_idt = lambda_idt

    def forward(
        self,
        imgs_a,
        imgs_b,
        a_fake_pred,
        b_fake_pred,
        rec_a,
        rec_b,
        idt_a=None,
        idt_b=None,
    ):
        loss_gan_a2b = self.adversarial_loss_func(b_fake_pred)
        loss_gan_b2a = self.adversarial_loss_func(a_fake_pred)

        loss_cyc_a = self.cycle_loss_func(imgs_a, rec_a)
        loss_cyc_b = self.cycle_loss_func(imgs_b, rec_b)
        loss_cyc = loss_cyc_a + loss_cyc_b

        total = loss_gan_a2b + loss_gan_b2a + self.lambda_cyc * loss_cyc

        # identity loss: G_ba(a) ≈ a, G_ab(b) ≈ b
        if self.lambda_idt > 0 and idt_a is not None and idt_b is not None:
            loss_idt = self.identity_loss_func(idt_a, imgs_a) + self.identity_loss_func(idt_b, imgs_b)
            total = total + self.lambda_idt * loss_idt

        return total
