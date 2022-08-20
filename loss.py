import torch.nn as nn
import torch
from evaluate import dice_coef, IoU
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice相似系数损失函数，训练过程可能出现不稳定的情况

    """

    def __init__(self, ep=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = ep

    def forward(self, predict, target):
        score = dice_coef(predict, target, self.epsilon)
        return 1 - score


# class DiceLoss(nn.Module):
#
#     def __init__(self, alpha=0.5, beta=0.5, size_average=True, reduce=True):
#         super(DiceLoss, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#
#         self.size_average = size_average
#         self.reduce = reduce
#
#     def forward(self, preds, targets):
#         # 样本数
#         N = preds.size(0)
#         # 通道数
#         C = preds.size(1)
#
#         # 得到预测的概率值
#         # P = F.softmax(preds, dim=1)
#         P = preds
#         # 定义平滑系数，值设置为1e-5
#         smooth = torch.zeros(C, dtype=torch.float32).fill_(0.00001)
#
#         class_mask = torch.zeros(preds.shape).to(preds.device)
#         class_mask.scatter_(1, targets.to(torch.int64), 1.)
#
#         ones = torch.ones(preds.shape).to(preds.device)
#         P_ = ones - P
#         class_mask_ = ones - class_mask
#
#         TP = P * class_mask
#         FP = P * class_mask_
#         FN = P_ * class_mask
#
#         smooth = smooth.to(preds.device)
#         self.alpha = FP.transpose(0, 1).reshape(C, -1).sum(dim=(1)) / ((FP.transpose(0, 1).reshape(C, -1).sum(
#             dim=(1)) + FN.transpose(0, 1).reshape(C, -1).sum(dim=(1))) + smooth)
#
#         self.alpha = torch.clamp(self.alpha, min=0.2, max=0.8)
#         # print('alpha:', self.alpha)
#         self.beta = 1 - self.alpha
#         num = torch.sum(TP.transpose(0, 1).reshape(C, -1), dim=(1)).float()
#         den = num + self.alpha * torch.sum(FP.transpose(0, 1).reshape(C, -1), dim=(1)).float() + self.beta * torch.sum(
#             FN.transpose(0, 1).reshape(C, -1), dim=(1)).float()
#
#         dice = num / (den + smooth)
#
#         if not self.reduce:
#             loss = torch.ones(C).to(dice.device) - dice
#             return loss
#
#         loss = 1 - dice
#         loss = loss.sum()
#
#         # 控制是否需要对结果求平均，多分类需要对结果除以通道数求平均
#         if self.size_average:
#             loss /= C
#
#         return loss


class IoULoss(nn.Module):
    """IoU损失函数

    """

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, predict, target):
        return IoU(predict, target)
