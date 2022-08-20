import torch
import numpy as np
import sklearn.metrics as metrics


def confusion_matrix(predict, target):
    return metrics.confusion_matrix(target, predict)


def jaccard_score(predict, target):
    """计算jaccard相似系数

    :param predict:
    :param target:
    :return:
    """
    return metrics.jaccard_score(target, predict)


def dice_coef(predict, target, ep=1e-8):
    """dice相似系数（值域为[0,1]），也称为重叠指数，表示两个物体相交的面积占总面积的比值
    用于相似性评估

    :param ep: 平滑系数
    :param predict: 预测值(类别，非概率)
    :param target: 目标值
    :return: 相似系数
    """
    num = predict.size(0)
    pre = predict.view(num, -1)
    tar = target.view(num, -1)
    intersection = (pre * tar).sum()
    return (2. * intersection + ep) / (pre.sum() + tar.sum() + ep)


def IoU(predict, target):
    """计算IoU

    :param predict: 预测值
    :param target: 真实值
    :return: IoU值
    """
    # todo 完成IoU指标的计算
    pass


def mIoU(predict, target):
    pass


def recall(predict, target):
    """敏感度，即召回率
    recall/sensitivity = TP / (TP + FN)

    :param predict: 预测值
    :param target: 真实值
    :return: 敏感度
    """
    predict, target = _process(predict, target)
    tp = np.count_nonzero(predict & target)
    fn = np.count_nonzero(~predict & target)  # ~表示取反
    try:
        rec = tp / float(tp + fn)
    except ZeroDivisionError:
        rec = 0.0
    return rec


def precision(predict, target):
    """计算精度
    precision = TP / (TP + FP)

    :param predict: 预测值
    :param target: 真实值
    :return: 精度值
    """
    predict, target = _process(predict, target)
    tp = np.count_nonzero(predict & target)
    fp = np.count_nonzero(predict & ~target)
    try:
        pr = tp / (tp + fp)
    except ZeroDivisionError:
        pr = 0.0
    return pr


def specificity(predict, target):
    """计算特异性
    specificity = TN / (TN + FP)

    :param predict:
    :param target:
    :return:
    """
    predict, target = _process(predict, target)
    tn = np.count_nonzero(~predict & ~target)
    fp = np.count_nonzero(predict & ~target)
    try:
        spec = tn / float(tn + fp)
    except ZeroDivisionError:
        spec = 0.0
    return spec


def _process(predict, target):
    if torch.is_tensor(predict):
        predict = torch.sigmoid(predict).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    predict = np.atleast_1d(predict.astype(np.bool))
    target = np.atleast_1d(target.astype(np.bool))
    return predict, target
