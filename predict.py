import torch

from model.unet import UNet
from utils import load_dataset
import transforms
from ignite import metrics
from ignite.utils import to_onehot
from evaluate import dice_coef


def predict():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = UNet()
    pretrained_params = torch.load('pretrained/v1/best.pth', map_location=device)
    net.load_state_dict(pretrained_params['best_model_state_dict'])
    net.eval()

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    cm = metrics.ConfusionMatrix(num_classes=2, device=device)
    precision = metrics.Precision()
    recall = metrics.Recall()
    dice_coefficient = metrics.DiceCoefficient(cm, ignore_index=0)
    test_loader = load_dataset(dataset_select='A', batch_size=16, train=False, test_transforms=test_transforms)
    for index, x, y in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device).to(torch.float32)
        predict_mask = net(x)
        predict_mask = to_onehot(predict_mask, num_classes=2)
        cm.update((predict_mask, y))
        precision.updata((predict_mask, y))
        recall.update((predict_mask, y))
        dice_coefficient.update((predict_mask, y))

    print('precision: {}'.format(precision.compute()))
    print('recall: {}'.format(recall.compute()))
    print('confusion matrix: {}'.format(cm.compute()))
    print('dice coefficient: {}'.format(dice_coefficient.compute()))


