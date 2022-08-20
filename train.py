import torch
import torch.nn as nn
import transforms
from loss import DiceLoss
from model.unet import UNet
import torch.optim as optim
import argparse
from utils import load_dataset

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args_parser.add_argument('--te', type=int, default=150, help='total epoch')
args_parser.add_argument('--ce', type=int, default=0, help='current epoch')
args_parser.add_argument('--bs', type=int, default=32, help='batch size')
args = args_parser.parse_args()


def train(train_loader, val_loader, net, optimizer, total_epoch, current_epoch=0, device='cpu'):
    bce_loss = nn.BCELoss()
    dice_loss = DiceLoss()
    save_best = {}
    save_last = {}
    loss_history = {}
    train_losses = []
    val_losses = []
    search_best = SearchBest()
    for i in range(current_epoch, total_epoch):
        if not net.training:
            net.train()
        total_loss = 0.0
        for index, (image, mask) in enumerate(train_loader):
            image = image.to(device)
            mask = mask.to(device)
            predict_mask = net(image)

            loss = bce_loss(predict_mask, mask) + dice_loss(predict_mask, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print('Epoch {}: Batch {}/{} train loss = {:.4f}'.format(i + 1, index + 1, len(train_loader), loss.item))
        val_loss = valid(val_loader, net, device)
        train_losses.append(total_loss / len(train_loader))
        val_losses.append(val_loss)
        search_best(val_loss)
        if search_best.counter == 0:
            save_best['best_model_state_dict'] = net.state_dict()
            save_best['best_epoch'] = i + 1
    save_last['last_model_state_dict'] = net.state_dict()
    save_last['last_optimizer_state_dict'] = optimizer.state_dict()
    save_last['last_epoch'] = total_epoch
    loss_history['train_loss_history'] = train_losses
    loss_history['val_loss_history'] = val_losses
    torch.save(save_last, './pretrained/last.pth')
    torch.save(save_best, './pretrained/best.pth')
    torch.save(loss_history, './pretrained/loss_history.pth')


class SearchBest(object):
    def __init__(self, min_delta=0):
        super(SearchBest, self).__init__()
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best is None:
            self.best = val_loss
        if self.best - val_loss > self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print('performance reducing: {}'.format(self.counter))


def valid(val_loader, net, device):
    bce_loss = nn.BCELoss()
    dice_loss = DiceLoss()
    total_loss = 0.0
    for image, mask in val_loader:
        image = image.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            predict_mask = net(image)
            loss = bce_loss(predict_mask, mask) + dice_loss(predict_mask, mask)
            total_loss += loss.item()
    return total_loss / len(val_loader)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=(0, 180)),
        transforms.RandomCrop(256),
        transforms.GrayScale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    val_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    net = UNet()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))
    train_loader, val_loader = load_dataset('B')
    train(train_loader, val_loader, net, optimizer, total_epoch=args.te, current_epoch=args.ce, device=device)
