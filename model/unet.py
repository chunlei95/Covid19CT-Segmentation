import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional


class UNet(nn.Module):
    """U-Net模型的pytorch实现。
    论文地址：https://arxiv.org/abs/1505.04597
    模型的总体结构: 编码器 -> 一个ConvBlock -> 解码器 -> 一个Conv 1 * 1
    """

    def __init__(self):
        super(UNet, self).__init__()
        # 编码器部分
        self.eb1 = EncoderBlock(1, 64, 64, kernel_size=2)
        self.eb2 = EncoderBlock(64, 128, 128, kernel_size=2)
        self.eb3 = EncoderBlock(128, 256, 256, kernel_size=2)
        self.eb4 = EncoderBlock(256, 512, 512, kernel_size=2)
        # 编码器与解码器之间有一个ConvBlock
        self.cb = ConvBlock(512, 1024, 1024)
        # 解码器部分
        self.db1 = DecoderBlock(1024, 512, 512)
        self.db2 = DecoderBlock(512, 512, 256)
        self.db3 = DecoderBlock(256, 128, 128)
        self.db4 = DecoderBlock(128, 64, 64)
        # 一个Conv 1 * 1, 二分类，结果为两个通道
        self.conv1x1 = nn.Conv2d(64, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ex1, skip_x1 = self.eb1(x)
        ex2, skip_x2 = self.eb2(ex1)
        ex3, skip_x3 = self.eb3(ex2)
        ex4, skip_x4 = self.eb4(ex3)
        cbx = self.cb(ex4)
        dx1 = self.db1(cbx, skip_x4)
        dx2 = self.db2(dx1, skip_x3)
        dx3 = self.db3(dx2, skip_x2)
        dx4 = self.db4(dx3, skip_x1)
        crop = transforms.CenterCrop(size=(x.shape[-1], x.shape[-2]))
        # normalize = transforms.Normalize((0.5,), (0.5,))
        return self.sigmoid(self.conv1x1(crop(dx4)))
        # return self.conv1x1(crop(dx4))


class ConvBlock(nn.Module):
    """一个Conv2d卷积后跟一个Relu激活函数，卷积核大小为3 * 3

    :param in_channels: 层次块的输入通道数
    :param mid_channels: 层次块中间一层卷积的通道数
    :param out_channels: 层次块输出层的通道数
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(ConvBlock, self).__init__()
        conv_relu_list = [nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=2),
                          nn.BatchNorm2d(mid_channels),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=2),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(inplace=True)]
        self.conv_relu = nn.Sequential(*conv_relu_list)

    def forward(self, x):
        return self.conv_relu(x)


class DownSampling(nn.Module):
    """下采样，使用max pool方法执行，核大小为 2 * 2，用在编码器的ConvBlock后面

    :param kernel_size: 下采样层（即最大池化层）的核大小
    """

    def __init__(self, kernel_size):
        super(DownSampling, self).__init__()
        self.down_sample = nn.MaxPool2d(kernel_size=kernel_size)

    def forward(self, x):
        return self.down_sample(x)


class UpSampling(nn.Module):
    """上采样，用在解码器的ConvBlock前面，使用转置卷积，同时通道数减半，

    C_out = out_channels
    H_out = (H_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    W_out = (W_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    :param in_channels: 转置卷积的输入通道数
    :param out_channels: 转置卷积的输出通道数
    :param kernel_size: 转置卷积的卷积核大小，默认为2
    :param stride: 转置卷积的步幅，默认为2
    """

    def __init__(self, in_channels, out_channels, kernel_size=7, stride=2, dilation=1, padding=0, output_padding=1):
        super(UpSampling, self).__init__()
        # self.up_sample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        # stride=2, kernel_size=2相当于宽高翻倍
        self.up_sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilation=dilation, padding=padding, output_padding=output_padding)

    def forward(self, x):
        return self.up_sample(x)


class EncoderBlock(nn.Module):
    """编码器中的一个层次块

    :param in_channels: 层次块的输入通道数
    :param mid_channels: 层次块中间一层卷积的通道数
    :param out_channels: 层次块输出层的通道数
    :param kernel_size: 下采样层（即最大池化层）的核大小
    """

    def __init__(self, in_channels, mid_channels, out_channels, kernel_size):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock(in_channels, mid_channels, out_channels)
        self.down_sample = DownSampling(kernel_size)

    def forward(self, x):
        x1 = self.conv_block(x)
        return self.down_sample(x1), x1


class ConcatLayer(nn.Module):
    """跳跃连接，在通道维上连接

    """

    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, x, skip_x):
        # 将从编码器传过来的特征图裁剪到与输入相同尺寸
        x1 = functional.center_crop(skip_x, [x.shape[-2], x.shape[-1]])
        # crop = transforms.RandomCrop(x.shape)
        # x1 = crop(skip_x).unsqueeze(dim=0)
        # x1 = x.unsqueeze(dim=0)
        # F.grid_sample()
        # x2 = x.unsqueeze(dim=0)
        if x1.shape != x.shape:
            raise Exception('要连接的两个特征图尺寸不一致，skip_x.shape={}，x.shape={}'.format(skip_x.shape, x.shape))
        # 通道维连接
        return torch.cat([x, x1], dim=1)


class DecoderBlock(nn.Module):
    """解码器中的层次块，每个层次块都是UpSampling -> Concat -> ConvBlock

    :param in_channels: 层次块的输入通道数
    :param mid_channels: 层次块中间一层卷积的通道数
    :param out_channels: 层次块输出层的通道数
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up_sample = UpSampling(in_channels, out_channels)
        self.conv_block = ConvBlock(in_channels, mid_channels, out_channels)

    def forward(self, x, skip_x):
        x1 = self.up_sample(x)
        concat = ConcatLayer()
        x2 = concat(x1, skip_x)
        return self.conv_block(x2)
