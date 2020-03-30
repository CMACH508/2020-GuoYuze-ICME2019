import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_btn(nn.Module):

    def __init__(self, param, activate_fn=True):
        super(conv_btn, self).__init__()
        self.param = param
        self.bn = nn.BatchNorm2d(num_features=self.param.size()[0])
        self.activate_fn = activate_fn
        self.weight = nn.Parameter(self.param)

    def forward(self, input):
        output = F.conv2d(input, self.weight, stride=2, padding=1)
        output = self.bn(output)

        if self.activate_fn:
            output = F.relu(output)

        return output


class conv(nn.Module):

    def __init__(self, nums_in, nums_out, activate_fn=True):
        super(conv, self).__init__()
        self.conv1 = nn.Conv2d(nums_in, nums_out, kernel_size=[3, 3], stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(nums_out)
        self.conv2 = nn.Conv2d(nums_out, nums_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(nums_out)
        self.activate_fn = activate_fn

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        if self.activate_fn:
            output = F.relu(output)

        output = self.conv2(output)
        output = self.bn2(output)
        if self.activate_fn:
            output = F.relu(output)

        return output


class deconv(nn.Module):

    def __init__(self, nums_in, nums_out, activate_fn=True):
        super(deconv, self).__init__()

        self.deconv = nn.ConvTranspose2d(nums_in, nums_out, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(nums_out)
        self.activate_fn = activate_fn

    def forward(self, input):
        output = self.deconv(input)
        output = self.bn(output)
        if self.activate_fn:
            output = F.relu(output)

        return output


class deconv_btn(nn.Module):

    def __init__(self, param, activate_fn=True):
        super(deconv_btn, self).__init__()
        self.param = param
        self.bn = nn.BatchNorm2d(num_features=self.param.size()[1])
        self.activate_fn = activate_fn
        self.weight = nn.Parameter(self.param)

    def forward(self, input):
        output = F.conv_transpose2d(input, self.weight, stride=2, padding=1, output_padding=1)
        output = self.bn(output)

        if self.activate_fn:
            output = F.relu(output)

        return output

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels)
        )
    def forward(self, input):
        output = F.relu(input + self.resblock(input))
        return output

class AttenLayer(nn.Module):

    def __init__(self, channel, reduction):
        super(AttenLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0),
            nn.Sigmoid()
            # nn.Softmax(dim=1)
        )


    def forward(self, input):
        output = self.avg_pool(input)
        output = self.conv(output)
        return input * output

class fake_atten_layer(nn.Module):
    def __init__(self):
        super(fake_atten_layer, self).__init__()
    def forward(self, input):
        return input

