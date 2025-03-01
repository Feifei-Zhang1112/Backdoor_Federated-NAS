import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
"""
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'fire_1': lambda C, stride, affine: Fire(C, C, 1/8, stride),
    'group_conv_3x3': lambda C, stride, affine: GroupedConv2d(C, C, 3, stride, padding=1),
    'depth_conv_5x5': lambda C, stride, affine: DepthwiseSeparableConv2d(C, C, 5, stride=stride, padding=2),
    'fire_2': lambda C, stride, affine: Fire(C, C, 7/8, stride),
    'ghost_conv_7x7': lambda C, stride, affine: GhostModule(C, C, 0.9, stride),
    'group_conv_7x7': lambda C, stride, affine: GroupedConv2d(C, C, 7, stride, padding=3),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3,0), bias=False),
        nn.BatchNorm2d(C, affine=affine)),
}
"""


"""
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'dil_conv_3x3_2': lambda C, stride, affine: DilConv(C, C, 3, stride, 1, 2, affine=affine),
    'dil_conv_5x5_1': lambda C, stride, affine: DilConv(C, C, 5, stride, 2, 1, affine=affine),
    'dil_conv_5x5_2': lambda C, stride, affine: DilConv(C, C, 5, stride, 2, 2, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, 1, affine=affine),
    'eca_3': lambda C, stride, affine: ECAModule(C, 3, stride),
    'eca_5': lambda C, stride, affine: ECAModule(C, 5, stride),
    'eca_7': lambda C, stride, affine: ECAModule(C, 7, stride),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}
"""
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else ShuffleNetUnit(C, C, 2, 16),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, 1, affine=affine),
    'eca_3': lambda C, stride, affine: ECAModule(C, 3, stride, C),
    'eca_5': lambda C, stride, affine: ECAModule(C, 5, stride, C),
    'eca_7': lambda C, stride, affine: ECAModule(C, 7, stride, C),
    'dil_conv_3x3_1': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_3x3_2': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'dil_conv_5x5_2': lambda C, stride, affine: DilConv(C, C, 5, stride, 2, 1, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}
"""
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'fire_1': lambda C, stride, affine: Fire(C, C, 1/8, stride),
    'group_conv_3x3': lambda C, stride, affine: GroupedConv2d(C, C, 3, stride, padding=1),
    'depth_conv_5x5': lambda C, stride, affine: DepthwiseSeparableConv2d(C, C, 5, stride=stride, padding=2),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'ghost_conv_7x7': lambda C, stride, affine: GhostModule(C, C, 0.9, stride),
    'senet': lambda C, stride, affine: SENet(C, C, stride, 1),
    'alter_connect': lambda C, stride, affine: Identity() if stride == 1 else ECAModule(C, 3, stride),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)),
}
"""

class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, kernel_size=3):
        super(ShuffleNetUnit, self).__init__()

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

        # Grouped convolutions reduce the number of parameters
        self.gconv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.gconv2 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

        self.gconv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # a 1x1 grouped conv with BN and activation
        out = F.relu(self.bn1(self.gconv1(x)))
        out = channel_shuffle(out, self.groups)
        out = F.relu(self.bn2(self.gconv2(out)))
        out = self.bn3(self.gconv3(out))

        if self.stride == 1:
            residual = self.residual_conv(x)
            out = F.relu(out + residual)

        return out
def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)

    return x
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * x.sigmoid()
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
class SENet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, reduction=16):
        super(SENet, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.seblock = SEBlock(out_channels, reduction=reduction)

    def forward(self, x):
        x = self.op(x)
        x = self.seblock(x)
        return x


class ECAModule(nn.Module):
    def __init__(self, channel, k_size, stride, fixed_out_channels=None, output_size=None):
        super(ECAModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.fixed_out = fixed_out_channels is not None
        self.output_size = output_size

        if self.fixed_out:
            self.channel_adjust = nn.Conv2d(channel, fixed_out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            """
            if self.output_size is not None:
                self.size_adjust = nn.Upsample(size=self.output_size, mode='bilinear', align_corners=True)
            else:
                if stride > 1:
                    self.size_adjust = nn.MaxPool2d(kernel_size=stride, stride=stride)
                else:
                    self.size_adjust = nn.Upsample(scale_factor=2, mode='nearest')
                    """

    def forward(self, x):
        # original_size = x.size()[2:]

        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)
        x = x * y.expand_as(x)

        if self.fixed_out:
            x = self.channel_adjust(x)
            """
            if self.output_size is not None or self.size_adjust is not None:
                x = self.size_adjust(x) """

        return x
class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(GroupedConv2d, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True)
        )

    def forward(self, x):
        return self.op(x)

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(DepthwiseSeparableConv2d, self).__init__()

        self.op = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        )

    def forward(self, x):
        return self.op(x)

class GhostModule(nn.Module):
    def __init__(self, in_channels, fixed_ghost_channels, ratio, stride=1, kernel_size=1, dw_size=3, relu=True):
        super(GhostModule, self).__init__()
        real_channels = int(fixed_ghost_channels * ratio)
        self.oup = fixed_ghost_channels
        init_channels = real_channels
        new_channels = init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.oup, :, :]


class Fire(nn.Module):
    def __init__(self, inplanes, out_channels, ratio, stride):
        super(Fire, self).__init__()
        expand1x1_planes = int(out_channels * ratio)
        expand3x3_planes = out_channels - expand1x1_planes
        squeeze_planes = int(inplanes * (4 / 3))

        self.squeeze = nn.Sequential(
            nn.Conv2d(inplanes, squeeze_planes, kernel_size=1, stride=stride),
            nn.ReLU(inplace=True)
        )
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1)


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)




class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )


    def forward(self, x):
        return self.op(x)

    def _conv2d_size_out(self, size, kernel_size, stride, padding, dilation):
        return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, group, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=group, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=group, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
