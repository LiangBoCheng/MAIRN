import torch
import torch.nn as nn
import torch.nn.functional as F

from .Swin import SwinTransformer

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


def US2(x):
    """if size!=None:
        return F.interpolate(x, size=size, mode='bilinear')
    else:"""
    return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class MSAR(nn.Module):
    def __init__(self, channel):
        super(MSAR, self).__init__()

        self.conv_f1 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f2 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f3 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f4 = nn.Sequential(
            BasicConv2d(3 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.conv_f5 = nn.Sequential(
            BasicConv2d(3 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f6 = nn.Sequential(
            BasicConv2d(3 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.msa1 = MSA(channel)
        self.msa2 = MSA(channel)
        self.msa3 = MSA(channel)
        self.msa4 = MSA(channel)

    def forward(self, x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4):
        a1 = self.msa1(x_s1)
        a2 = self.msa2(x_s2)
        a3 = self.msa3(x_s3)
        a4 = self.msa4(x_s4)
        e4 = x_e4
        e3 = torch.cat((x_e3, e4), 1)
        e2 = torch.cat((x_e2, US2(self.conv_f1(e3))), 1)
        e1 = torch.cat((x_e1, US2(self.conv_f2(e2))), 1)
        o4 = self.conv_f3(torch.cat((e4, a4), 1)) + a4
        o3 = self.conv_f4(torch.cat((e3, a3), 1)) + a3
        o2 = self.conv_f5(torch.cat((e2, a2), 1)) + a2
        o1 = self.conv_f6(torch.cat((e1, a1), 1)) + a1

        return o1, o2, o3, o4, a1, a2, a3, a4


class ER(nn.Module):
    def __init__(self, channel):
        super(ER, self).__init__()

        self.conv_f1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.conv_f5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f6 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self, x_a1, x_a2, x_a3, x_a4, x_e1, x_e2, x_e3, x_e4):
        a4 = x_a4
        a3 = x_a3 * a4
        a2 = x_a2 * US2(self.conv_f1(a3))
        a1 = x_a1 * US2(self.conv_f2(a2))
        e4 = self.conv_f3(x_e4 * a4) + x_e4
        e3 = self.conv_f4(x_e3 * a3) + x_e3
        e2 = self.conv_f5(x_e2 * a2) + x_e2
        e1 = self.conv_f6(x_e1 * a1) + x_e1

        return e1, e2, e3, e4


class PFC(nn.Module):
    def __init__(self, channel):
        super(PFC, self).__init__()

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat4 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat5 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat6 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x1, x2, x3, x4):
        xa3 = x3 + self.conv_cat1(torch.cat((x3, x4), 1))
        xa2 = x2 + self.conv_cat2(torch.cat((x2, US2(x3)), 1))
        xa1 = x1 + self.conv_cat3(torch.cat((x1, US2(x2)), 1))

        xb2 = xa2 + self.conv_cat4(torch.cat((xa2, US2(xa3)), 1))
        xb1 = xa1 + self.conv_cat5(torch.cat((xa1, US2(xa2)), 1))

        xc1 = xb1 + self.conv_cat6(torch.cat((xb1, US2(xb2)), 1))

        x = self.output(xc1)
        return x

class MSA(nn.Module):
    def __init__(self, channel):
        super(MSA, self).__init__()
        self.branches0 = BasicConv2d(channel, channel, 3, padding=2, dilation=2)
        self.branches1 = BasicConv2d(channel, channel, 3, padding=4, dilation=4)
        self.branches2 = BasicConv2d(channel, channel, 3, padding=6, dilation=6)
        self.branches3 = BasicConv2d(channel, channel, 3, padding=8, dilation=8)
        # self.branches4 = BasicConv2d(channel, channel, 3, padding=8, dilation=8)

        self.ca1 = ChannelAttention(channel)
        self.ca2 = ChannelAttention(channel)
        self.ca3 = ChannelAttention(channel)
        self.ca4 = ChannelAttention(channel)
        # self.ca5 = ChannelAttention(channel)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        # self.sa5 = SpatialAttention()
        self.conv1_1 = BasicConv2d(channel, channel, 1)
        self.conv3_3 = BasicConv2d(channel, channel, 3, padding=1)

    def forward(self, x):
        x0 = self.conv1_1(x)
        F_11 = self.branches0(x0)
        F_ca = self.ca1(F_11) * F_11
        F0 = self.sa1(F_ca) * F_ca
        out1 = self.branches1(x0)
        out1 = self.ca2(out1) * out1
        F1 = (self.sa2(out1) + self.sa1(F_ca)) * out1
        out2 = self.branches2(x0)
        out2 = self.ca3(out2) * out2
        F2 = (self.sa3(out2) + self.sa2(out1) + self.sa1(F_ca)) * out2
        out3 = self.branches3(x0)
        out3 = self.ca4(out3) * out3
        F3 = (self.sa4(out3) + self.sa3(out2) + self.sa2(out1) + self.sa1(F_ca)) * out3
        gather = F0 + F1 + F2 + F3

        return self.conv3_3(gather) + x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MAIRN(nn.Module):

    def __init__(self, channel=32):
        super(MAIRN, self).__init__()
        # Backbone model
        self.swin = SwinTransformer(img_size=224, window_size=7)
        self.swin.load_state_dict(torch.load('./model/swin224.pth')['model'], strict=False)
        self.reduce_s1 = Reduction(128, channel)
        self.reduce_s2 = Reduction(256, channel)
        self.reduce_s3 = Reduction(512, channel)
        self.reduce_s4 = Reduction(512, channel)

        self.reduce_e1 = Reduction(128, channel)
        self.reduce_e2 = Reduction(256, channel)
        self.reduce_e3 = Reduction(512, channel)
        self.reduce_e4 = Reduction(512, channel)

        self.msar1 = MSAR(channel)
        self.msar2 = MSAR(channel)
        self.msar3 = MSAR(channel)
        self.msar4 = MSAR(channel)

        self.er1 = ER(channel)
        self.er2 = ER(channel)
        self.er3 = ER(channel)
        self.er4 = ER(channel)

        self.pfc_o = PFC(channel)
        self.pfc_e = PFC(channel)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        size = x.size()[2:]
        x1, x2, x3, x4 = self.swin(x)

        x_s1 = self.reduce_s1(x1)
        x_s2 = self.reduce_s2(x2)
        x_s3 = self.reduce_s3(x3)
        x_s4 = self.reduce_s4(x4)

        x_e1 = self.reduce_e1(x1)
        x_e2 = self.reduce_e2(x2)
        x_e3 = self.reduce_e3(x3)
        x_e4 = self.reduce_e4(x4)

        o1, o2, o3, o4, a1, a2, a3, a4 = self.msar1(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        e1, e2, e3, e4 = self.er1(a1, a2, a3, a4, x_e1, x_e2, x_e3, x_e4)

        o1, o2, o3, o4, a1, a2, a3, a4 = self.msar2(o1, o2, o3, o4, e1, e2, e3, e4)
        e1, e2, e3, e4 = self.er2(a1, a2, a3, a4, e1, e2, e3, e4)

        o1, o2, o3, o4, a1, a2, a3, a4 = self.msar3(o1, o2, o3, o4, e1, e2, e3, e4)
        e1, e2, e3, e4 = self.er3(a1, a2, a3, a4, e1, e2, e3, e4)

        o1, o2, o3, o4, a1, a2, a3, a4 = self.msar4(o1, o2, o3, o4, e1, e2, e3, e4)
        e1, e2, e3, e4 = self.er4(a1, a2, a3, a4, e1, e2, e3, e4)

        pred_o = self.pfc_o(o1, o2, o3, o4)
        pred_e = self.pfc_e(e1, e2, e3, e4)

        pred_s = F.interpolate(pred_o, size=size, mode='bilinear', align_corners=True)
        pred_e = F.interpolate(pred_e, size=size, mode='bilinear', align_corners=True)

        return pred_s, self.sigmoid(pred_s), pred_e