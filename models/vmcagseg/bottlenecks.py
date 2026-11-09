import torch.nn as nn

import torch
import torch.nn.functional as F
from einops import reduce
from . import attentions as AM


def get_neck(ver):
    _make_neck = dict(
        pcsa=PCAPSANeck,
        pcsa2=PCSANeck,
        spm=SPMNeck,
        mpm=MPMNeck,
        aspp=ASPP,
        no=None,
    ).get(ver, None)

    return _make_neck

class PSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.prob = nn.Softmax2d()

    def forward(self, x):
        # Compute the spatial mean of the input
        s = reduce(x, 'b c w h -> b w h', 'mean')

        # Apply pointwise convolution
        x = self.pw(x)

        # Compute the spatial mean after convolution
        s_ = reduce(x, 'b c w h -> b w h', 'mean')

        # Compute the attention scores
        raise_sp = self.prob(s_ - s)
        att_score = torch.sigmoid(s_ * (1 + raise_sp))

        # Ensure dimensions match correctly for multiplication
        att_score = att_score.unsqueeze(1)  # Shape [batch_size, 1, height, width]
        return x * att_score  # Broadcasting to match the dimensions


# Priority Channel Attention (PCA)
class PCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=dim, padding="same")
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        # Compute the channel-wise mean of the input
        c = reduce(x, 'b c w h -> b c', 'mean')

        # Apply depthwise convolution
        x = self.dw(x)

        # Compute the channel-wise mean after convolution
        c_ = reduce(x, 'b c w h -> b c', 'mean')

        # Compute the attention scores
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))

        # Ensure dimensions match correctly for multiplication
        att_score = att_score.unsqueeze(2).unsqueeze(3)  # Shape [batch_size, channels, 1, 1]
        return x * att_score  # Broadcasting to match the dimensions

# The structure of BottleNeck in U-shape model, named IPCA.
class PCAPSANeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        assert in_channels == out_channels
        self.pca = PCA(in_channels)
        self.psa = PSA(in_channels)
        #self.res= Residual()
    def forward(self, x):
        # Apply PCA with residual connection
        pca_out=self.pca(x)

        # Apply PSA
        psa_out = self.psa(pca_out)

        return psa_out


class PCSANeck(nn.ModuleList):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super().__init__()
        inter_channels = in_channels
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                norm_layer(inter_channels),
                nn.ReLU(True)
        )
        
        self.pca = PCA(in_channels)
        self.psa = PSA(in_channels)

        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                norm_layer(inter_channels // 2),
                nn.ReLU(True),
                nn.Dropout2d(0.1, False),
                nn.Conv2d(inter_channels // 2, out_channels, 1))
        # self.ca = AM.CoordinateAttention(channel=out_channels)

    def forward(self, x):
        x = self.trans_layer(x)
        x = self.pca(x)
        x = self.psa(x)
        x = self.score_layer(x)
        return x


class StripPooling(nn.Module):
    """
    Reference:
    """
    def __init__(self, in_channels, pool_size, norm_layer, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super(StripPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                norm_layer(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                norm_layer(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                norm_layer(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                norm_layer(in_channels))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), **self._up_kwargs)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), **self._up_kwargs)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), **self._up_kwargs)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), **self._up_kwargs)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)


class SPMNeck(nn.ModuleList):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super().__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                norm_layer(inter_channels),
                nn.ReLU(True)
        )
        self.strip_pool1 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.strip_pool2 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                norm_layer(inter_channels // 2),
                nn.ReLU(True),
                nn.Dropout2d(0.1, False),
                nn.Conv2d(inter_channels // 2, out_channels, 1))
        self.sea = AM.SEAttention(channel=out_channels)

    def forward(self, x):
        x = self.trans_layer(x)
        x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.score_layer(x)
        x = self.sea(x)
        return x


class MPMNeck(nn.ModuleList):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, up_kwargs={'mode': 'bilinear', 'align_corners': True}):
        super().__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                norm_layer(inter_channels),
                nn.ReLU(True)
        )
        # self.strip_pool1 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.strip_pool2 = StripPooling(inter_channels, (20, 12), norm_layer, up_kwargs)
        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                norm_layer(inter_channels // 2),
                nn.ReLU(True),
                nn.Dropout2d(0.1, False),
                nn.Conv2d(inter_channels // 2, out_channels, 1))
        self.ca = AM.CoordinateAttention(channel=out_channels)

    def forward(self, x):
        x = self.trans_layer(x)
        # x = self.strip_pool1(x)
        x = self.strip_pool2(x)
        x = self.score_layer(x)
        x = self.ca(x)
        return x


# 空洞卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

# 池化 -> 1*1 卷积 -> 上采样
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # 自适应均值池化
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        # 上采样
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  

# 整个 ASPP 架构
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        print(f"ASPP in:{in_channels} rates:{atrous_rates} out:{out_channels}")
        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        # 多尺度空洞卷积
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # 池化
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        
        # 拼接后的卷积
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)