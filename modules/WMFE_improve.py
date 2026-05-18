import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.Addmoudles.PPA import *
from ultralytics.nn.Addmoudles.WTConv import *
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class LightweightAttention(nn.Module):
    """轻量级通道注意力机制，减少计算量的同时提高特征表达能力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class OptimizedDWConv(nn.Module):
    """优化的深度可分离卷积，使用更小的卷积核和分组操作"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                                  padding=kernel_size//2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(self.bn(x))


class LightweightInceptionDWConv2d(nn.Module):
    """轻量化的Inception风格深度可分离卷积，减少参数量和计算量"""
    def __init__(self, in_channels, out_channels, branch_ratio=0.1):
        super().__init__()
        # 降低分支比例，减少参数量
        gc = max(1, int(in_channels * branch_ratio))
        
        # 使用更小的卷积核
        self.dwconv_hw = DSCWTConv(gc, gc, kernel_size=3)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, 5), padding=(0, 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(5, 1), padding=(2, 0), groups=gc)
        
        # 调整分割索引，减少分支数量
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
        # 最后使用轻量级卷积
        self.conv = OptimizedDWConv(in_channels, out_channels, kernel_size=3)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        x = torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )
        return self.conv(x)


class IDC(nn.Module):
    """轻量化的Bottleneck_IDC模块"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        # 使用1x1卷积进行降维
        self.cv1 = Conv(c1, c_, 1, 1)
        # 使用轻量化的InceptionDWConv2d
        self.cv2 = LightweightInceptionDWConv2d(c_, c2)
        # 添加轻量级注意力机制
        self.attention = LightweightAttention(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        residual = x
        x = self.cv2(self.cv1(x))
        x = self.attention(x)
        return x + residual if self.add else x


class C2f_Improved(nn.Module):
    """优化的C2f模块，减少计算量"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(IDC(self.c, self.c, shortcut, g) for _ in range(n))
        self.att=PPA(c2,c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.att(self.cv2(torch.cat(y, 1)))


class C3k2_IDC_Improved(C2f_Improved):
    """轻量化的C3k2_IDC模块，用于水下垃圾检测"""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.25, g=1, shortcut=True, use_attention=True):
        """初始化轻量化的C3k2_IDC模块
        参数:
            c1: 输入通道数
            c2: 输出通道数
            n: Bottleneck重复次数
            c3k: 是否使用C3k模块
            e: 扩展比例，降低e可以进一步减少参数量
            g: 分组卷积的组数
            shortcut: 是否使用残差连接
            use_attention: 是否使用注意力机制
        """
        # 降低默认的扩展比例，进一步减少参数量
        super().__init__(c1, c2, n, shortcut, g, e)
        # 根据需要选择使用C3k或EfficientBottleneck_IDC
        self.m = nn.ModuleList(
            self._create_middle_block(self.c, shortcut, g) for _ in range(n)
        )
        # 根据需要添加注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.attention = LightweightAttention(c2)

    def _create_middle_block(self, c, shortcut, g):
        """创建中间块，根据需求选择合适的模块"""
        # 直接使用EfficientBottleneck_IDC，更轻量
        return IDC(c, c, shortcut, g)

    def forward(self, x):
        x = super().forward(x)
        if self.use_attention:
            x = self.attention(x)
        return x


class WMFE_Light(nn.Module):
    """轻量化的小波特征增强模块，专为水下垃圾检测设计"""
    def __init__(self, c1, c2, n=1, reduction=4):
        super().__init__()
        # 使用1x1卷积进行通道调整
        self.conv1 = Conv(c1, c2, 1, 1)
        # 使用轻量化的C3k2_IDC_Improved模块
        self.backbone = C3k2_IDC_Improved(c2, c2, n=n, e=0.25)  # 降低扩展比例到0.25，进一步减少参数量
        # 添加全局注意力机制
        self.global_attention = LightweightAttention(c2, reduction)

    def forward(self, x):
        # 通道调整
        x = self.conv1(x)
        # 特征提取
        x = self.backbone(x)
        # 全局注意力增强
        x = self.global_attention(x)
        return x


# 用于不同规模的轻量级模块变体
class C3k2_IDC_Small(C3k2_IDC_Improved):
    """超轻量级版本，用于资源受限场景"""
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__(c1, c2, n=n, e=0.25, shortcut=shortcut, use_attention=False)


class C3k2_IDC_Medium(C3k2_IDC_Improved):
    """中等轻量化版本，平衡精度和速度"""
    def __init__(self, c1, c2, n=2, shortcut=True):
        super().__init__(c1, c2, n=n, e=0.375, shortcut=shortcut, use_attention=True)


class C3k2_IDC_Large(C3k2_IDC_Improved):
    """轻度轻量化版本，优先保证精度"""
    def __init__(self, c1, c2, n=3, shortcut=True):
        super().__init__(c1, c2, n=n, e=0.5, shortcut=shortcut, use_attention=True)