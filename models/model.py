import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Activations ────────────────────────────────────────────────────────────────
class HSigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6

class HSwish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        return x * (F.relu6(x + 3, inplace=self.inplace) / 6)

# ─── Squeeze-Excite ─────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, in_ch: int, reduction: int = 4):
        super().__init__()
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc1   = nn.Conv2d(in_ch, in_ch // reduction,    1, bias=True)
        self.relu  = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv2d(in_ch // reduction, in_ch,    1, bias=True)
        self.hsig  = HSigmoid()
    def forward(self, x):
        s = self.pool(x)
        s = self.relu(self.fc1(s))
        s = self.hsig(self.fc2(s))
        return x * s

# ─── Inverted Residual Block ────────────────────────────────────────────────────
class InvertedResidual(nn.Module):
    def __init__(self, inp, exp, oup, k, s, use_se, nl):
        super().__init__()
        self.use_res = (s == 1 and inp == oup)
        Activation = HSwish if nl == 'HS' else nn.ReLU
        act_kwargs = {'inplace': True}

        # 1×1 expand
        self.expand = nn.Sequential(
            nn.Conv2d(inp, exp,    1, bias=False),
            nn.BatchNorm2d(exp),
            Activation(**act_kwargs)
        ) if exp != inp else nn.Identity()

        # depthwise conv
        self.dw = nn.Sequential(
            nn.Conv2d(exp, exp, k, padding=k//2, stride=s, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            Activation(**act_kwargs),
            SEBlock(exp) if use_se else nn.Identity()
        )

        # 1×1 project
        self.project = nn.Sequential(
            nn.Conv2d(exp, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        y = self.expand(x)
        y = self.dw(y)
        y = self.project(y)
        return x + y if self.use_res else y

# ─── MobileNetV3-Large Backbone ─────────────────────────────────────────────────
class MobileNetV3LargeBackbone(nn.Module):
    def __init__(self, width_mult: float = 1):
        super().__init__()
        # config: k, exp, out, use_se, nl, s
        cfgs = [
            # stage1
            (3,  16,  16, False, 'RE', 1),
            # stage2
            (3,  64,  24, False, 'RE', 2),
            (3,  72,  24, False, 'RE', 1),
            # stage3
            (5,  72,  40, True,  'RE', 2),
            (5, 120,  40, True,  'RE', 1),
            (5, 120,  40, True,  'RE', 1),
            # stage4
            (3, 240,  80, False, 'HS', 2),
            (3, 200,  80, False, 'HS', 1),
            (3, 184,  80, False, 'HS', 1),
            (3, 184,  80, False, 'HS', 1),
            (3, 480, 112, True,  'HS', 1),
            (3, 672, 112, True,  'HS', 1),
            (5, 672, 160, True,  'HS', 2),
            (5, 960, 160, True,  'HS', 1),
            (5, 960, 160, True,  'HS', 1),
        ]
        input_channel = 16
        self.stem = nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            HSwish()
        )
        # build inverted residual blocks
        layers = []
        for k, exp, out, se, nl, s in cfgs:
            exp_ch  = int(exp * width_mult)
            out_ch  = int(out * width_mult)
            layers.append(InvertedResidual(input_channel, exp_ch, out_ch, k, s, se, nl))
            input_channel = out_ch
        self.blocks = nn.Sequential(*layers)
        # final conv after blocks
        final_exp = int(960 * width_mult)
        self.head = nn.Sequential(
            nn.Conv2d(input_channel, final_exp, 1, bias=False),
            nn.BatchNorm2d(final_exp),
            HSwish()
        )

    def forward(self, x):
        x = self.stem(x)        # 3×512×512 → 16×256×256
        x = self.blocks(x)      # down to C×H/32×W/32 (≈160 ch @ 16×16)
        x = self.head(x)        # → final_exp×H/32×W/32
        return x

# ─── Segmentation Head ──────────────────────────────────────────────────────────
class SegHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int = 19):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (N, in_ch, H/32, W/32)
        x = self.conv(x)                                 # → (N,19,H/32,W/32)
        return F.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
        # → (N,19,512,512)

# ─── Full Model ─────────────────────────────────────────────────────────────────
class MobileNetV3_Segmenter(nn.Module):
    def __init__(self, num_classes: int = 19, width_mult: float = 1.0):
        super().__init__()
        self.backbone = MobileNetV3LargeBackbone(width_mult)
        # backbone.head outputs final_exp channels
        final_exp = int(960 * width_mult)
        self.head     = SegHead(final_exp, num_classes)

    def forward(self, x):
        """
        x: (N,3,512,512) → out: (N,19,512,512)
        """
        feats = self.backbone(x)
        out   = self.head(feats)
        return out


model_v3 = MobileNetV3_Segmenter(num_classes=19)
print("Params (M):", sum(p.numel() for p in model_v3.parameters())/1e6)