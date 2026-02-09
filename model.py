import torch
import torch.nn as nn
from gcn_layers import GraphConvolution
import torch.nn.functional as F
from torch.nn import init
import math
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class ThreeFeatureGating(nn.Module):     #通过门控机制（gating mechanism）融合三个输入特征张量   对应论文图5
    def __init__(self, input_dim):
        super(ThreeFeatureGating, self).__init__()
        hidden_dim = input_dim

        self.conv_gate1 = nn.Conv2d(in_channels=input_dim,
                                    out_channels=hidden_dim,
                                    kernel_size=1)
        self.conv_gate2 = nn.Conv2d(in_channels=input_dim,
                                    out_channels=hidden_dim,
                                    kernel_size=1)
        self.conv_gate3 = nn.Conv2d(in_channels=input_dim,
                                    out_channels=hidden_dim,
                                    kernel_size=1)

    def forward(self, input_tensor1, input_tensor2, input_tensor3):   #三个形状为 (batch, input_dim, height, width) 的特征张量
        gate1 = torch.sigmoid(self.conv_gate1(input_tensor1))
        gate2 = torch.sigmoid(self.conv_gate2(input_tensor2))
        gate3 = torch.sigmoid(self.conv_gate3(input_tensor3))


        fused_features = input_tensor1 * gate1 + input_tensor2 * gate2 + input_tensor3 * gate3

        return fused_features


class GCNConvBlock(nn.Module):         # 从 gcn_layers 里导入 GraphConvolution
    def __init__(self, ch_in, ch_out):
        super(GCNConvBlock, self).__init__()
        self.conv1 = GraphConvolution(ch_in, ch_out)
        self.conv2 = GraphConvolution(ch_out, ch_out)
        self.drop_prob = 0.3    # 用于在训练时进行随机失活，防止过拟合
        self.drop = nn.Dropout(self.drop_prob)
        self.act = nn.LeakyReLU()

    def forward(self, x, adj):     #前向传播过程  对应论文图3
        x = self.drop(self.act(self.conv1(x, adj)))
        x = self.drop(self.act(self.conv2(x, adj)))
        return x


class CNNConvBlock(nn.Module):       #CNNConvBlock, 1, 和 2 是高光谱图像处理的 CNN 模块，逐步增加卷积层数   1 2没有使用
    def __init__(self, ch_in, ch_out, k, h, w):
        super(CNNConvBlock, self).__init__()
        self.BN = nn.BatchNorm2d(ch_in)
        self.conv_in = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv_out = nn.Conv2d(ch_out, ch_out, kernel_size=k, padding=k//2, stride=1, groups=ch_out)
        self.pool = nn.AvgPool2d(3, padding=1, stride=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.BN(x)
        x = self.act(self.conv_in(x))
        x = self.pool(x)
        x = self.act(self.conv_out(x))

        return x


class CNNConvBlock1(nn.Module):
    def __init__(self, ch_in, ch_out, k, h, w):
        super(CNNConvBlock1, self).__init__()
        self.BN = nn.BatchNorm2d(ch_in)
        self.conv_in = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv_out = nn.Conv2d(ch_out, ch_out, kernel_size=k, padding=k//2, stride=1, groups=ch_out)
        self.conv_out1 = nn.Conv2d(ch_out, ch_out, kernel_size=k, padding=k//2, stride=1)
        self.pool = nn.AvgPool2d(3, padding=1, stride=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.BN(x)
        x = self.act(self.conv_in(x))
        x = self.pool(x)
        x = self.act(self.conv_out1(self.conv_out(x)))

        return x


class CNNConvBlock2(nn.Module):
    def __init__(self, ch_in, ch_out, k, h, w):
        super(CNNConvBlock2, self).__init__()
        self.BN = nn.BatchNorm2d(ch_in)
        self.conv_in = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv_out = nn.Conv2d(ch_out, ch_out, kernel_size=k, padding=k//2, stride=1, groups=ch_out)
        self.conv_out1 = nn.Conv2d(ch_out, ch_out, kernel_size=k, padding=k//2, stride=1)
        self.conv_out2 = nn.Conv2d(ch_out, ch_out, kernel_size=k, padding=k//2, stride=1)
        self.pool = nn.AvgPool2d(3, padding=1, stride=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.BN(x)
        x = self.act(self.conv_in(x))
        x = self.pool(x)
        x = self.act(self.conv_out2(self.conv_out1(self.conv_out(x))))

        return x


class FFM(nn.Module):   #对应ccc.py
    def __init__(self, channels=64, r=4):
        super(FFM, self).__init__()
        inter_channels = int(channels // r)

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.global_att1 = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        xa = x1 + x2
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xg1 = self.global_att1(xa)
        xlg = xl + xg + xg1
        w = self.sigmoid(xlg)

        xo = x1 * w + x2 * (1 - w)
        return xo

class Net(nn.Module):
    def __init__(self,
                 height: int,
                 width: int,
                 channel: int,
                 class_count: int,
                 Q1: torch.Tensor,
                 A1: torch.Tensor,
                 Q2: torch.Tensor,
                 A2: torch.Tensor):

        super(Net, self).__init__()

        self.class_count = class_count
        self.channel = channel
        self.height = height
        self.width = width
        self.Q1 = Q1
        self.A1 = A1
        self.Q2 = Q2
        self.A2 = A2
        self.norm_col_Q1 = torch.sum(Q1, 0, keepdim=True)    #计算分配矩阵的列和
        self.norm_col_Q2 = torch.sum(Q2, 0, keepdim=True)

        self.gcf = ThreeFeatureGating(64)    #定义三路门控特征融合模块
        self.feat = nn.Linear(self.channel, 64)  #定义全连接层
        self.BN_GCN = nn.BatchNorm1d(64)    #定义 1D 批归一化
        self.GCN_Branch = GCNConvBlock(64, 64)   #定义 GCN 模块

        self.CNNlayerA1 = CNNConvBlock(self.channel, 64, 3, self.height, self.width)  #定义第一组 CNN 模块
        self.CNNlayerA2 = CNNConvBlock(64, 64, 3, self.height, self.width)
        self.CNNlayerA3 = CNNConvBlock(64, 64, 3, self.height, self.width)

        self.CNNlayerB1 = CNNConvBlock(self.channel, 64, 3, self.height, self.width)   #定义第二组 CNN 模块
        self.CNNlayerB2 = CNNConvBlock(64, 64, 3, self.height, self.width)
        self.CNNlayerB3 = CNNConvBlock(64, 64, 3, self.height, self.width)

        self.CNNlayerC1 = CNNConvBlock(self.channel, 64, 3, self.height, self.width)    #定义第三组 CNN 模块  三组CNN对应论文图4
        self.CNNlayerC2 = CNNConvBlock(64, 64, 3, self.height, self.width)
        self.CNNlayerC3 = CNNConvBlock(64, 64, 3, self.height, self.width)
        self.cross = FFM(64, 4)                                                        #定义交叉融合模块GCN+CNN
        self.Cross_fc_out = nn.Linear(128, self.class_count)                      #定义分类全连接层

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # GCN Conv,
        (h1, w1, c1) = x1.shape
        (h2, w2, c2) = x2.shape

        x1_flatten = x1.reshape([h1 * w1, -1])               #展平第一个时相图像
        superpixels_flatten_1 = torch.mm(self.Q1.t(), x1_flatten)   #聚合像素特征为超像素特征
        superpixels_flatten_1 = superpixels_flatten_1 / self.norm_col_Q1.t()   #归一化超像素特征
        superpixels_flatten_1 = self.feat(superpixels_flatten_1)            #全连接层映射
        superpixels_flatten_1 = self.BN_GCN(superpixels_flatten_1)         #批归一化

        x2_flatten = x2.reshape([h2 * w2, -1])
        superpixels_flatten_2 = torch.mm(self.Q2.t(), x2_flatten)
        superpixels_flatten_2 = superpixels_flatten_2 / self.norm_col_Q2.t()
        superpixels_flatten_2 = self.feat(superpixels_flatten_2)
        superpixels_flatten_2 = self.BN_GCN(superpixels_flatten_2)
        # 第一层gcn
        out11 = self.GCN_Branch(superpixels_flatten_1, self.A1)
        out11 = out11 + superpixels_flatten_1               #残差连接，增强稳定性
        out11 = self.BN_GCN(out11)
        out11 = F.relu(out11)
        # 第二层gcn
        out12 = self.GCN_Branch(out11, self.A1)
        #out12 = out12 + out11 + superpixels_flatten_1      #残差连接，增强稳定性
        out12 = self.BN_GCN(out12)
        out12 = F.relu(out12)
        # decoder为h*w,c）的形式
        GCNout1 = torch.matmul(self.Q1, out12)
        # 第一层gcn
        out21 = self.GCN_Branch(superpixels_flatten_2, self.A2)
        out21 = out21 + superpixels_flatten_2
        out21 = self.BN_GCN(out21)
        out21 = F.relu(out21)
        # 第二层gcn
        out22 = self.GCN_Branch(out21, self.A2)
        out22 = out22 + out21 + superpixels_flatten_2
        out22 = self.BN_GCN(out22)
        out22 = F.relu(out22)
        GCNout2 = torch.matmul(self.Q2, out22)

        # CNN Conv, 要先增加第一个维度
        CNNin1 = torch.unsqueeze(x1.permute([2, 0, 1]), 0)
        CNNin2 = torch.unsqueeze(x2.permute([2, 0, 1]), 0)

        CNNmid1_A = self.CNNlayerA1(CNNin1)
        CNNmid1_B = self.CNNlayerB1(CNNin1)
        CNNmid1_C = self.CNNlayerC1(CNNin1)

        CNNin = CNNmid1_A + CNNmid1_B + CNNmid1_C

        CNNmid2_A = self.CNNlayerA2(CNNin)
        CNNmid2_B = self.CNNlayerB2(CNNin)
        CNNmid2_C = self.CNNlayerC2(CNNin)

        CNNin = CNNmid2_A + CNNmid2_B + CNNmid2_C

        CNNout_A = self.CNNlayerA3(CNNin)
        CNNout_B = self.CNNlayerB3(CNNin)
        CNNout_C = self.CNNlayerC3(CNNin)

        #CNNout1 = torch.cat([CNNout_A, CNNout_B, CNNout_C], dim=1) # 64*3
        CNNout1 = self.gcf(CNNout_A, CNNout_B, CNNout_C)

        CNNmid1_A = self.CNNlayerA1(CNNin2)
        CNNmid1_B = self.CNNlayerB1(CNNin2)
        CNNmid1_C = self.CNNlayerC1(CNNin2)


        CNNin = CNNmid1_A + CNNmid1_B + CNNmid1_C

        CNNmid2_A = self.CNNlayerA2(CNNin)
        CNNmid2_B = self.CNNlayerB2(CNNin)
        CNNmid2_C = self.CNNlayerC2(CNNin)


        CNNin = CNNmid2_A + CNNmid2_B + CNNmid2_C

        CNNout_A = self.CNNlayerA3(CNNin)
        CNNout_B = self.CNNlayerB3(CNNin)
        CNNout_C = self.CNNlayerC3(CNNin)

        #CNNout2 = torch.cat([CNNout_A, CNNout_B, CNNout_C], dim=1) # 64*3
        CNNout2 = self.gcf(CNNout_A, CNNout_B, CNNout_C)

        CNNout1 = torch.squeeze(CNNout1, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])
        CNNout2 = torch.squeeze(CNNout2, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])
        # Cross Attn
        CNNout1 = CNNout1.transpose(0, 1).reshape([64, self.height, self.width]).unsqueeze(0)
        CNNout2 = CNNout2.transpose(0, 1).reshape([64, self.height, self.width]).unsqueeze(0)

        GCNout1 = GCNout1.transpose(0, 1).reshape([64, self.height, self.width]).unsqueeze(0)
        GCNout2 = GCNout2.transpose(0, 1).reshape([64, self.height, self.width]).unsqueeze(0)

        out1 = self.cross(CNNout1, GCNout1)
        out2 = self.cross(CNNout2, GCNout2)

        out1 = torch.squeeze(out1, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])
        out2 = torch.squeeze(out2, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])

        out = torch.cat([out1, out2], dim=-1)
        out = self.Cross_fc_out(out)
        out = F.softmax(out, -1)

        return out




