import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

torch.manual_seed(1234)

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        print("self.weight:", self.weight.shape)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # 计算度矩阵 D
        degree = torch.sum(adj, dim=1)
        # 计算单位矩阵 I
        I = torch.eye(adj.size(0), device=adj.device)
        # 计算 A_tilde = 2I - adj
        A_tilde = 2 * I - adj
        # 计算 D_tilde 的对角线元素：2 + degree
        d_tilde_diag = 2 + degree
        # 计算 D_tilde_inv_sqrt 的对角线元素：1 / sqrt(d_tilde_diag)
        d_tilde_inv_sqrt_diag = 1.0 / torch.sqrt(d_tilde_diag)
        # 创建 D_tilde_inv_sqrt 作为对角矩阵
        D_tilde_inv_sqrt = torch.diag(d_tilde_inv_sqrt_diag)
        # 计算拉普拉斯锐化矩阵 L_sharpen = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
        L_sharpen = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
        # 计算 support = input @ self.weight
        support = torch.matmul(input, self.weight)
        # 计算 output = L_sharpen @ support
        output = torch.matmul(L_sharpen, support)
        # 添加偏置（如果存在）
        if self.bias is not None:
            output = output + self.bias
        # 应用 ReLU 激活函数
        output = torch.relu(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'