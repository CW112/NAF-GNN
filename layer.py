from __future__ import division
import torch
import torch.nn as nn
from torch.nn import init
import numbers
import torch.nn.functional as F
from torch_geometric.nn import ARMAConv

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()

class dy_nconv(nn.Module):
    def __init__(self):
        super(dy_nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)


class prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(prop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


import torch
import torch.nn as nn
import torch.nn.functional as F


# 假设 nconv 和 linear 已经定义
# class nconv(nn.Module): ...
# class linear(nn.Module): ...

class mixprop_bern(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop_bern, self).__init__()
        self.nconv = nconv()

        # 1. MLP 现在只处理 c_in -> c_out，因为聚合后维度不变
        self.mlp = linear(c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha  # 用于传播

        # 2. 引入一个新的可学习参数 t，用于伯恩斯坦聚合
        # 我们初始化为 0，通过 sigmoid 映射到 (0, 1) 区间
        self.t_bern = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def _get_bernstein_coeffs(self, t, n):
        """
        计算 n 阶伯恩斯坦多项式在 t 处的系数
        返回一个 [n+1] 维的向量
        """
        i = torch.arange(0, n + 1, device=t.device, dtype=t.dtype)

        # 使用 lgamma (log-gamma) 函数来计算 log(nCk)，以保证数值稳定性
        log_comb = (torch.lgamma(torch.tensor(n + 1.0, device=t.device)) -
                    torch.lgamma(i + 1.0) -
                    torch.lgamma(torch.tensor(n, device=t.device) - i + 1.0))
        comb = torch.exp(log_comb)

        # 计算系数
        coeffs = comb * (t ** i) * ((1 - t) ** (n - i))
        return coeffs

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)

        # 传播循环 (与原版一致)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)

        # ----- 聚合阶段 (替换部分) -----

        # 1. 将 list 堆叠成一个张量 [gdep+1, N, c_in]
        H_stack = torch.stack(out, dim=0)

        # 2. 计算伯恩斯坦参数 t (映射到 0-1)
        t = torch.sigmoid(self.t_bern)

        # 3. 计算伯恩斯坦系数 [gdep+1]
        coeffs = self._get_bernstein_coeffs(t, self.gdep)

        # 4. 调整形状以进行广播 [gdep+1, 1, 1]
        coeffs_reshaped = coeffs.view(self.gdep + 1, 1, 1)

        # 5. 执行加权求和 [N, c_in]
        ho = torch.sum(coeffs_reshaped * H_stack, dim=0)

        # 6. 通过最后的 MLP [N, c_out]
        ho = self.mlp(ho)
        return ho





class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha


    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class ARMAKernel(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, activation=F.relu, lambda_k=1.0):
        super().__init__()
        self.nconv = nconv()
        self.gdep = gdep
        self.dropout = dropout
        self.activation = activation
        self.lambda_k = lambda_k  # 固定 λ_k

        self.W0 = linear(c_in, c_out, bias= True)
        self.W1 = linear(c_in, c_out, bias=  True)
        self.V = linear(c_out, c_out, bias =  True)

    def forward(self, x, a):
        """
        x: [B,C_in,N,T]
        adj: [N,N] 固定邻接矩阵
        """
        # 邻接矩阵归一化



        # d_inv_sqrt = torch.pow(d, -0.5)  # 加一个很小的epsilon
        # d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0  # 防止inf
        # # 对称归一化
        # a = d_inv_sqrt.view(-1, 1) * adj * d_inv_sqrt.view(1, -1)

        h = x


        x_v = self.V(x)
        h_agg = self.nconv(h, a)
        h_w = self.W0(h_agg)
        h = F.relu6(h_w + x_v)
        for i in range(self.gdep-1):
            h_agg = self.nconv(h, a)
            h_w = self.W1(h_agg)
            if i < self.gdep - 2:  # 前几层加激活
                h = F.relu(h_w + x_v)
            else:  # 最后一层不加激活
                h = h_w + x_v
        return h


# -----------------------
# 多核 ARMAKhorizon
# -----------------------
class ARMAK(nn.Module):
    def __init__(self, c_in, c_out, gdep = 2, K =  2, dropout = 0.2, activation=F.relu, agg="sum"):
        super().__init__()
        self.K = K
        self.agg = agg

        # 每个核固定 λ_k
        lambda_fixed = 1.0
        self.kernels = nn.ModuleList([
            ARMAKernel(c_in, c_out, gdep, 0.5, activation, lambda_k=lambda_fixed)
            for k in range(K)
        ])


        if agg == "param":
            # 可训练参数，用 softmax 做归一化
            self.alpha = nn.Parameter(torch.ones(K) / K)

    def forward(self, x, adj):
        outs = []

        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        dv = d
        a = torch.eye(adj.size(0)).to(x.device) - adj / dv.view(-1, 1)

        for k in range(self.K):
            outs.append(self.kernels[k](x, a))

        if self.agg == "sum":
            out = torch.stack(outs, dim=0).sum(0)

        elif self.agg == "param":
            weights = F.sigmoid(self.alpha)  # [K]
            outs = torch.stack(outs, dim=0)  # [K,B,C,N,T]
            out = torch.einsum("k,kbcnt->bcnt", weights, outs)  # 加权求和

        else:
            raise ValueError(f"Unknown agg {self.agg}")

        return out



class dy_mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(dy_mixprop, self).__init__()
        self.nconv = dy_nconv()
        self.mlp1 = linear((gdep+1)*c_in,c_out)
        self.mlp2 = linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = linear(c_in,c_in)
        self.lin2 = linear(c_in,c_in)


    def forward(self,x):
        #adj = adj + torch.eye(adj.size(0)).to(x.device)
        #d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)

        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)


        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2




class dilated_1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class graph_global(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_global, self).__init__()
        self.nnodes = nnodes
        self.A = nn.Parameter(torch.randn(nnodes, nnodes).to(device), requires_grad=True).to(device)

    def forward(self, idx):
        return F.relu(self.A)


class graph_undirected(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_undirected, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb1(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj



class graph_directed(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_directed, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


