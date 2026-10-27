import torch
import torch.nn as nn
from utils.functional import knn, symmetry, normalize, apply_non_linearity
from utils.metric import Cosine

class Normalize(nn.Module):

    def __init__(self, style='symmetric', add_loop=True, p=None):
        super(Normalize, self).__init__()
        self.style = style
        self.add_loop = add_loop
        self.p = p

    def forward(self, adj):
        return normalize(adj, self.style, self.add_loop, self.p)


class Symmetry(nn.Module):

    def __init__(self, i=2):
        super(Symmetry, self).__init__()
        self.i = i

    def forward(self, adj):
        return symmetry(adj, self.i)


class KNN(nn.Module):

    def __init__(self, K, self_loop=True, set_value=None, metric='cosine', sparse_out=False):
        super(KNN, self).__init__()
        self.K = K
        self.self_loop = self_loop
        self.set_value = set_value
        self.sparse_out = sparse_out
        if metric:
            if metric == 'cosine':
                self.metric = Cosine()

    def forward(self, x=None, adj=None):

        assert not (x is None and adj is None)
        if x is not None:
            dist = self.metric(x)
        else:
            dist = adj
        return knn(dist, self.K, self.self_loop, set_value=self.set_value, sparse_out=self.sparse_out)

class NonLinear(nn.Module):
    """
    非线性变换函数
    """
    def __init__(self, non_linearity, i=None):
        super(NonLinear, self).__init__()
        self.non_linearity = non_linearity
        self.i = i

    def forward(self, adj):

        return apply_non_linearity(adj, self.non_linearity, self.i)