import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import inits
import torch.nn.init as init
from sklearn.neighbors import kneighbors_graph
import numpy as np


class WeightedCosine(nn.Module):
    """
    Compute weighted cosine similarity between node embeddings.

    Parameters
    ----------
    d_in : int
        Dimension of input node features.
    num_pers : int
        Number of learnable perspectives or attention heads.
    weighted : bool
        Whether to use learnable perspective weights. If False, standard cosine similarity is used.
    normalize : bool
        Whether to normalize node embeddings before similarity computation.
    """

    def __init__(self, d_in, num_pers=16, weighted=True, normalize=True):
        super(WeightedCosine, self).__init__()
        self.normalize = normalize
        self.w = None

        if weighted:
            self.w = nn.Parameter(torch.FloatTensor(num_pers, d_in))
            self.reset_parameters()

    def reset_parameters(self):
        """Initialize learnable perspective weights with Xavier uniform initialization."""
        init.xavier_uniform_(self.w)

    def forward(self, x, y=None, non_negative=False):
        """
        Compute pairwise weighted cosine similarities between two groups of node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Source node embeddings.
        y : torch.Tensor, optional
            Target node embeddings. If None, x is used as y.
        non_negative : bool
            Whether to mask negative similarity values.

        Returns
        -------
        adj : torch.Tensor
            Pairwise similarity matrix.
        """
        if y is None:
            y = x

        context_x = x.unsqueeze(0)
        context_y = y.unsqueeze(0)

        # Apply learnable perspective weights if weighted cosine is enabled.
        if self.w is not None:
            expand_weight_tensor = self.w.unsqueeze(1)
            context_x = context_x * expand_weight_tensor
            context_y = context_y * expand_weight_tensor

        # Normalize embeddings before cosine similarity computation.
        if self.normalize:
            context_x = F.normalize(context_x, p=2, dim=-1)
            context_y = F.normalize(context_y, p=2, dim=-1)

        # Average similarities over all perspectives.
        adj = torch.matmul(context_x, context_y.transpose(-1, -2)).mean(0)

        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)

        return adj


class Cosine(nn.Module):
    """
    Compute standard cosine similarity or dot-product similarity between node embeddings.
    """

    def __init__(self):
        super(Cosine, self).__init__()

    def forward(self, x, y=None, non_negative=False, use_normlize=False):
        """
        Compute pairwise similarities between two groups of node embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Source node embeddings.
        y : torch.Tensor, optional
            Target node embeddings. If None, x is used as y.
        non_negative : bool
            Whether to mask negative similarity values.
        use_normlize : bool
            Whether to normalize embeddings before similarity computation.

        Returns
        -------
        adj : torch.Tensor
            Pairwise similarity matrix.
        """
        if y is None:
            y = x

        if use_normlize:
            context_x = F.normalize(x, p=2, dim=-1)
            context_y = F.normalize(y, p=2, dim=-1)
        else:
            context_x = x
            context_y = y

        adj = torch.matmul(context_x, context_y.T)

        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)

        return adj


class InnerProduct(nn.Module):
    """
    Compute inner-product similarity between node embeddings.
    """

    def __init__(self):
        super(InnerProduct, self).__init__()

    def forward(self, x, y=None, non_negative=False):
        """
        Compute pairwise inner-product similarities.

        Parameters
        ----------
        x : torch.Tensor
            Source node embeddings.
        y : torch.Tensor, optional
            Target node embeddings. If None, x is used as y.
        non_negative : bool
            Whether to mask negative similarity values.

        Returns
        -------
        adj : torch.Tensor
            Pairwise inner-product similarity matrix.
        """
        if y is None:
            y = x

        adj = torch.matmul(x, y.T)

        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)

        return adj


class GeneralizedMetric(nn.Module):
    """
    Generalized node similarity metric for graph structure learning.

    This module introduces multiple learnable transformation matrices to compute node-pair
    similarities in different semantic spaces. Compared with standard cosine or inner-product
    metrics, it can capture richer semantic relations, but it also introduces more parameters.
    """

    def __init__(self, d_in, num_pers=16, normalize=True):
        super(GeneralizedMetric, self).__init__()
        self.normalize = normalize

        # Initialize each perspective matrix as an identity matrix.
        self.Q = nn.Parameter(torch.eye(d_in).unsqueeze(0).repeat(num_pers, 1, 1))

    def reset_parameters(self):
        """Initialize transformation matrices with Xavier uniform initialization."""
        init.xavier_uniform_(self.Q)

    def forward(self, x, y=None, non_negative=False):
        """
        Compute pairwise generalized metric similarities.

        Parameters
        ----------
        x : torch.Tensor
            Source node embeddings.
        y : torch.Tensor, optional
            Target node embeddings. If None, x is used as y.
        non_negative : bool
            Whether to mask negative similarity values.

        Returns
        -------
        adj : torch.Tensor
            Pairwise similarity matrix averaged over all perspectives.
        """
        Q = F.softmax(self.Q, dim=-1)
        num_heads = self.Q.shape[0]

        if y is None:
            y = x

        context_x = x.unsqueeze(0)
        context_y = y.unsqueeze(0)

        if self.normalize:
            context_x = F.normalize(context_x, p=2, dim=-1)
            context_y = F.normalize(context_y, p=2, dim=-1)

        adj = torch.bmm(
            torch.bmm(context_x.repeat(num_heads, 1, 1), Q),
            context_y.transpose(-1, -2).repeat(num_heads, 1, 1)
        ).mean(0)

        if non_negative:
            mask = (adj > 0).detach().float()
            adj = adj * mask + 0 * (1 - mask)

        return adj


class FGP(nn.Module):
    """
    Learnable full graph parameterization used as an alternative graph structure learning module.

    This module learns a dense adjacency matrix during training and transforms it through a
    differentiable nonlinear function. It is a typical graph structure learning strategy used
    in scenarios with missing, weak, or noisy graph structures.
    """

    def __init__(self, n, nonlinear=None, init_adj=None):
        super(FGP, self).__init__()

        self.Adj = nn.Parameter(torch.FloatTensor(n, n))
        self.nonlinear = lambda adj: F.elu(adj) + 1

        if nonlinear:
            self.nonlinear = eval(nonlinear)

        if init_adj:
            self.init_estimation(init_adj)

    def reset_parameters(self, features, k, metric, i):
        """
        Initialize the learnable adjacency matrix with a k-nearest-neighbor graph.

        Parameters
        ----------
        features : np.ndarray
            Node feature matrix.
        k : int
            Number of nearest neighbors.
        metric : str
            Distance metric used by kneighbors_graph.
        i : float
            Scaling factor for the initialized adjacency matrix.
        """
        adj = kneighbors_graph(features, k, metric=metric)
        adj = np.array(adj.todense(), dtype=np.float32)
        adj += np.eye(adj.shape[0])
        adj = adj * i - i
        self.Adj.data.copy_(torch.tensor(adj))

    def init_estimation(self, adj):
        """Initialize the learnable adjacency matrix with a given adjacency matrix."""
        self.Adj.data.copy_(adj)

    def forward(self, x):
        """
        Return the transformed learnable adjacency matrix.

        The input x is kept for interface consistency with other metric modules.
        """
        return self.nonlinear(self.Adj)


class GeneralizedMahalanobis(nn.Module):
    """
    Learn a Mahalanobis distance matrix for node-pair similarity computation.

    This metric can capture more complex relations between node embeddings and can be used
    for graph construction or edge prediction.

    Metric from the paper:
    "Adaptive Graph Convolutional Neural Networks" <http://arxiv.org/abs/1801.03226>
    """

    def __init__(self, d_in, sigma=1):
        super(GeneralizedMahalanobis, self).__init__()

        self.W = nn.Parameter(torch.FloatTensor(d_in, d_in))
        self.sigma = sigma

    def forward(self, x, y=None, edge=None):
        """
        Compute Mahalanobis-kernel similarities.

        Parameters
        ----------
        x : torch.Tensor
            Source node embeddings.
        y : torch.Tensor, optional
            Target node embeddings. If None, x is used as y.
        edge : torch.Tensor, optional
            Sparse edge index. If provided, similarities are computed only on the given edges.

        Returns
        -------
        torch.Tensor
            Dense similarity matrix if edge is None; otherwise, a sparse similarity matrix.
        """
        device = x.device

        if y is None:
            y = x

        M = self.W @ self.W.T

        if edge:
            d = torch.index_select(x, 0, edge[0]) - torch.index_select(y, 0, edge[1])
            D = torch.sqrt(((d @ M) * d).sum(1))
            D = torch.exp(-D / (2 * self.sigma ** 2))
            return torch.sparse.FloatTensor(edge, D, [x.shape[0], y.shape[0]]).to(device)

        D = torch.zeros(x.shape[0], y.shape[0])

        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                d = x[i] - y[j]
                D[i, j] = d @ M @ d.T

        D = torch.exp(-D / (2 * self.sigma ** 2))

        return D