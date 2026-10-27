import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import dgl
import random


"""
Graph conversion utilities for sparse and dense representations.
"""


def edge_index_to_dense(edge_index, num_nodes):
    """
    Convert an edge_index tensor with shape [2, E] into a dense adjacency matrix.

    Parameters
    ----------
    edge_index : torch.LongTensor
        Edge index tensor with shape [2, E].
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    adj : torch.FloatTensor
        Dense adjacency matrix with shape [N, N]. Entries with edges are set to 1,
        and entries without edges are set to 0.
    """
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def convert_to_sparse_coo(sparse_tensor):
    row, col, value = sparse_tensor.coo()
    coo_indices = torch.stack([row, col], dim=0)
    sparse_coo_tensor = torch.sparse_coo_tensor(coo_indices, value, sparse_tensor.sizes())
    return sparse_coo_tensor


def scipy_sparse_to_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.

    Parameters
    ----------
    sparse_mx : scipy.sparse_matrix
        Sparse matrix to convert.

    Returns
    -------
    sparse_tensor : torch.Tensor
        Sparse tensor in COO format.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sparse_tensor_to_scipy_sparse(sparse_tensor):
    """
    Convert a torch sparse tensor to a scipy sparse matrix.

    Parameters
    ----------
    sparse_tensor : torch.Tensor
        Sparse tensor to convert.

    Returns
    -------
    sparse_mx : scipy.sparse_matrix
        Converted scipy sparse matrix.
    """
    sparse_tensor = sparse_tensor.cpu()
    row = sparse_tensor.coalesce().indices()[0].numpy()
    col = sparse_tensor.coalesce().indices()[1].numpy()
    values = sparse_tensor.coalesce().values().numpy()
    return sp.coo_matrix((values, (row, col)), shape=sparse_tensor.shape)


"""
Feature and adjacency normalization utilities.
"""


def normalize(mx, style='symmetric', add_loop=True, p=None):
    """
    Normalize a feature matrix or an adjacency matrix.

    Parameters
    ----------
    mx : torch.Tensor
        Dense or sparse tensor to be normalized.
    style : str
        Normalization type. Supported options are:
        - 'row': row-wise normalization;
        - 'symmetric': symmetric GCN-style normalization;
        - 'softmax': row-wise softmax normalization;
        - 'row-norm': F.normalize-based row normalization.
    add_loop : bool
        Whether to add self-loops when symmetric normalization is used.
    p : int or float, optional
        Norm degree used by F.normalize when style='row-norm'.

    Returns
    -------
    torch.Tensor
        Normalized dense or sparse tensor.
    """
    # Row-wise normalization.
    if style == 'row':
        if mx.is_sparse:
            return row_normalize_sp(mx)
        else:
            return row_nomalize(mx)

    # Symmetric normalization used by GCN.
    elif style == 'symmetric':
        if mx.is_sparse:
            return normalize_sp_tensor_tractable(mx, add_loop)
        else:
            return normalize_tensor(mx, add_loop)

    # Softmax-based normalization.
    elif style == 'softmax':
        if mx.is_sparse:
            return torch.sparse.softmax(mx, dim=-1)
        else:
            return F.softmax(mx, dim=-1)

    # F.normalize-based row normalization.
    elif style == 'row-norm':
        assert p is not None
        if mx.is_sparse:
            # TODO: implement sparse row normalization with F.normalize-like behavior.
            pass
        else:
            return F.normalize(mx, dim=-1, p=p)
    else:
        raise KeyError("The normalize style is not provided.")


def row_nomalize(mx):
    """
    Apply row normalization to a dense adjacency matrix.

    Parameters
    ----------
    mx : torch.Tensor
        Dense adjacency matrix.

    Returns
    -------
    mx : torch.Tensor
        Row-normalized dense adjacency matrix.
    """
    r_sum = mx.sum(1)
    r_inv = r_sum.pow(-1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx

    return mx


def row_normalize_sp(mx):
    """
    Apply row normalization to a sparse adjacency matrix.

    Parameters
    ----------
    mx : torch.Tensor
        Sparse adjacency matrix.

    Returns
    -------
    torch.Tensor
        Row-normalized sparse adjacency matrix.
    """
    adj = mx.coalesce()
    inv_sqrt_degree = 1. / (torch.sparse.sum(mx, dim=1).values() + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]]
    new_values = adj.values() * D_value
    return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def normalize_sp_tensor_tractable(adj, add_loop=True):
    """
    Apply symmetric normalization to a sparse adjacency matrix using torch tensors.

    Parameters
    ----------
    adj : torch.Tensor
        Sparse adjacency matrix.
    add_loop : bool
        Whether to add self-loops before normalization.

    Returns
    -------
    torch.Tensor
        Symmetrically normalized sparse adjacency matrix.
    """
    n = adj.shape[0]
    device = adj.device
    if add_loop:
        adj = adj + torch.eye(n, device=device).to_sparse()
    adj = adj.coalesce()
    inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + 1e-12)
    D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]
    new_values = adj.values() * D_value
    return torch.sparse_coo_tensor(adj.indices(), new_values, adj.size())


def normalize_tensor(adj, add_loop=True):
    """
    Apply symmetric normalization to a dense adjacency matrix.

    Parameters
    ----------
    adj : torch.Tensor
        Dense adjacency matrix.
    add_loop : bool
        Whether to add self-loops before normalization.

    Returns
    -------
    A : torch.Tensor
        Symmetrically normalized dense adjacency matrix.
    """
    device = adj.device
    adj_loop = adj + torch.eye(adj.shape[0]).to(device) if add_loop else adj
    rowsum = adj_loop.sum(1)
    r_inv = rowsum.pow(-1 / 2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    A = r_mat_inv @ adj_loop
    A = A @ r_mat_inv
    return A


def normalize_sp_matrix(adj, add_loop=True):
    """
    Apply symmetric normalization to a scipy sparse adjacency matrix.

    Parameters
    ----------
    adj : scipy.sparse_matrix
        Sparse adjacency matrix.
    add_loop : bool
        Whether to add self-loops before normalization.

    Returns
    -------
    new : scipy.sparse_matrix
        Symmetrically normalized sparse matrix.
    """
    mx = adj + sp.eye(adj.shape[0]) if add_loop else adj
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    new = mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    return new


def normalize_conf(node_conf: torch.Tensor, eps: float = 1e-8):
    """
    Normalize node confidence scores into the range [0, 1].

    Parameters
    ----------
    node_conf : torch.Tensor
        Confidence scores for nodes. The tensor can have any shape.
    eps : float
        Small constant used to avoid division by zero.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape as node_conf, linearly mapped into [0, 1].
    """
    min_val = node_conf.min()
    max_val = node_conf.max()
    return (node_conf - min_val) / (max_val - min_val + eps)


"""
Graph structure operations and graph construction utilities.
"""


def symmetry(adj, i=2):
    """
    Convert a dense or sparse adjacency matrix into a symmetric matrix.

    Parameters
    ----------
    adj : torch.Tensor
        Dense or sparse adjacency matrix.
    i : float
        Divisor used when averaging the original matrix and its transpose.

    Returns
    -------
    torch.Tensor
        Symmetric adjacency matrix.
    """
    if adj.is_sparse:
        n = adj.shape[0]
        adj_t = torch.sparse.FloatTensor(adj.indices()[[1, 0]], adj.values(), [n, n])
        return (adj_t + adj).coalesce() / i
    else:
        return (adj.t() + adj) / i


def knn(adj, K, self_loop=True, set_value=None, sparse_out=False):
    """
    Build a Top-K adjacency structure from an input similarity matrix.

    Parameters
    ----------
    adj : torch.Tensor
        Dense similarity matrix or adjacency matrix.
    K : int
        Maximum number of neighbors retained for each node.
    self_loop : bool
        Whether to keep self-loops.
    set_value : float or None
        If provided, all retained Top-K entries are assigned this fixed value.
        Otherwise, their original similarity scores are retained.
    sparse_out : bool
        Whether to return the result as a sparse tensor. If True, the return value
        is a torch sparse tensor. If False, the return value is a dense tensor with
        non-Top-K entries set to 0.

    Returns
    -------
    torch.Tensor
        Top-K graph represented as either a dense tensor or a sparse tensor.
    """
    if adj.is_sparse:
        # TODO: implement Top-K pruning for sparse adjacency matrices.
        pass
    else:
        device = adj.device
        values, indices = adj.topk(k=int(K), dim=-1)
        assert torch.max(indices) < adj.shape[1]
        if sparse_out:
            n = adj.shape[0]
            new_indices = torch.stack([
                torch.arange(n).view(-1, 1).expand(-1, int(K)).contiguous().flatten().to(device),
                indices.flatten()
            ])
            new_values = values.flatten()
            return torch.sparse.FloatTensor(new_indices, new_values, [n, n]).coalesce()
        else:
            mask = torch.zeros(adj.shape).to(device)
            mask[torch.arange(adj.shape[0]).view(-1, 1), indices] = 1.
            if not self_loop:
                mask[
                    torch.arange(adj.shape[0]).view(-1, 1),
                    torch.arange(adj.shape[0]).view(-1, 1)
                ] = 0
            mask.requires_grad = False
            new_adj = adj * mask
            if set_value:
                new_adj[new_adj.nonzero()[:, 0], new_adj.nonzero()[:, 1]] = set_value
            return new_adj


def build_knn_graph_torch(X, k=5):
    """
    Build a symmetric KNN adjacency matrix with PyTorch.

    This implementation corresponds to Formula (6) in GCNet.

    Parameters
    ----------
    X : torch.Tensor
        Node feature matrix with shape [N, d].
    k : int
        Number of nearest neighbors for each node.

    Returns
    -------
    A : torch.Tensor
        Symmetric binary KNN adjacency matrix.
    """
    # Compute pairwise Euclidean distances.
    dist = torch.cdist(X, X, p=2)
    N = X.size(0)
    A = torch.zeros((N, N), dtype=torch.int)

    # Exclude self-connections by setting diagonal distances to infinity.
    dist.fill_diagonal_(float('inf'))

    # Select the k nearest neighbors for each node by applying top-k to negative distances.
    knn_indices = torch.topk(-dist, k, dim=1).indices

    # Build a symmetric adjacency matrix.
    for i in range(N):
        for j in knn_indices[i]:
            A[i, j] = 1
            A[j, i] = 1

    return A


def enn(adj, epsilon, set_value=None):
    """
    Build an epsilon-nearest-neighbor graph from a similarity matrix.

    This function filters out edges whose similarity scores are not greater than
    epsilon and keeps only sufficiently similar node pairs.

    Parameters
    ----------
    adj : torch.Tensor
        Dense or sparse similarity matrix with shape [N, N].
    epsilon : float
        Similarity threshold. Only entries satisfying A_ij > epsilon are retained.
    set_value : float or None
        If provided, all retained edges are assigned this fixed value. Otherwise,
        their original similarity scores are retained.

    Returns
    -------
    torch.Tensor
        Filtered dense or sparse adjacency matrix.
    """
    if adj.is_sparse:
        n = adj.shape[0]
        values = adj.values()
        mask = values > epsilon
        mask.requires_grad = False
        new_values = values[mask]
        if set_value:
            new_values[:] = set_value
        new_indices = adj.indices()[:, mask]
        return torch.sparse.FloatTensor(new_indices, new_values, [n, n])
    else:
        mask = adj > epsilon
        mask.requires_grad = False
        new_adj = adj * mask
        if set_value:
            new_adj[mask] = set_value
        return new_adj


def to_undirected(adj):
    """
    Convert an adjacency matrix into an undirected adjacency matrix.

    This function supports both sparse and dense adjacency matrices.

    Parameters
    ----------
    adj : torch.Tensor
        Input adjacency matrix. Sparse input is assumed to be a binary graph
        whose edge weights are all 1. Dense input may contain arbitrary real-valued
        edge weights.

    Returns
    -------
    torch.Tensor
        Undirected adjacency matrix. Sparse input returns a sparse tensor, and
        dense input returns a dense tensor.

    Notes
    -----
    For sparse input:
        - Reverse each edge index from (i, j) to (j, i).
        - Concatenate original edges and reversed edges.
        - Remove duplicate edges and construct a new sparse adjacency matrix.

    For dense input:
        - Construct a symmetric matrix and keep the larger weight when A_ij and
          A_ji are different.
    """
    if adj.is_sparse:
        device = adj.device
        assert (adj.values() == 1).all()
        n = adj.shape[0]
        indices_t = adj.indices()[[1, 0]]
        new_indices = torch.cat([adj.indices(), indices_t], dim=1)
        new_indices = torch.unique(new_indices, dim=1)
        new_values = torch.ones(new_indices.shape[1]).to(device)
        new_adj = torch.sparse.FloatTensor(new_indices, new_values, [n, n])
        return new_adj
    else:
        return adj + adj.T - adj * (adj <= adj.T) - adj.T * (adj > adj.T)


def knn_fast(X, k, b):
    """
    Build a cosine-similarity KNN graph in batches for large-scale node features.

    The output is represented by sparse adjacency triples: rows, cols, and values.

    Parameters
    ----------
    X : torch.Tensor
        Node feature matrix with shape [N, d]. Features are L2-normalized before
        similarity computation.
    k : int
        Number of neighbors retained for each node.
    b : int
        Batch size used for block-wise similarity computation to reduce memory usage.

    Returns
    -------
    rows : torch.Tensor
        Source node indices of constructed edges.
    cols : torch.Tensor
        Target node indices of constructed edges.
    values : torch.Tensor
        Symmetrically normalized edge weights.

    Procedure
    ---------
    1. Apply L2 normalization to all node features.
    2. Compute cosine similarities between each batch of nodes and all nodes.
    3. Select the top-(k+1) most similar nodes for each node.
    4. Accumulate row and column degrees for normalization.
    5. Return sparse edge triples with symmetric normalization weights.

    Notes
    -----
    This function uses CUDA tensors through .cuda(). For CPU execution, remove
    the explicit .cuda() calls.
    """
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


def apply_non_linearity(adj, non_linearity, i):
    """
    Apply a nonlinear transformation to a graph structure.

    This function is used to enhance the expressive capacity of a graph structure
    or control the distribution of edge strengths.

    Parameters
    ----------
    adj : torch.Tensor
        Input adjacency matrix, usually treated as a dense tensor here.
    non_linearity : str
        Nonlinearity type. Supported options are:
        - 'elu': Exponential Linear Unit with scaling and shifting;
        - 'relu': Rectified Linear Unit;
        - 'none': no nonlinear transformation.
    i : float
        Scaling factor used only for ELU.

    Returns
    -------
    torch.Tensor
        Transformed adjacency matrix for graph structure learning or graph construction.

    Transformation definitions
    --------------------------
    ELU:
        f(A) = ELU(A * i - i) + 1.
        This transformation shifts the output upward and keeps edge weights positive.

    ReLU:
        f(A) = max(0, A).
        This transformation suppresses negative edge weights.

    none:
        The original graph structure is returned without transformation.

    Raises
    ------
    KeyError
        If non_linearity is not supported.
    """
    if non_linearity == 'elu':
        return F.elu(adj * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(adj)
    elif non_linearity == 'none':
        return adj
    else:
        raise KeyError('We dont support the non-linearity yet')


def get_node_homophily(label, adj):
    """
    Calculate node homophily of a graph.

    Parameters
    ----------
    label : torch.Tensor
        Ground-truth labels.
    adj : torch.Tensor
        Dense adjacency matrix.

    Returns
    -------
    homophily : float
        Mean node homophily over nodes with positive degree.
    """
    label = label.cpu().numpy()
    adj = adj.cpu().numpy()
    num_node = len(label)
    label = label.repeat(num_node).reshape(num_node, -1)
    n = (np.multiply((label == label.T), adj)).sum(axis=1)
    d = adj.sum(axis=1)
    homos = []
    for i in range(num_node):
        if d[i] > 0:
            homos.append(n[i] * 1. / d[i])
    return np.mean(homos)


def get_edge_homophily(label, adj):
    """
    Calculate edge homophily of a graph.

    Parameters
    ----------
    label : torch.Tensor
        Ground-truth labels.
    adj : torch.Tensor
        Dense adjacency matrix.

    Returns
    -------
    homophily : torch.Tensor
        Edge homophily of the graph.
    """
    num_edge = adj.sum()
    cnt = 0
    for i, j in adj.nonzero():
        if label[i] == label[j]:
            cnt += adj[i, j]
    return cnt / num_edge


def get_homophily(label, adj, type='node', fill=None):
    """
    Calculate node or edge homophily of a graph.

    Parameters
    ----------
    label : torch.Tensor
        Ground-truth labels.
    adj : torch.Tensor
        Dense adjacency matrix.
    type : str
        Homophily type. Options are 'node' and 'edge'.
    fill : str or None
        Value used to fill the diagonal of adj. If None, the diagonal is not modified.

    Returns
    -------
    homophily : float
        Node or edge homophily of the graph.
    """
    if fill:
        np.fill_diagonal(adj, fill)
    return eval('get_' + type + '_homophily(label, adj)')


def get_adjusted_homophily(_label, adj):
    """
    Calculate adjusted homophily of a graph.

    Parameters
    ----------
    _label : torch.Tensor
        Ground-truth labels.
    adj : torch.Tensor
        Dense adjacency matrix.

    Returns
    -------
    homophily : torch.Tensor
        Adjusted homophily of the graph.
    """
    label = _label.long()
    labels = label.max() + 1
    d = adj.sum(1)
    E = d.sum()
    D = torch.zeros(labels)
    for i in range(adj.shape[0]):
        D[label[i]] += d[i]

    h_edge = get_edge_homophily(label, adj)
    sum_pk = ((D / E) ** 2).sum()

    return (h_edge - sum_pk) / (1 - sum_pk)


def get_label_informativeness(_label, adj):
    """
    Calculate label informativeness of a graph.

    Parameters
    ----------
    _label : torch.Tensor
        Ground-truth labels.
    adj : torch.Tensor
        Dense adjacency matrix.

    Returns
    -------
    label_informativeness : torch.Tensor
        Label informativeness of the graph.
    """
    label = _label.long()
    labels = label.max() + 1
    LI_1 = 0
    LI_2 = 0

    p = torch.zeros((labels, labels))
    for i, j in adj.nonzero():
        p[label[i]][label[j]] = p[label[i]][label[j]] + adj[i][j]

    d = adj.sum(1)
    E = d.sum()
    D = torch.zeros(labels)

    for i in range(adj.shape[0]):
        D[label[i]] = D[label[i]] + d[i]

    for i in range(labels):
        for j in range(labels):
            p[i][j] = p[i][j] / E

    p_ = D / E
    LI_2 = (p_ * torch.log(p_)).sum()
    for i in range(labels):
        for j in range(labels):
            if p[i][j] != 0:
                LI_1 += p[i][j] * torch.log(p[i][j] / (p_[i] * p_[j]))

    return -LI_1 / LI_2


def one_hot(y):
    device = y.device
    c = y.max() + 1
    e = torch.eye(c)
    return e[y].to(device)


def accuracy(logits, labels):
    """
    Calculate classification accuracy.

    Parameters
    ----------
    logits : np.ndarray
        Model output logits or prediction scores.
    labels : np.ndarray
        Ground-truth labels.

    Returns
    -------
    float
        Classification accuracy.
    """
    return np.sum(logits.argmax(1) == labels) / len(labels)


def add_loop(adj):
    n = adj.shape[0]
    device = adj.device
    adj = adj + torch.eye(n, device=device).to_sparse()
    adj = adj.coalesce()
    return adj


def set_seed(seed):
    """
    Set random seeds to make experimental results reproducible.

    Parameters
    ----------
    seed : int
        Random seed to set.
    """
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# if __name__ == '__main__':
#     from torch_geometric import seed_everything
#     seed_everything(42)
#     # adj = torch.rand(5, 5).to_sparse()
#     # adj = torch.sparse.FloatTensor(
#     #     torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4],
#     #                   [1, 2, 3, 4, 0, 1, 2, 3, 3]]),
#     #     torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1]),
#     #     [5, 5]
#     # )
#     adj = torch.rand(3, 3).to_sparse()
#     x = torch.rand(10, 3)
#     print(adj)
#     print(enn(adj, 0.5))
