import torch
import torch.nn.functional as F
import torch_geometric.utils as utils


def edge_reconstruct_loss(conf, edge_index, representations, mode='bce', tau=1.0):
    """
    Stable edge reconstruction loss.

    Two modes are supported:
    - mode='cosine_mse':
        Normalize node representations first, then compute cosine similarity.
        Positive edges are optimized toward 1, and negative edges are optimized toward 0.

    - mode='bce':
        Use the dot product between node representations as logits and apply
        binary cross-entropy with logits. Positive edges are labeled as 1,
        and negative edges are labeled as 0. The temperature tau controls the
        logit scale.

    Parameters
    ----------
    conf : object
        Configuration object. The number of negative samples is read from
        conf.gsl['neg_numbers'].
    edge_index : torch.LongTensor, shape [2, E]
        Edge index of the graph. The graph may contain bidirectional edges.
        This function keeps only upper-triangular edges to avoid duplicate
        pairs such as (u, v) and (v, u).
    representations : torch.FloatTensor, shape [N, d]
        Node representation matrix.
    mode : str
        Reconstruction loss type. Options are 'cosine_mse' and 'bce'.
    tau : float
        Temperature coefficient. In BCE mode, the dot-product logits are divided
        by tau. Larger tau reduces the logit scale and improves stability, while
        smaller tau strengthens the contrast.

    Returns
    -------
    loss : torch.Tensor
        Scalar reconstruction loss.
    """
    device = representations.device
    num_nodes = representations.size(0)

    # Keep only upper-triangular positive edges to remove duplicate directions
    # and self-loops.
    pos_e = edge_index[:, edge_index[0] < edge_index[1]]

    # Sample negative edges according to the configured number of negatives.
    neg_e = utils.negative_sampling(
        edge_index,
        num_nodes=num_nodes,
        num_neg_samples=conf.gsl['neg_numbers']
    )
    neg_e = neg_e[:, neg_e[0] < neg_e[1]]

    # If negative sampling returns no valid negative edge, sample again using
    # the number of positive edges. If it is still empty, return zero to avoid NaN.
    if neg_e.numel() == 0:
        neg_e = utils.negative_sampling(
            edge_index,
            num_nodes=num_nodes,
            num_neg_samples=pos_e.size(1)
        )
        neg_e = neg_e[:, neg_e[0] < neg_e[1]]

        if neg_e.numel() == 0:
            return torch.zeros([], device=device, dtype=representations.dtype)

    # Extract node representations for positive and negative edge endpoints.
    p0, p1 = representations[pos_e[0]], representations[pos_e[1]]
    n0, n1 = representations[neg_e[0]], representations[neg_e[1]]

    if mode == 'cosine_mse':
        # Normalize representations and compute cosine similarity in [-1, 1].
        Z = F.normalize(representations, p=2, dim=1)

        p0, p1 = Z[pos_e[0]], Z[pos_e[1]]
        n0, n1 = Z[neg_e[0]], Z[neg_e[1]]

        pos = torch.sum(p0 * p1, dim=1)
        neg = torch.sum(n0 * n1, dim=1)

        # Positive edges are encouraged to have similarity close to 1.
        # Negative edges are encouraged to have similarity close to 0.
        loss_pos = F.mse_loss(pos, torch.ones_like(pos), reduction='mean')
        loss_neg = F.mse_loss(neg, torch.zeros_like(neg), reduction='mean')

        loss = loss_pos + loss_neg

    elif mode == 'bce':
        # Use dot products as logits and scale them by tau.
        pos_logits = torch.sum(p0 * p1, dim=1) / tau
        neg_logits = torch.sum(n0 * n1, dim=1) / tau

        # Binary labels: positive edges are 1, negative edges are 0.
        target_pos = torch.ones_like(pos_logits)
        target_neg = torch.zeros_like(neg_logits)

        # BCEWithLogitsLoss internally applies sigmoid and is numerically stable.
        loss_pos = F.binary_cross_entropy_with_logits(
            pos_logits,
            target_pos,
            reduction='mean'
        )
        loss_neg = F.binary_cross_entropy_with_logits(
            neg_logits,
            target_neg,
            reduction='mean'
        )

        loss = loss_pos + loss_neg

    else:
        raise ValueError(f"Unknown mode={mode}, choose from ['cosine_mse','bce']")

    return loss


def sim(z1, z2, tau):
    """
    Compute the exponentiated pairwise cosine similarity between two embedding matrices.

    Parameters
    ----------
    z1 : torch.Tensor, shape [N, d]
        The first embedding matrix.
    z2 : torch.Tensor, shape [M, d]
        The second embedding matrix.
    tau : float
        Temperature coefficient for similarity scaling.

    Returns
    -------
    sim_matrix : torch.Tensor, shape [N, M]
        Pairwise similarity matrix after exponential scaling.
    """
    z1_norm = torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = torch.norm(z2, dim=-1, keepdim=True)

    dot_numerator = torch.mm(z1, z2.t())
    dot_denominator = torch.mm(z1_norm, z2_norm.t())

    sim_matrix = torch.exp(dot_numerator / dot_denominator / tau)

    return sim_matrix


def graph_infonce(z1_proj, z2_proj):
    """
    Compute graph-level InfoNCE loss between two projected graph views.

    Each node representation in one view treats the representation of the same
    node in the other view as its positive sample, while all other nodes are
    treated as implicit negative samples.

    Parameters
    ----------
    z1_proj : torch.Tensor, shape [N, d]
        Projected node representations from the first view.
    z2_proj : torch.Tensor, shape [N, d]
        Projected node representations from the second view.

    Returns
    -------
    loss : torch.Tensor
        Symmetric InfoNCE loss between the two views.
    """
    matrix_z1z2 = sim(z1_proj, z2_proj)
    matrix_z2z1 = matrix_z1z2.t()

    matrix_z1z2 = matrix_z1z2 / (torch.sum(matrix_z1z2, dim=1).view(-1, 1) + 1e-8)
    lori_v1v2 = -torch.log(matrix_z1z2.diag() + 1e-8).mean()

    matrix_z2z1 = matrix_z2z1 / (torch.sum(matrix_z2z1, dim=1).view(-1, 1) + 1e-8)
    lori_v2v1 = -torch.log(matrix_z2z1.diag() + 1e-8).mean()

    return (lori_v1v2 + lori_v2v1) / 2


def dvc_infonce(z1, z2, tau=0.5):
    """
    Compute symmetric DVC-style InfoNCE loss between two normalized views.

    z1 and z2 are assumed to be L2-normalized projected representations.
    For each node, its counterpart in the other view is treated as the positive
    sample, while all other nodes in the batch are treated as negative samples.

    The loss is defined as:

        L = (1 / 2N) * sum_i [
            -log( exp(sim(z1_i, z2_i) / tau) / sum_j exp(sim(z1_i, z2_j) / tau) )
            -log( exp(sim(z2_i, z1_i) / tau) / sum_j exp(sim(z2_i, z1_j) / tau) )
        ]

    Parameters
    ----------
    z1 : torch.Tensor, shape [N, d]
        Normalized projected representations from the first view.
    z2 : torch.Tensor, shape [N, d]
        Normalized projected representations from the second view.
    tau : float
        Temperature coefficient.

    Returns
    -------
    loss : torch.Tensor
        Symmetric contrastive loss.
    """
    logits12 = (z1 @ z2.t()) / tau
    logits21 = logits12.t()

    # Use logsumexp for numerical stability. Diagonal entries are positive pairs.
    l1 = torch.diag(logits12) - torch.logsumexp(logits12, dim=1)
    l2 = torch.diag(logits21) - torch.logsumexp(logits21, dim=1)

    loss = -(l1.mean() + l2.mean()) * 0.5

    return loss