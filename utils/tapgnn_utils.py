"""Sparse propagation, unit axes, and per-node TAPGNN objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_row_normalized_adjacency(adj, device=None, dtype=torch.float32):
    """Build sparse P while preserving the supplied matrix orientation.

    Row i aggregates column j. No symmetrization is applied. Input diagonal
    entries are replaced by one self-loop per node, so pre-existing loops
    cannot be counted twice. Nonnegative off-diagonal weights are preserved.
    No dense N-by-N intermediate is created for a sparse input.
    """
    if not torch.is_tensor(adj) or adj.ndim != 2 or adj.size(0) != adj.size(1):
        raise ValueError("adj must be a square torch adjacency matrix.")
    if adj.size(0) == 0:
        raise ValueError("The graph must contain at least one node.")
    adj = adj.detach().to(device=device, dtype=dtype)
    if adj.layout != torch.sparse_coo:
        adj = adj.to_sparse_coo()
    adj = adj.coalesce()
    rows, cols = adj.indices()
    values = adj.values()
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
        raise ValueError("Adjacency weights must be finite and nonnegative.")
    keep = (rows != cols) & (values != 0)
    nodes = torch.arange(adj.size(0), device=adj.device)
    rows = torch.cat((rows[keep], nodes))
    cols = torch.cat((cols[keep], nodes))
    values = torch.cat((values[keep], torch.ones_like(nodes, dtype=dtype)))
    degrees = torch.zeros(adj.size(0), device=adj.device, dtype=dtype)
    degrees.index_add_(0, rows, values)
    return torch.sparse_coo_tensor(
        torch.stack((rows, cols)), values / degrees[rows], adj.shape,
        device=adj.device, dtype=dtype,
    ).coalesce()


def normalize_axis(raw_axis, eps=1e-8):
    """Normalize to unit length, with a fixed unit-axis fallback near zero."""
    norms = torch.linalg.vector_norm(raw_axis, dim=-1, keepdim=True)
    normalized = raw_axis / norms.clamp_min(eps)
    fallback = torch.zeros_like(raw_axis)
    fallback[..., 0] = 1.0
    return torch.where(norms >= eps, normalized, fallback)


def build_propagation_state(states, classifier, logits_self):
    """Keep representation gradients; stop only the prediction-state branch."""
    with torch.no_grad():
        p0 = logits_self.softmax(dim=-1)
        predictions = [p0]
        for hidden in states[1:]:
            predictions.append(classifier(hidden).softmax(dim=-1) - p0)
    return torch.cat(list(states) + predictions, dim=-1)


def as_bool_mask(mask_or_index, num_nodes, device):
    """Accept a boolean mask or integer indices without guessing 0/1 masks."""
    values = torch.as_tensor(mask_or_index, device=device)
    if values.ndim != 1:
        raise ValueError("Select one split before constructing the predictor.")
    if values.dtype == torch.bool:
        if values.numel() != num_nodes:
            raise ValueError("A boolean mask must have N entries.")
        return values
    if values.is_floating_point() or values.is_complex():
        raise TypeError("Use a bool mask or integer node indices.")
    indices = values.long()
    if indices.numel() and bool(((indices < 0) | (indices >= num_nodes)).any()):
        raise IndexError("Node indices must be in [0, N).")
    mask = torch.zeros(num_nodes, device=device, dtype=torch.bool)
    mask[indices] = True
    return mask


def node_cross_entropy(logits, targets):
    """Compute -sum_c q_ic * log p_ic with a stable log_softmax.

    Integer labels are converted to one-hot q. An N-by-C probability target
    is also accepted; the main predictor uses only ground-truth hard labels.
    The output is per-node, with no reduction over the batch dimension.
    """
    if logits.ndim != 2 or logits.size(1) < 2:
        raise ValueError("Expected N-by-C logits with C >= 2.")
    if targets.ndim == 1:
        if targets.is_floating_point() or targets.is_complex():
            raise TypeError("Class-index targets must be integer tensors.")
        q = F.one_hot(targets.long(), num_classes=logits.size(1))
        q = q.to(device=logits.device, dtype=logits.dtype)
    elif targets.shape == logits.shape:
        q = targets.to(device=logits.device, dtype=logits.dtype)
        if not bool(torch.isfinite(q).all()) or bool((q < 0).any()):
            raise ValueError("Probability targets must be finite and nonnegative.")
        if not torch.allclose(q.sum(dim=-1), torch.ones_like(q[:, 0]), atol=1e-6):
            raise ValueError("Probability targets must sum to one per node.")
    else:
        raise ValueError("Targets must be N class indices or an N-by-C distribution.")
    if q.shape != logits.shape:
        raise ValueError("Logits and targets must have matching node counts.")
    return -(q * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def tapgnn_objective(logits, logits_self, labels, train_mask,
                    lambda_self=1.0, lambda_harm=1.0):
    """Compute all losses only on training nodes, before any averaging."""
    mask = as_bool_mask(train_mask, logits.size(0), logits.device)
    if not bool(mask.any()):
        raise ValueError("TAPGNN requires at least one training node.")
    if lambda_self < 0 or lambda_harm < 0:
        raise ValueError("Loss coefficients must be nonnegative.")
    labels = labels.to(logits.device)
    loss_plus = node_cross_entropy(logits[mask], labels[mask])
    loss_self = node_cross_entropy(logits_self[mask], labels[mask])
    excess = loss_plus - loss_self.detach()
    cls = loss_plus.mean()
    own = loss_self.mean()
    harm = excess.relu().mean()
    total = cls + lambda_self * own + lambda_harm * harm
    parts = {
        "cls": cls.detach(),
        "self": own.detach(),
        "harm": harm.detach(),
        "harm_fraction": (excess.detach() > 0).float().mean(),
    }
    return total, parts
