"""
Functions for graph structure fusion.
"""


class Replace:
    """
    Directly replace the original adjacency matrix with the new adjacency matrix.
    """

    def __init__(self):
        pass

    def __call__(self, adj, raw_adj):
        """
        Parameters
        ----------
        adj : torch.Tensor
            New graph structure.
        raw_adj : torch.Tensor
            Original graph structure.

        Returns
        -------
        fused_adj : torch.Tensor
            The fused adjacency matrix, which is equal to adj.
        """
        fused_adj = adj
        return fused_adj


class Interpolate:
    """
    Linearly combine the new graph structure and the original graph structure.

    Parameters
    ----------
    lamb1 : float
        Weight assigned to the new graph structure.
    lamb2 : float, optional
        Weight assigned to the original graph structure. If None, it is set to
        1 - lamb1.
    """

    def __init__(self, lamb1, lamb2=None):
        self.lamb1 = lamb1
        self.lamb2 = lamb2

        if lamb2 is None:
            self.lamb2 = 1 - self.lamb1

    def __call__(self, adj, raw_adj):
        """
        Parameters
        ----------
        adj : torch.Tensor
            New graph structure.
        raw_adj : torch.Tensor
            Original graph structure.

        Returns
        -------
        fused_adj : torch.Tensor
            The linearly fused adjacency matrix.
        """
        fused_adj = self.lamb1 * adj + self.lamb2 * raw_adj
        return fused_adj


class Multiply:
    """
    Fuse the new graph structure and the original graph structure by element-wise multiplication.
    """

    def __init__(self):
        pass

    def __call__(self, adj, raw_adj):
        """
        Parameters
        ----------
        adj : torch.Tensor
            New graph structure.
        raw_adj : torch.Tensor
            Original graph structure.

        Returns
        -------
        fused_adj : torch.Tensor
            The element-wise product of adj and raw_adj.
        """
        return adj * raw_adj