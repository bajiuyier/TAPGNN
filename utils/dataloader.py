import torch

from utils.functional import normalize
from utils.tools import get_npz_data, get_homophily

from torch_geometric.datasets import (
    Planetoid,
    Amazon,
    Coauthor,
    WikiCS,
    WikipediaNetwork,
    WebKB,
    Actor,
    AttributedGraphDataset,
    TUDataset,
    CitationFull,
    HeterophilousGraphDataset,
)
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import degree, to_undirected

import os
import json
from utils.datasplit import (
    get_split,
    get_default_split,
    canonical_split_name,
    DEFAULT_SPLIT_PROTOCOLS,
)
import numpy as np
import pickle
import urllib.request
from utils.social_benchmarks import (
    SOCIAL_BENCHMARKS,
    ensure_social_raw,
    prepare_social_benchmarks,
)


try:
    from torch_geometric.datasets import LINKXDataset
except ImportError:
    LINKXDataset = None


try:
    from ogb.nodeproppred import PygNodePropPredDataset
except ImportError:
    PygNodePropPredDataset = None


def _register_social_default_split(name, split_dir):
    """Support stored social split banks with older datasplit registries.

    Read only protocol metadata here. get_default_split still validates the
    actual NPZ checksum, label order, split fingerprints and disjointness.
    No masks are generated or downloaded.
    """
    name = canonical_split_name(name)
    if name not in SOCIAL_BENCHMARKS or name in DEFAULT_SPLIT_PROTOCOLS:
        return
    manifest_path = os.path.join(split_dir, 'manifest.json')
    with open(manifest_path, encoding='utf-8') as stream:
        manifest = json.load(stream)
    record = manifest.get('datasets', {}).get(name)
    if not isinstance(record, dict):
        raise ValueError(f'{name}: no split record in {manifest_path}.')
    num_nodes = SOCIAL_BENCHMARKS[name]['num_nodes']
    if record.get('num_nodes') != num_nodes or record.get('num_splits') != 10:
        raise ValueError(
            f'{name}: expected {num_nodes} original nodes and 10 stored '
            f'partitions in {manifest_path}.'
        )
    DEFAULT_SPLIT_PROTOCOLS[name] = dict(
        kind='local_author_masks', official=True, num_splits=10,
        num_nodes=num_nodes, ratios=None,
        reference=record.get('reference') or 'Fixed partitions recorded in manifest.json',
        source=record.get('source') or manifest_path,
    )


def _save_single_graph_dataset(dataset, data):
    """Save one PyG Data object across old and new PyG versions."""
    if hasattr(dataset, 'save'):
        dataset.save([data], dataset.processed_paths[0])
    else:
        torch.save(dataset.collate([data]), dataset.processed_paths[0])


def _load_single_graph_dataset(dataset):
    """Load one PyG Data object across old and new PyG versions."""
    if hasattr(dataset, 'load'):
        dataset.load(dataset.processed_paths[0])
    else:
        dataset.data, dataset.slices = torch.load(dataset.processed_paths[0])


def _make_stratified_masks(labels, num_splits=5, seed=12345):
    """
    Create deterministic 50/25/25 masks for datasets without public splits.

    Nodes with negative labels are regarded as unlabeled and are excluded
    from train/validation/test masks.
    """
    num_nodes = labels.numel()

    train_mask = torch.zeros(
        (num_nodes, num_splits),
        dtype=torch.bool,
    )
    val_mask = torch.zeros(
        (num_nodes, num_splits),
        dtype=torch.bool,
    )
    test_mask = torch.zeros(
        (num_nodes, num_splits),
        dtype=torch.bool,
    )

    # Only valid labels participate in supervised splits.
    valid_labels = labels[labels >= 0].unique(sorted=True)

    for split in range(num_splits):
        generator = torch.Generator().manual_seed(seed + split)

        for label in valid_labels:
            class_idx = torch.where(labels == label)[0]

            class_idx = class_idx[
                torch.randperm(
                    class_idx.numel(),
                    generator=generator,
                )
            ]

            train_end = int(0.50 * class_idx.numel())
            val_end = train_end + int(0.25 * class_idx.numel())

            train_mask[class_idx[:train_end], split] = True
            val_mask[class_idx[train_end:val_end], split] = True
            test_mask[class_idx[val_end:], split] = True

    return train_mask, val_mask, test_mask


def _make_index_mask(indices, num_nodes):
    """Convert an index tensor to a one-dimensional boolean mask."""
    mask = torch.zeros(
        num_nodes,
        dtype=torch.bool,
    )
    mask[torch.as_tensor(indices).view(-1).long()] = True
    return mask


def _make_even_quantile_labels(values, num_classes=5):
    """Discretize numerical values into equal-frequency classes.

    This follows the arxiv-year label construction used by the LINKX
    benchmark: publication years are divided by empirical quantiles.
    """
    values = np.asarray(values).reshape(-1)

    if values.size == 0:
        raise ValueError('Cannot create labels from an empty year array.')

    boundaries = np.quantile(
        values,
        np.linspace(
            0.0,
            1.0,
            num_classes + 1,
        )[1:-1],
    )

    labels = np.digitize(
        values,
        boundaries,
        right=False,
    )

    return torch.from_numpy(
        labels.astype(np.int64)
    )


class OGBArxivDataset:
    """PyG-compatible wrapper for ogbn-arxiv and arxiv-year.

    ``ogbn-arxiv`` uses the official OGB subject labels and official temporal
    split. ``arxiv-year`` reuses the same graph/features, replaces the labels
    with five publication-year quantile classes. Its five original LINKX
    partitions are loaded by get_default_split, not regenerated here.

    The citation graph remains directed for both datasets.
    """

    def __init__(
        self,
        root,
        name,
    ):
        if PygNodePropPredDataset is None:
            raise ImportError(
                'ogbn-arxiv and arxiv-year require the OGB package. '
                'Install it with: pip install ogb'
            )

        normalized_name = name.lower()

        if normalized_name == 'obgn-arxiv':
            normalized_name = 'ogbn-arxiv'

        if normalized_name not in [
            'ogbn-arxiv',
            'arxiv-year',
        ]:
            raise ValueError(
                f'Unsupported OGB arXiv dataset: {name}'
            )

        self.name = normalized_name
        self.root = root

        ogb_dataset = PygNodePropPredDataset(
            name='ogbn-arxiv',
            root=root,
        )
        source_data = ogb_dataset[0]

        x = source_data.x.to(torch.float)
        edge_index = source_data.edge_index.to(torch.long)
        num_nodes = x.shape[0]

        if normalized_name == 'ogbn-arxiv':
            y = source_data.y.view(-1).to(torch.long)
            split_indices = ogb_dataset.get_idx_split()

            train_mask = _make_index_mask(
                split_indices['train'],
                num_nodes,
            )
            val_mask = _make_index_mask(
                split_indices['valid'],
                num_nodes,
            )
            test_mask = _make_index_mask(
                split_indices['test'],
                num_nodes,
            )

            self.num_classes = int(
                y.max().item()
            ) + 1

        else:
            if not hasattr(source_data, 'node_year'):
                raise AttributeError(
                    'The downloaded ogbn-arxiv data does not contain '
                    'node_year, which is required for arxiv-year.'
                )

            node_year = (
                source_data.node_year
                .detach()
                .cpu()
                .numpy()
            )
            y = _make_even_quantile_labels(
                node_year,
                num_classes=5,
            )

            # utils.datasplit loads the original LINKX indices for default.
            # Percent splits remain available without downloading these masks.
            train_mask = val_mask = test_mask = None

            self.num_classes = 5

        self._data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        if hasattr(source_data, 'node_year'):
            self._data.node_year = source_data.node_year

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(
                'OGBArxivDataset contains only one graph.'
            )
        return self._data

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'(name={self.name})'
        )


class FilteredWikipediaNetwork(InMemoryDataset):
    """Leakage-filtered Chameleon/Squirrel datasets from Platonov et al."""

    url = (
        'https://github.com/yandex-research/heterophilous-graphs/'
        'raw/main/data'
    )

    def __init__(
        self,
        root,
        name,
        transform=None,
        pre_transform=None,
    ):
        self.name = name.lower().replace('-', '_')

        if self.name not in [
            'chameleon_filtered',
            'squirrel_filtered',
        ]:
            raise ValueError(
                f'Unsupported filtered Wikipedia dataset: {name}'
            )

        super().__init__(
            root,
            transform,
            pre_transform,
        )

        _load_single_graph_dataset(self)

    @property
    def raw_dir(self):
        return os.path.join(
            self.root,
            self.name,
            'raw',
        )

    @property
    def processed_dir(self):
        return os.path.join(
            self.root,
            self.name,
            'processed',
        )

    @property
    def raw_file_names(self):
        return f'{self.name}.npz'

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        download_url(
            f'{self.url}/{self.raw_file_names}',
            self.raw_dir,
        )

    def process(self):
        raw = np.load(
            self.raw_paths[0],
            'r',
        )

        x = torch.from_numpy(
            raw['node_features']
        ).to(torch.float)

        y = torch.from_numpy(
            raw['node_labels']
        ).to(torch.long)

        edge_index = torch.from_numpy(
            raw['edges']
        ).t().contiguous().to(torch.long)

        train_mask = torch.from_numpy(
            raw['train_masks']
        ).t().contiguous().bool()

        val_mask = torch.from_numpy(
            raw['val_masks']
        ).t().contiguous().bool()

        test_mask = torch.from_numpy(
            raw['test_masks']
        ).t().contiguous().bool()

        # The official filtered protocol keeps these Wikipedia graphs directed.
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        _save_single_graph_dataset(
            self,
            data,
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}'
            f'(name={self.name})'
        )


class TwitchGamerDataset(InMemoryDataset):
    """Twitch-Gamer mature-content node-classification dataset from SNAP."""

    url = 'https://snap.stanford.edu/data/twitch_gamers.zip'

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(
            root,
            transform,
            pre_transform,
        )

        _load_single_graph_dataset(self)

    @property
    def raw_dir(self):
        return os.path.join(
            self.root,
            'twitch_gamer',
            'raw',
        )

    @property
    def processed_dir(self):
        return os.path.join(
            self.root,
            'twitch_gamer',
            'processed',
        )

    @property
    def raw_file_names(self):
        return [
            'large_twitch_edges.csv',
            'large_twitch_features.csv',
        ]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        archive = download_url(
            self.url,
            self.raw_dir,
        )

        extract_zip(
            archive,
            self.raw_dir,
        )

        os.unlink(archive)

    def process(self):
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                'Twitch-Gamer requires pandas. '
                'Install it with: pip install pandas'
            ) from exc

        edge_path = os.path.join(
            self.raw_dir,
            'large_twitch_edges.csv',
        )

        feature_path = os.path.join(
            self.raw_dir,
            'large_twitch_features.csv',
        )

        edges = pd.read_csv(edge_path)
        nodes = pd.read_csv(feature_path)

        edge_index = torch.from_numpy(
            edges.iloc[:, :2].to_numpy()
        ).t().long()

        edge_index = to_undirected(
            edge_index,
            num_nodes=len(nodes),
        )

        y = torch.from_numpy(
            nodes['mature']
            .astype(np.int64)
            .to_numpy()
        ).long()

        features = nodes.drop(
            columns=[
                'mature',
                'numeric_id',
            ]
        ).copy()

        for column in [
            'created_at',
            'updated_at',
        ]:
            features[column] = (
                features[column]
                .astype(str)
                .str.replace(
                    '-',
                    '',
                    regex=False,
                )
                .astype(np.int64)
            )

        features['language'] = pd.factorize(
            features['language']
        )[0]

        x = torch.from_numpy(
            features.to_numpy(
                dtype=np.float32
            )
        )

        # Match the LINKX preprocessing and prevent date/view fields
        # from dominating the remaining user attributes.
        std = x.std(
            dim=0,
            keepdim=True,
        ).clamp_min(1e-12)

        x = (
            x
            - x.mean(
                dim=0,
                keepdim=True,
            )
        ) / std

        train_mask, val_mask, test_mask = (
            _make_stratified_masks(y)
        )

        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        _save_single_graph_dataset(
            self,
            data,
        )


class SocialAttributedGraphDataset(AttributedGraphDataset):
    """PyG's small social graphs, with verified downloads and original node IDs."""

    def download(self):
        ensure_social_raw(self.name, self.root)


class Dataset:
    """
    Dataset class.

    This class loads, preprocesses and splits various datasets.

    Parameters
    ----------
    data : str
        Dataset name.
    feat_norm : bool
        Whether to normalize node features.
    verbose : bool
        Whether to print statistics.
    path : str
        Dataset storage path.
    """

    def __init__(
        self,
        data,
        feat_norm=True,
        adj_norm=True,
        verbose=True,
        path=None,
        train_size=None,
        val_size=None,
        test_size=None,
        train_percent=None,
        val_percent=None,
        test_percent=None,
        train_examples_per_class=None,
        val_examples_per_class=None,
        test_examples_per_class=None,
        add_self_loop=True,
        split_type='default',
        from_npz=False,
        device='cuda:0',
        split_id=0,
        split_seed=12345,
        split_cache_dir=None,
        split_dir=None,
    ):
        # Keep the common misspelling working, while exposing the official
        # Open Graph Benchmark dataset name everywhere else.
        if data == 'obgn-arxiv':
            print(
                "Dataset name 'obgn-arxiv' is deprecated; "
                "using 'ogbn-arxiv'."
            )
            data = 'ogbn-arxiv'

        data = canonical_split_name(data)
        self.name = data
        self.path = os.path.join(
            os.getcwd(),
            path or './data',
        )

        self.device = torch.device(device)

        self.single_graph = True
        self.self_loop = add_self_loop
        self.split_type = split_type
        self.split_id = split_id
        self.split_seed = split_seed
        self.split_dir = split_dir or split_cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data_splits',
        )
        # Compatibility with older callers; this directory contains author masks.
        self.split_cache_dir = self.split_dir
        self.split_info = None
        self.num_splits = 1

        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        self.train_percent = train_percent
        self.val_percent = val_percent
        self.test_percent = test_percent

        self.train_examples_per_class = (
            train_examples_per_class
        )

        self.val_examples_per_class = (
            val_examples_per_class
        )

        self.test_examples_per_class = (
            test_examples_per_class
        )

        self.prepare_data(
            data,
            feat_norm,
            from_npz,
        )

        self.feats = self.feats.to(
            torch.float
        )

        self.adj = self.adj.coalesce()

        self.edge_index = (
            self.adj
                .coalesce()
                .indices()
        )

        if self.single_graph:
            self.split_data(verbose)
        else:
            self.split_graphs(verbose)

        # self.homophily = get_homophily(
        #     self.labels,
        #     self.adj.to_dense(),
        #     type='edge',
        #     fill=None,
        # )

        # ============================================================
        # Self-loop
        # ============================================================

        # By default, use the original adjacency.
        # This guarantees that adj_with_loop always exists even when
        # add_self_loop=False.
        self.adj_with_loop = self.adj
        self.edge_index_loop = self.edge_index

        if add_self_loop:
            num_nodes = self.adj.shape[0]

            loop_nodes = torch.arange(
                num_nodes,
                device=self.adj.device,
                dtype=torch.long,
            )

            loop_index = torch.stack(
                [
                    loop_nodes,
                    loop_nodes,
                ],
                dim=0,
            )

            loop_values = torch.ones(
                num_nodes,
                device=self.adj.device,
                dtype=self.adj.dtype,
            )

            self_loop_adj = torch.sparse_coo_tensor(
                loop_index,
                loop_values,
                size=self.adj.shape,
                device=self.adj.device,
                dtype=self.adj.dtype,
            ).coalesce()

            self.adj_with_loop = (
                    self.adj
                    + self_loop_adj
            ).coalesce()

            self.edge_index_loop = (
                self.adj_with_loop
                    .coalesce()
                    .indices()
            )

        # ============================================================
        # Adjacency normalization
        # ============================================================

        if adj_norm:
            self.normalized_adj = normalize(
                self.adj_with_loop,
                add_loop=False,
            )

            self.normalized_adj = (
                self.normalized_adj
                    .coalesce()
            )

            self.edge_index_normliazed = (
                self.normalized_adj
                    .coalesce()
                    .indices()
            )

        else:
            # Keep these attributes available for compatibility.
            self.normalized_adj = self.adj_with_loop

            self.edge_index_normliazed = (
                self.normalized_adj
                    .coalesce()
                    .indices()
            )

    def prepare_data(
        self,
        ds_name,
        feat_norm,
        from_npz,
    ):
        """
        Load the requested dataset.

        The results are stored in:
            self.feats
            self.adj
            self.labels
        """

        if from_npz:
            adj, features, labels = get_npz_data(
                self.path + ds_name + '.npz',
                self_loop=self.self_loop,
            )

            self.adj = (
                adj
                .to(self.device)
                .coalesce()
            )

            self.feats = features.to(
                self.device
            )

            self.labels = torch.tensor(
                labels,
                dtype=torch.int64,
            ).to(self.device)

            self.n_nodes = self.feats.shape[0]
            self.dim_feats = self.feats.shape[1]

            self.n_edges = (
                self.adj.indices().shape[1] / 2
            )

            # Negative labels are treated as unlabeled.
            valid_labels = labels[
                labels >= 0
            ]

            if len(valid_labels) == 0:
                raise ValueError(
                    f'Dataset {ds_name} contains no valid labels.'
                )

            self.n_classes = (
                int(valid_labels.max()) + 1
            )

            self.clean_label = (
                self.labels.clone()
            )

            if feat_norm:
                self.feats = normalize(
                    self.feats,
                    style='row',
                )

        elif ds_name in [
            'cora',
            'pubmed',
            'citeseer',
            'amazoncom',
            'amazonpho',
            'coauthorcs',
            'coauthorph',
            'blogcatalog',
            'flickr',
            'wikics',
            'cornell',
            'texas',
            'wisconsin',
            'dblp',
            'amazon-ratings',
            'roman-empire',
            'minesweeper',
            'tolokers',
            'questions',
            'actor',
            'chameleon',
            'squirrel',
            'chameleon-f',
            'chameleon-filtered',
            'squirrel-f',
            'squirrel-filtered',
            'penn94',
            'genius',
            'twitch-gamer',
            'twitch-gamers',
            'ogbn-arxiv',
            'arxiv-year',
        ]:
            self.data_raw = pyg_load_dataset(
                ds_name,
                path=self.path,
            )

            self.g = self.data_raw[0]

            # Unnormalized features.
            self.feats = self.g.x

            if ds_name in SOCIAL_BENCHMARKS:
                # PyG may return torch_sparse.SparseTensor, which has neither
                # Tensor.shape nor Tensor.layout. Convert before inspecting it.
                if not isinstance(self.feats, torch.Tensor):
                    self.feats = self.feats.to_dense()
                elif self.feats.layout != torch.strided:
                    self.feats = self.feats.to_dense()

                meta = SOCIAL_BENCHMARKS[ds_name]
                if tuple(self.feats.shape) != (meta['num_nodes'], meta['num_features']):
                    raise ValueError(
                        f'{ds_name}: expected the original attributed graph with '
                        f'{meta["num_nodes"]} nodes and {meta["num_features"]} features.'
                    )
                if self.g.y.ndim != 1 or self.g.y.numel() != meta['num_nodes']:
                    raise ValueError(f'{ds_name}: expected original single-label node targets.')
                self.g.num_nodes = meta['num_nodes']

            self.n_nodes = (
                self.feats.shape[0]
            )

            self.dim_feats = (
                self.feats.shape[1]
            )

            self.labels = self.g.y

            self.adj = torch.sparse_coo_tensor(
                self.g.edge_index,
                torch.ones(
                    self.g.edge_index.shape[1]
                ),
                [
                    self.n_nodes,
                    self.n_nodes,
                ],
            )

            if self.g.is_directed():
                self.n_edges = self.g.num_edges
            else:
                self.n_edges = (
                    self.g.num_edges / 2
                )

            self.n_classes = (
                self.data_raw.num_classes
            )

            self.feats = self.feats.to(
                self.device
            )

            self.labels = self.labels.to(
                self.device
            )

            self.adj = self.adj.to(
                self.device
            )

            if feat_norm:
                self.feats = normalize(
                    self.feats,
                    style='row',
                )

        else:
            raise ValueError(
                f'Unsupported dataset: {ds_name}'
            )

        self.adj = self.adj.coalesce()

        row = self.adj.indices()[0]

        d = degree(
            row,
            self.n_nodes,
        )

        self.ave_degree = float(
            torch.mean(d)
        )

        self.clean_label = (
            self.labels.clone()
        )

        # Additional label statistics.
        labels_cpu = (
            self.labels
            .detach()
            .cpu()
            .reshape(-1)
        )

        self.n_labeled = int(
            (labels_cpu >= 0)
            .sum()
            .item()
        )

        self.n_unlabeled = (
            self.n_nodes
            - self.n_labeled
        )

        print(
            """----Data statistics------'
                Name: %s
                #Nodes %d
                #Edges %d
                #Classes %d
                #Labeled nodes %d
                #Unlabeled nodes %d
                #Ave_degree %.2f"""
            % (
                self.name,
                self.n_nodes,
                self.n_edges,
                self.n_classes,
                self.n_labeled,
                self.n_unlabeled,
                self.ave_degree,
            )
        )

    def split_data(
        self,
        verbose=True,
    ):
        """
        Split nodes into train/validation/test subsets.

        Three split types are supported:

        1. default
           Read downloaded author masks from split_dir and select split_id.
           No random masks are generated and no split files are downloaded here.

        2. percent
           Split labeled nodes according to specified proportions.

        3. samples_per_class
           Sample a specified number of labeled nodes per class.

        Nodes with label < 0 are regarded as unlabeled. They remain in the
        graph and participate in message passing, but they never enter the
        train/validation/test supervised subsets.
        """

        self.train_masks = None
        self.val_masks = None
        self.test_masks = None
        self.split_info = None
        self.num_splits = 1

        # -------------------------------------------------------------
        # Valid labeled nodes
        # -------------------------------------------------------------
        labels_np = (
            self.labels
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        valid_indices = np.where(
            labels_np >= 0
        )[0]

        valid_labels = labels_np[
            valid_indices
        ]

        n_labeled = len(
            valid_indices
        )

        if n_labeled == 0:
            raise ValueError(
                f'Dataset {self.name} contains no labeled nodes.'
            )

        # -------------------------------------------------------------
        # Default/public split
        # -------------------------------------------------------------
        if self.split_type == 'default':
            train_type = 'default'
            val_type = 'default'
            test_type = 'default'

            assert self.name in [
                'cora',
                'citeseer',
                'pubmed',
                'blogcatalog',
                'flickr',
                'roman-empire',
                'amazon-ratings',
                'amazoncom',
                'chameleon',
                'squirrel',
                'minesweeper',
                'tolokers',
                'questions',
                'wikics',
                'airport',
                'actor',
                'texas',
                'cornell',
                'wisconsin',
                'chameleon-f',
                'chameleon-filtered',
                'squirrel-f',
                'squirrel-filtered',
                'penn94',
                'genius',
                'twitch-gamer',
                'twitch-gamers',
                'ogbn-arxiv',
                'arxiv-year',
                'brazil',
                'europe',
                'facehook',
            ], (
                'This dataset has no public splits.'
            )

            _register_social_default_split(self.name, self.split_dir)

            # Registered author partitions, including BlogCatalog/Flickr.
            if canonical_split_name(self.name) in DEFAULT_SPLIT_PROTOCOLS:
                (
                    train_indices,
                    val_indices,
                    test_indices,
                    self.split_info,
                ) = get_default_split(
                    dataset_name=self.name,
                    labels=self.labels,
                    data=getattr(self, 'g', None),
                    split_id=self.split_id,
                    split_dir=self.split_dir,
                    return_info=True,
                )
                self.num_splits = self.split_info['num_splits']
                description = 'author split'
                train_type = val_type = test_type = description

            # Legacy Airport split support. Social graphs always use the NPZ bank.
            elif self.name in [
                'airport',
            ]:
                def load_obj(file_name):
                    with open(
                        file_name,
                        'rb',
                    ) as f:
                        return pickle.load(f)

                def download(name):
                    url = (
                        'https://github.com/'
                        'zhao-tong/GAug/raw/master/'
                        'data/graphs/'
                    )

                    try:
                        print(
                            'Downloading',
                            url + name,
                        )

                        urllib.request.urlretrieve(
                            url + name,
                            os.path.join(
                                self.path,
                                self.name,
                                name,
                            ),
                        )

                        print('Done!')

                    except Exception:
                        raise Exception(
                            'Download failed! Make sure you '
                            'have a stable Internet connection '
                            'and enter the correct dataset name.'
                        )

                split_file = (
                    self.name
                    + '_tvt_nids.pkl'
                )

                split_path = os.path.join(
                    self.path,
                    self.name,
                    split_file,
                )

                if not os.path.exists(
                    split_path
                ):
                    download(split_file)

                (
                    train_indices,
                    val_indices,
                    test_indices,
                ) = load_obj(split_path)

            # Datasets with masks stored in PyG Data.
            elif self.name in [
                'roman-empire',
                'amazon-ratings',
                'minesweeper',
                'tolokers',
                'questions',
                'wikics',
                'actor',
                'texas',
                'cornell',
                'wisconsin',
                'chameleon',
                'squirrel',
                'chameleon-f',
                'chameleon-filtered',
                'squirrel-f',
                'squirrel-filtered',
                'penn94',
                'twitch-gamer',
                'twitch-gamers',
                'ogbn-arxiv',
                'arxiv-year',
            ]:
                train_mask = (
                    self.g.train_mask
                )

                val_mask = (
                    self.g.val_mask
                )

                test_mask = (
                    self.g.test_mask
                )

                # Compatibility path for other datasets such as WikiCS.
                # Registered benchmarks use get_default_split above.
                for mask in (train_mask, val_mask, test_mask):
                    count = mask.shape[1] if mask.dim() == 2 else 1
                    if not 0 <= self.split_id < count:
                        raise ValueError(f'{self.name}: invalid split_id={self.split_id}.')
                if train_mask.dim() == 2:
                    train_mask = (
                        train_mask[:, self.split_id]
                    )

                if val_mask.dim() == 2:
                    val_mask = (
                        val_mask[:, self.split_id]
                    )

                if test_mask.dim() == 2:
                    test_mask = (
                        test_mask[:, self.split_id]
                    )

                train_indices = (
                    torch.nonzero(
                        train_mask,
                        as_tuple=False,
                    )
                    .view(-1)
                    .cpu()
                    .numpy()
                )

                val_indices = (
                    torch.nonzero(
                        val_mask,
                        as_tuple=False,
                    )
                    .view(-1)
                    .cpu()
                    .numpy()
                )

                test_indices = (
                    torch.nonzero(
                        test_mask,
                        as_tuple=False,
                    )
                    .view(-1)
                    .cpu()
                    .numpy()
                )

            else:
                raise ValueError(
                    f'{self.name}: no default split handler was selected. '
                    'Check utils.datasplit.DEFAULT_SPLIT_PROTOCOLS and split_dir.'
                )

        # -------------------------------------------------------------
        # Percentage split
        # -------------------------------------------------------------
        elif self.split_type == 'percent':

            if self.train_size is not None:
                train_size = (
                    self.train_size
                )

                train_type = 'specified'

            elif self.train_percent is not None:
                train_size = int(
                    n_labeled
                    * self.train_percent
                )

                train_type = (
                    str(
                        self.train_percent
                        * 100
                    )
                    + ' % of labeled nodes'
                )

            else:
                raise ValueError(
                    'Split error: split_type=percent. '
                    'train_size or train_percent '
                    'must be configured.'
                )

            if self.val_size is not None:
                val_size = (
                    self.val_size
                )

                val_type = 'specified'

            elif self.val_percent is not None:
                val_size = int(
                    n_labeled
                    * self.val_percent
                )

                val_type = (
                    str(
                        self.val_percent
                        * 100
                    )
                    + ' % of labeled nodes'
                )

            else:
                raise ValueError(
                    'Split error: split_type=percent. '
                    'val_size or val_percent '
                    'must be configured.'
                )

            if self.test_size is not None:
                test_size = (
                    self.test_size
                )

                test_type = 'specified'

            elif self.test_percent is not None:
                test_size = int(
                    n_labeled
                    * self.test_percent
                )

                test_type = (
                    str(
                        self.test_percent
                        * 100
                    )
                    + ' % of labeled nodes'
                )

            else:
                test_size = None
                test_type = 'remaining'

            if train_size < 0:
                raise ValueError(
                    'train_size must be non-negative.'
                )

            if val_size < 0:
                raise ValueError(
                    'val_size must be non-negative.'
                )

            if (
                test_size is not None
                and test_size < 0
            ):
                raise ValueError(
                    'test_size must be non-negative.'
                )

            requested_size = (
                train_size
                + val_size
                + (
                    0
                    if test_size is None
                    else test_size
                )
            )

            if (
                test_size is not None
                and requested_size > n_labeled
            ):
                raise ValueError(
                    f'Requested split size '
                    f'({requested_size}) exceeds '
                    f'the number of labeled nodes '
                    f'({n_labeled}) in dataset '
                    f'{self.name}.'
                )

            # IMPORTANT:
            # get_split sees labeled nodes only.
            (
                train_local,
                val_local,
                test_local,
            ) = get_split(
                valid_labels,
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
            )

            # Convert local split indices back to original graph indices.
            train_indices = valid_indices[
                np.asarray(
                    train_local,
                    dtype=np.int64,
                ).reshape(-1)
            ]

            val_indices = valid_indices[
                np.asarray(
                    val_local,
                    dtype=np.int64,
                ).reshape(-1)
            ]

            test_indices = valid_indices[
                np.asarray(
                    test_local,
                    dtype=np.int64,
                ).reshape(-1)
            ]

        # -------------------------------------------------------------
        # Samples-per-class split
        # -------------------------------------------------------------
        elif (
            self.split_type
            == 'samples_per_class'
        ):
            train_size = None
            val_size = None
            test_size = None

            if (
                self.train_examples_per_class
                is not None
            ):
                train_examples_per_class = (
                    self.train_examples_per_class
                )

                train_type = (
                    str(
                        self.train_examples_per_class
                    )
                    + ' labeled nodes per class'
                )

            elif self.train_size is not None:
                train_examples_per_class = None
                train_size = self.train_size
                train_type = 'specified'

            else:
                raise ValueError(
                    'Split error: '
                    'split_type=samples_per_class. '
                    'train_examples_per_class or '
                    'train_size must be configured.'
                )

            if (
                self.val_examples_per_class
                is not None
            ):
                val_examples_per_class = (
                    self.val_examples_per_class
                )

                val_type = (
                    str(
                        self.val_examples_per_class
                    )
                    + ' labeled nodes per class'
                )

            elif self.val_size is not None:
                val_examples_per_class = None
                val_size = self.val_size
                val_type = 'specified'

            else:
                raise ValueError(
                    'Split error: '
                    'split_type=samples_per_class. '
                    'val_examples_per_class or '
                    'val_size must be configured.'
                )

            if (
                self.test_examples_per_class
                is not None
            ):
                test_examples_per_class = (
                    self.test_examples_per_class
                )

                test_type = (
                    str(
                        self.test_examples_per_class
                    )
                    + ' labeled nodes per class'
                )

            elif self.test_size is not None:
                test_examples_per_class = None
                test_size = self.test_size
                test_type = 'specified'

            else:
                test_examples_per_class = None
                test_size = None
                test_type = 'remaining'

            (
                train_local,
                val_local,
                test_local,
            ) = get_split(
                valid_labels,
                train_examples_per_class=(
                    train_examples_per_class
                ),
                val_examples_per_class=(
                    val_examples_per_class
                ),
                test_examples_per_class=(
                    test_examples_per_class
                ),
                train_size=train_size,
                val_size=val_size,
                test_size=test_size,
            )

            train_indices = valid_indices[
                np.asarray(
                    train_local,
                    dtype=np.int64,
                ).reshape(-1)
            ]

            val_indices = valid_indices[
                np.asarray(
                    val_local,
                    dtype=np.int64,
                ).reshape(-1)
            ]

            test_indices = valid_indices[
                np.asarray(
                    test_local,
                    dtype=np.int64,
                ).reshape(-1)
            ]

        else:
            raise NotImplementedError(
                'Split type '
                + self.split_type
                + ' is not implemented.'
            )

        # -------------------------------------------------------------
        # Normalize index representation
        # -------------------------------------------------------------
        train_indices = np.asarray(
            train_indices,
            dtype=np.int64,
        ).reshape(-1)

        val_indices = np.asarray(
            val_indices,
            dtype=np.int64,
        ).reshape(-1)

        test_indices = np.asarray(
            test_indices,
            dtype=np.int64,
        ).reshape(-1)

        # -------------------------------------------------------------
        # Remove negative-label nodes from public/default splits as well.
        #
        # Normally public splits should already be valid. This is an
        # additional safety layer for datasets such as Penn94.
        # -------------------------------------------------------------
        train_indices = train_indices[
            labels_np[train_indices] >= 0
        ]

        val_indices = val_indices[
            labels_np[val_indices] >= 0
        ]

        test_indices = test_indices[
            labels_np[test_indices] >= 0
        ]

        # -------------------------------------------------------------
        # Safety checks
        # -------------------------------------------------------------
        if len(train_indices) == 0:
            raise ValueError(
                f'Training split for {self.name} is empty.'
            )

        if len(val_indices) == 0:
            raise ValueError(
                f'Validation split for {self.name} is empty.'
            )

        if len(test_indices) == 0:
            raise ValueError(
                f'Test split for {self.name} is empty.'
            )

        assert np.all(
            labels_np[train_indices] >= 0
        ), (
            'Negative labels found in '
            'the training split.'
        )

        assert np.all(
            labels_np[val_indices] >= 0
        ), (
            'Negative labels found in '
            'the validation split.'
        )

        assert np.all(
            labels_np[test_indices] >= 0
        ), (
            'Negative labels found in '
            'the test split.'
        )

        assert np.all(
            labels_np[train_indices]
            < self.n_classes
        ), (
            'Training labels exceed '
            'the number of classes.'
        )

        assert np.all(
            labels_np[val_indices]
            < self.n_classes
        ), (
            'Validation labels exceed '
            'the number of classes.'
        )

        assert np.all(
            labels_np[test_indices]
            < self.n_classes
        ), (
            'Test labels exceed '
            'the number of classes.'
        )

        # Ensure that train/val/test do not overlap.
        train_set = set(
            train_indices.tolist()
        )

        val_set = set(
            val_indices.tolist()
        )

        test_set = set(
            test_indices.tolist()
        )

        if train_set & val_set:
            raise RuntimeError(
                'Train and validation splits overlap.'
            )

        if train_set & test_set:
            raise RuntimeError(
                'Train and test splits overlap.'
            )

        if val_set & test_set:
            raise RuntimeError(
                'Validation and test splits overlap.'
            )

        # -------------------------------------------------------------
        # Store NumPy indices
        # -------------------------------------------------------------
        self.train_masks = train_indices
        self.val_masks = val_indices
        self.test_masks = test_indices

        # -------------------------------------------------------------
        # Unlabeled / unused nodes
        #
        # This contains:
        # 1. nodes whose original label is -1;
        # 2. valid nodes not selected into train/val/test.
        #
        # They remain in the graph for message passing.
        # -------------------------------------------------------------
        all_idx = np.arange(
            self.n_nodes
        )

        used_idx = np.unique(
            np.concatenate(
                [
                    self.train_masks,
                    self.val_masks,
                    self.test_masks,
                ]
            )
        )

        unlabel_idx = np.setdiff1d(
            all_idx,
            used_idx,
            assume_unique=True,
        )

        self.unlabel_masks = (
            unlabel_idx
        )

        # -------------------------------------------------------------
        # Tensor indices
        # -------------------------------------------------------------
        self.idx_unlabel = (
            torch.from_numpy(
                unlabel_idx
            )
            .long()
            .to(self.device)
        )

        self.idx_train = (
            torch.from_numpy(
                self.train_masks
            )
            .long()
            .to(self.device)
        )

        self.idx_val = (
            torch.from_numpy(
                self.val_masks
            )
            .long()
            .to(self.device)
        )

        self.idx_test = (
            torch.from_numpy(
                self.test_masks
            )
            .long()
            .to(self.device)
        )

        # -------------------------------------------------------------
        # Boolean masks
        # -------------------------------------------------------------
        num_nodes = self.n_nodes

        self.mask_train = torch.zeros(
            num_nodes,
            dtype=torch.bool,
            device=self.device,
        )

        self.mask_val = torch.zeros(
            num_nodes,
            dtype=torch.bool,
            device=self.device,
        )

        self.mask_test = torch.zeros(
            num_nodes,
            dtype=torch.bool,
            device=self.device,
        )

        self.mask_unlabel = torch.zeros(
            num_nodes,
            dtype=torch.bool,
            device=self.device,
        )

        self.mask_train[
            self.idx_train
        ] = True

        self.mask_val[
            self.idx_val
        ] = True

        self.mask_test[
            self.idx_test
        ] = True

        self.mask_unlabel[
            self.idx_unlabel
        ] = True

        # -------------------------------------------------------------
        # Final runtime safety check
        # -------------------------------------------------------------
        assert torch.all(
            self.labels[
                self.mask_train
            ] >= 0
        ), (
            'Invalid negative label found '
            'in mask_train.'
        )

        assert torch.all(
            self.labels[
                self.mask_val
            ] >= 0
        ), (
            'Invalid negative label found '
            'in mask_val.'
        )

        assert torch.all(
            self.labels[
                self.mask_test
            ] >= 0
        ), (
            'Invalid negative label found '
            'in mask_test.'
        )

        assert torch.all(
            self.labels[
                self.mask_train
            ] < self.n_classes
        )

        assert torch.all(
            self.labels[
                self.mask_val
            ] < self.n_classes
        )

        assert torch.all(
            self.labels[
                self.mask_test
            ] < self.n_classes
        )

        # -------------------------------------------------------------
        # Statistics
        # -------------------------------------------------------------
        if verbose:
            if self.split_info is not None:
                info = self.split_info
                percentages = '/'.join(f'{100 * ratio:.2f}' for ratio in info['ratios'])
                print(
                    f'[Default split] {info["dataset"]} '
                    f'id={info["split_id"]}/{info["num_splits"] - 1} '
                    f'official={info["official"]} '
                    f'train/val/test={percentages}% '
                    f'hash={info["fingerprint"][:16]}'
                )
                print(f'[Split source] {info["reference"]}: {info["source"]}')
            print(
                """----Split statistics------'
                #Labeled nodes %d
                #Unlabeled nodes %d
                #Train samples %d (%s)
                #Val samples %d (%s)
                #Test samples %d (%s)
                #Unused/unlabeled graph nodes %d"""
                % (
                    n_labeled,
                    self.n_nodes - n_labeled,
                    len(self.train_masks),
                    train_type,
                    len(self.val_masks),
                    val_type,
                    len(self.test_masks),
                    test_type,
                    len(self.unlabel_masks),
                )
            )

    def set_split(self, split_id, verbose=True):
        """Select another default partition before constructing a predictor.

        Changing training seed alone never switches public masks. Returns
        self for convenience, and keeps graph/feature preprocessing unchanged.
        """
        if self.split_type != 'default':
            raise ValueError('set_split is for split_type=default only.')
        previous_id = self.split_id
        self.split_id = split_id
        try:
            self.split_data(verbose=verbose)
        except Exception:
            self.split_id = previous_id
            # Restore a usable partition if validation of the requested id failed.
            self.split_data(verbose=False)
            raise
        return self


def pyg_load_dataset(
    name,
    path='./data/',
):
    dic = {
        'cora': 'Cora',
        'citeseer': 'CiteSeer',
        'pubmed': 'PubMed',
        'amazoncom': 'Computers',
        'amazonpho': 'Photo',
        'coauthorcs': 'CS',
        'coauthorph': 'Physics',
        'wikics': 'WikiCS',
        'chameleon': 'Chameleon',
        'squirrel': 'Squirrel',
        'cornell': 'Cornell',
        'texas': 'Texas',
        'wisconsin': 'Wisconsin',
        'actor': 'Actor',
        'blogcatalog': 'blogcatalog',
        'flickr': 'flickr',
        'amazon-ratings': 'Amazon-ratings',
        'roman-empire': 'Roman-empire',
        'minesweeper': 'Minesweeper',
        'tolokers': 'Tolokers',
        'questions': 'Questions',
        'chameleon-f': 'Chameleon-filtered',
        'chameleon-filtered': 'Chameleon-filtered',
        'squirrel-f': 'Squirrel-filtered',
        'squirrel-filtered': 'Squirrel-filtered',
        'penn94': 'Penn94',
        'genius': 'Genius',
        'twitch-gamer': 'Twitch-gamer',
        'twitch-gamers': 'Twitch-gamer',
        'ogbn-arxiv': 'ogbn-arxiv',
        'obgn-arxiv': 'ogbn-arxiv',
        'arxiv-year': 'arxiv-year',
    }

    if name in dic:
        name = dic[name]

    if name in [
        'Cora',
        'CiteSeer',
        'PubMed',
    ]:
        dataset = Planetoid(
            root=path,
            name=name,
        )

    elif name in [
        'Computers',
        'Photo',
    ]:
        dataset = Amazon(
            root=path,
            name=name,
        )

    elif name in [
        'CS',
        'Physics',
    ]:
        dataset = Coauthor(
            root=path,
            name=name,
        )

    elif name == 'WikiCS':
        dataset = WikiCS(
            root=os.path.join(
                path,
                name,
            )
        )

    elif name in [
        'Chameleon',
        'Squirrel',
        'Crocodile',
    ]:
        dataset = WikipediaNetwork(
            root=path,
            name=name,
        )

    elif name in [
        'Chameleon-filtered',
        'Squirrel-filtered',
    ]:
        dataset = FilteredWikipediaNetwork(
            root=path,
            name=name,
        )

    elif name in [
        'Cornell',
        'Texas',
        'Wisconsin',
    ]:
        dataset = WebKB(
            root=path,
            name=name,
        )

    elif name == 'Actor':
        dataset = Actor(
            root=os.path.join(
                path,
                name,
            )
        )

    elif name in [
        'blogcatalog',
        'flickr',
    ]:
        dataset = SocialAttributedGraphDataset(
            root=path,
            name=name,
        )

    elif name in [
        'Amazon-ratings',
        'Roman-empire',
        'Minesweeper',
        'Tolokers',
        'Questions',
    ]:
        dataset = HeterophilousGraphDataset(
            root=path,
            name=name,
        )

    elif name in ['Penn94', 'Genius']:
        if LINKXDataset is None:
            raise ImportError(
                'Penn94/Genius requires a PyG version '
                'that provides LINKXDataset. '
                'Please upgrade torch-geometric.'
            )

        dataset = LINKXDataset(
            root=path,
            name=name.lower(),
        )

    elif name == 'Twitch-gamer':
        dataset = TwitchGamerDataset(
            root=path,
        )

    elif name in [
        'ogbn-arxiv',
        'arxiv-year',
    ]:
        dataset = OGBArxivDataset(
            root=path,
            name=name,
        )

    elif name == 'dblp':
        dataset = CitationFull(
            root=path,
            name=name,
        )

    else:
        dataset = TUDataset(
            root=path,
            name=name,
        )

    return dataset


# Datasets downloaded by the standalone preparation entry point.
RECOMMENDED_HETEROPHILY_DATASETS = [
    'blogcatalog',
    'flickr',
    'roman-empire',
    'amazon-ratings',
    'minesweeper',
    'tolokers',
    'questions',
    'chameleon-f',
    'squirrel-f',
    'actor',
    'cora',
    'citeseer',
    'pubmed',
    'texas',
    'cornell',
    'wisconsin',
    'penn94',
    'genius',
    'twitch-gamer',
    'ogbn-arxiv',
    'arxiv-year',
    'chameleon',
    'squirrel'
]


def download_recommended_datasets(
    path='./data',
    datasets=None,
    split_dir=None,
    prepare_splits=True,
):
    """
    Download and process selected or all recommended datasets.

    Existing downloaded files are reused automatically. For BlogCatalog and
    Flickr, also export the 10 author masks and merge manifest.json by default.
    """

    os.makedirs(
        path,
        exist_ok=True,
    )

    loaded = {}

    for name in (
        RECOMMENDED_HETEROPHILY_DATASETS if datasets is None else datasets
    ):
        print(
            f'\nDownloading/preparing '
            f'dataset: {name}'
        )

        loaded[name] = pyg_load_dataset(
            name,
            path=path,
        )

        graph = loaded[name][0]

        print(
            f'Finished {name}: '
            f'{graph.num_nodes} nodes, '
            f'{graph.num_edges} stored edges'
        )
        if prepare_splits and name in SOCIAL_BENCHMARKS:
            prepare_social_benchmarks(
                names=(name,), data_path=path, split_dir=split_dir,
            )

    return loaded


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    parser = argparse.ArgumentParser(description='Download datasets and the social graph author splits.')
    parser.add_argument('--datasets', nargs='+', default=RECOMMENDED_HETEROPHILY_DATASETS)
    parser.add_argument('--data-path', default=str(PROJECT_ROOT / 'dataset'))
    parser.add_argument('--split-dir', default=str(PROJECT_ROOT / 'data_splits'))
    parser.add_argument('--skip-splits', action='store_true')
    args = parser.parse_args()
    print(f'Dataset directory: {args.data_path}')
    download_recommended_datasets(
        path=args.data_path, datasets=args.datasets, split_dir=args.split_dir,
        prepare_splits=not args.skip_splits,
    )


