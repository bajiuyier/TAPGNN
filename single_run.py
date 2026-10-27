"""Run TAPGNN with percentage splits or externally supplied official splits.

For ``split_type: office_split`` (the spelling requested by the project), the
runner uses the same ``Dataset(split_type="default", split_id, split_dir)``
entry point as the hyperparameter-search code.  These are author-released fixed
random partitions rather than mutually exclusive cross-validation folds.  The
split source, citation, reference URL, file, manifest and protocol fingerprint
are printed before training.  Test accuracy and ROC-AUC are calculated from
the same restored checkpoint for every partition.
"""

import argparse
import gc
import hashlib
import json
import os
import re
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from predictor import *
from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed

warnings.filterwarnings("ignore")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


PREDICTOR_REGISTRY = {
    "tapgnn": TAPGNNPredictor,
}

OFFICE_SPLIT_TYPES = {"office_split", "official_split"}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="genius",
        help="Dataset name.",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="tapgnn",
        choices=list(PREDICTOR_REGISTRY.keys()),
        help="Method name.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Training device, e.g. cuda:0 or cpu.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help=(
            "Number of runs. In office_split mode, this limits the number of "
            "selected folds; by default all configured folds are run."
        ),
    )

    parser.add_argument(
        "--split-id",
        type=int,
        default=None,
        help="Run only this zero-based official split (for example, 0).",
    )

    return parser.parse_args()


def get_seed_from_conf(conf):
    if "seed" in conf.training:
        return int(conf.training["seed"])
    return 42


def is_office_split(conf):
    split_type = str(conf.dataset.get("split_type", "percent")).lower()
    return split_type in OFFICE_SPLIT_TYPES


def get_office_split_options(conf):
    """Return a normalized office_split configuration dictionary."""
    options = conf.dataset.get("office_split", {})
    if options is None or options is True:
        return {}
    if options is False:
        raise ValueError(
            "dataset.office_split is false, but dataset.split_type requests "
            "office_split."
        )
    if isinstance(options, str):
        return {"path": options}
    if not hasattr(options, "get"):
        raise TypeError(
            "dataset.office_split must be a mapping, a path string, true, or null."
        )
    return options


def resolve_office_split_path(dataset_name, conf):
    options = get_office_split_options(conf)
    path_value = options.get("path")
    if path_value is None:
        path = resolve_office_split_dir(dataset_name, conf) / f"{dataset_name}.npz"
    else:
        path = Path(str(path_value).format(dataset=dataset_name))
    if path.is_dir():
        path = path / f"{dataset_name}.npz"
    return path


def resolve_office_split_dir(dataset_name, conf):
    options = get_office_split_options(conf)
    split_dir = options.get("split_dir")
    if split_dir is not None:
        return Path(str(split_dir).format(dataset=dataset_name))
    path_value = options.get("path")
    if path_value is not None:
        path = Path(str(path_value).format(dataset=dataset_name))
        return path if path.suffix.lower() != ".npz" else path.parent
    return Path("data_splits")


def _canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_split_manifest(split_dir, dataset_name, expected_splits):
    """Load the same released-split manifest used by the tuning runner."""
    manifest_path = Path(split_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Released-split manifest not found: {manifest_path.resolve()}\n"
            "Use the same data_splits directory as the hyperparameter-search code."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = manifest.get("datasets", {})
    if dataset_name not in datasets:
        raise KeyError(f"Dataset '{dataset_name}' is absent from {manifest_path}.")
    record = datasets[dataset_name]
    declared = int(record.get("num_splits", expected_splits))
    if declared != expected_splits:
        raise ValueError(
            f"Manifest declares {declared} splits for {dataset_name}, but "
            f"the NPZ bundle contains {expected_splits}."
        )
    fingerprint = hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()
    return manifest_path, record, fingerprint


def _first_metadata_value(record, keys):
    """Find a scalar metadata value without depending on one manifest schema."""
    if not isinstance(record, dict):
        return None
    lowered = {str(key).lower(): value for key, value in record.items()}
    for key in keys:
        value = lowered.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    for value in record.values():
        if isinstance(value, dict):
            found = _first_metadata_value(value, keys)
            if found:
                return found
    return None


def _reference_urls(value):
    urls = []

    def visit(item):
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            urls.extend(re.findall(r"https?://[^\s\]\[\)\}\"']+", item))

    visit(value)
    return list(dict.fromkeys(url.rstrip(".,;") for url in urls))


def print_split_protocol(
    dataset_name,
    conf,
    split_bundle,
    manifest_path,
    manifest_record,
    protocol_fingerprint,
    selected_ids,
):
    """Print provenance before training so every result is auditable."""
    options = get_office_split_options(conf)
    source = options.get("source") or _first_metadata_value(
        manifest_record, ("source", "source_name", "protocol", "benchmark")
    )
    citation = options.get("citation") or _first_metadata_value(
        manifest_record, ("citation", "paper", "reference", "publication")
    )
    configured_url = options.get("reference_url") or options.get("url")
    urls = _reference_urls([configured_url, manifest_record])

    print("Released split protocol")
    print(
        "Type: author-released fixed random partitions "
        "(not mutually exclusive cross-validation folds)"
    )
    print(f"Dataset: {dataset_name}")
    print(f"Split bundle: {split_bundle['path']}")
    print(f"Manifest: {manifest_path}")
    print(f"Released partitions: {split_bundle['num_splits']}")
    print(f"Selected split IDs: {selected_ids}")
    if source:
        print(f"Split source: {source}")
    if citation:
        print(f"Split citation: {citation}")
    if urls:
        for url in urls:
            print(f"Split reference URL: {url}")
    else:
        print("Split reference URL: unavailable in config/manifest")
    print(f"Protocol fingerprint (SHA-256): {protocol_fingerprint}")


def _normalize_mask_layout(mask, key, num_nodes):
    mask = np.asarray(mask)
    if mask.ndim == 1:
        mask = mask[:, None]
    if mask.ndim != 2:
        raise ValueError(f"{key} must be a 1-D or 2-D array, got {mask.shape}.")
    if mask.shape[0] == num_nodes:
        normalized = mask
    elif mask.shape[1] == num_nodes:
        normalized = mask.T
    else:
        raise ValueError(
            f"{key} has shape {mask.shape}; neither dimension equals "
            f"num_nodes={num_nodes}."
        )
    return normalized.astype(bool, copy=False)


def load_office_splits(path):
    """Load and validate all official masks once before the fold loop."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Official split file not found: {path.resolve()}\n"
            "Set dataset.office_split.path to the correct .npz file."
        )

    required = {"train_masks", "val_masks", "test_masks"}
    with np.load(path, allow_pickle=False) as split_file:
        missing = sorted(required.difference(split_file.files))
        if missing:
            raise KeyError(
                f"{path} is missing required arrays: {', '.join(missing)}"
            )

        if "num_nodes" in split_file.files:
            num_nodes = int(np.asarray(split_file["num_nodes"]).item())
        else:
            first_mask = np.asarray(split_file["train_masks"])
            if first_mask.ndim == 1:
                num_nodes = int(first_mask.shape[0])
            else:
                raise KeyError(
                    f"{path} has no num_nodes field; it is required for 2-D masks."
                )

        train_masks = _normalize_mask_layout(
            split_file["train_masks"], "train_masks", num_nodes
        ).copy()
        val_masks = _normalize_mask_layout(
            split_file["val_masks"], "val_masks", num_nodes
        ).copy()
        test_masks = _normalize_mask_layout(
            split_file["test_masks"], "test_masks", num_nodes
        ).copy()

        dataset_name = None
        if "dataset" in split_file.files:
            dataset_name = str(np.asarray(split_file["dataset"]).item())

        node_labels = None
        if "node_labels" in split_file.files:
            node_labels = np.asarray(split_file["node_labels"]).reshape(-1).copy()

        declared_splits = None
        if "num_splits" in split_file.files:
            declared_splits = int(np.asarray(split_file["num_splits"]).item())

    shapes = {train_masks.shape, val_masks.shape, test_masks.shape}
    if len(shapes) != 1:
        raise ValueError(
            "train_masks, val_masks and test_masks must have identical shapes; "
            f"got {train_masks.shape}, {val_masks.shape}, {test_masks.shape}."
        )

    num_splits = train_masks.shape[1]
    if declared_splits is not None and declared_splits != num_splits:
        raise ValueError(
            f"num_splits={declared_splits}, but mask arrays contain "
            f"{num_splits} splits."
        )
    if node_labels is not None and node_labels.shape[0] != num_nodes:
        raise ValueError(
            f"node_labels has {node_labels.shape[0]} rows, expected {num_nodes}."
        )

    for split_id in range(num_splits):
        train = train_masks[:, split_id]
        val = val_masks[:, split_id]
        test = test_masks[:, split_id]
        if not train.any() or not val.any() or not test.any():
            raise ValueError(f"Official split {split_id} contains an empty subset.")
        if np.any(train & val) or np.any(train & test) or np.any(val & test):
            raise ValueError(f"Official split {split_id} has overlapping subsets.")

    return {
        "path": path,
        "dataset": dataset_name,
        "num_nodes": num_nodes,
        "num_splits": num_splits,
        "train_masks": train_masks,
        "val_masks": val_masks,
        "test_masks": test_masks,
        "node_labels": node_labels,
    }


def resolve_office_split_ids(conf, split_bundle, split_id=None, runs=None):
    options = get_office_split_options(conf)
    configured = options.get("split_ids", "all")

    if split_id is not None:
        selected = [int(split_id)]
    elif configured is None or (
        isinstance(configured, str) and configured.lower() == "all"
    ):
        selected = list(range(split_bundle["num_splits"]))
    elif isinstance(configured, (int, np.integer)):
        selected = [int(configured)]
    elif isinstance(configured, str):
        selected = [int(item.strip()) for item in configured.split(",") if item.strip()]
    else:
        selected = [int(item) for item in configured]

    if not selected:
        raise ValueError("No official split IDs were selected.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"Duplicate official split IDs are not allowed: {selected}")

    invalid = [
        item
        for item in selected
        if item < 0 or item >= split_bundle["num_splits"]
    ]
    if invalid:
        raise IndexError(
            f"Official split IDs {invalid} are outside the valid range "
            f"0..{split_bundle['num_splits'] - 1}."
        )

    if runs is not None:
        if runs < 1:
            raise ValueError("--runs must be at least 1.")
        if runs > len(selected):
            raise ValueError(
                f"--runs={runs}, but only {len(selected)} official splits are selected."
            )
        selected = selected[:runs]
    return selected


def verify_predictor_office_split(predictor, split_bundle, split_id):
    """Fail before training if the predictor did not receive the official fold."""
    for subset in ("train", "val", "test"):
        expected = torch.as_tensor(
            split_bundle[f"{subset}_masks"][:, split_id],
            dtype=torch.bool,
            device=predictor.device,
        )
        actual = getattr(predictor, f"mask_{subset}", None)
        if actual is None or not torch.is_tensor(actual):
            raise RuntimeError(f"Predictor exposes no tensor mask_{subset} field.")
        actual = actual.to(device=predictor.device, dtype=torch.bool).reshape(-1)
        if actual.shape != expected.shape or not torch.equal(actual, expected):
            raise RuntimeError(
                f"Predictor mask_{subset} does not match office_split={split_id}."
            )


def build_dataset(
    dataset_name,
    conf,
    data_path,
    device,
    office_split_bundle=None,
    split_id=None,
):
    requested_split_type = str(conf.dataset.get("split_type", "percent")).lower()

    if requested_split_type in OFFICE_SPLIT_TYPES:
        if office_split_bundle is None or split_id is None:
            raise ValueError(
                "office_split_bundle and split_id are required in office_split mode."
            )
        split_dir = resolve_office_split_dir(dataset_name, conf)

        # This is intentionally identical to the released-partition entry used
        # by the hyperparameter-search runner.  Dataset, rather than this
        # script, constructs every index and Boolean-mask field.
        data = Dataset(
            dataset_name,
            path=data_path,
            feat_norm=conf.dataset.get("feat_norm", True),
            adj_norm=conf.dataset.get("adj_norm", True),
            add_self_loop=conf.dataset.get("add_self_loop", True),
            from_npz=conf.dataset.get("from_npz_largest_component", False),
            device=device,
            split_type="default",
            split_id=int(split_id),
            split_dir=str(split_dir),
        )

        info = getattr(data, "split_info", None)
        if not info or int(info.get("split_id", -1)) != int(split_id):
            raise RuntimeError(
                "Dataset did not load the requested released partition."
            )
        if int(data.n_nodes) != int(office_split_bundle["num_nodes"]):
            raise ValueError(
                f"Loaded graph has {data.n_nodes} nodes, but the split bundle "
                f"declares {office_split_bundle['num_nodes']}."
            )

        for subset in ("train", "val", "test"):
            expected = torch.as_tensor(
                office_split_bundle[f"{subset}_masks"][:, split_id],
                dtype=torch.bool,
                device=device,
            )
            actual = getattr(data, f"mask_{subset}", None)
            if actual is None or not torch.is_tensor(actual):
                raise RuntimeError(f"Dataset exposes no tensor mask_{subset} field.")
            actual = actual.to(device=device, dtype=torch.bool).reshape(-1)
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise RuntimeError(
                    f"Dataset mask_{subset} does not match released split {split_id}."
                )
        return data

    return Dataset(
        dataset_name,
        path=data_path,
        feat_norm=conf.dataset["feat_norm"],
        adj_norm=conf.dataset["adj_norm"],
        train_size=conf.dataset["train_size"],
        val_size=conf.dataset["val_size"],
        test_size=conf.dataset["test_size"],
        train_percent=conf.dataset["train_percent"],
        val_percent=conf.dataset["val_percent"],
        test_percent=conf.dataset["test_percent"],
        train_examples_per_class=conf.dataset["train_examples_per_class"],
        val_examples_per_class=conf.dataset["val_examples_per_class"],
        test_examples_per_class=conf.dataset["test_examples_per_class"],
        add_self_loop=conf.dataset["add_self_loop"],
        from_npz=conf.dataset["from_npz_largest_component"],
        device=device,
        split_type=requested_split_type,
    )


def _find_node_logits(value, num_nodes):
    """Find the node-level prediction tensor in a standard predictor output."""
    if torch.is_tensor(value):
        return value if value.ndim >= 1 and value.shape[0] == num_nodes else None
    if isinstance(value, np.ndarray):
        if value.ndim >= 1 and value.shape[0] == num_nodes:
            return torch.as_tensor(value)
        return None
    if isinstance(value, dict):
        preferred = (
            "final_logits",
            "logits",
            "output",
            "outputs",
            "prediction",
            "predictions",
            "probabilities",
            "probs",
        )
        for key in preferred:
            if key in value:
                found = _find_node_logits(value[key], num_nodes)
                if found is not None:
                    return found
        for item in value.values():
            found = _find_node_logits(item, num_nodes)
            if found is not None:
                return found
        return None
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _find_node_logits(item, num_nodes)
            if found is not None:
                return found
    return None


def get_final_logits(predictor):
    """Return logits produced by the checkpoint selected in Predictor.train()."""
    # TAPGNNPredictor.train() restores its best checkpoint and stores the
    # corresponding full-graph logits in best_pred.  Reusing them guarantees
    # that ACC and ROC-AUC refer to exactly the same selected checkpoint and
    # avoids making assumptions about get_prediction's method-specific
    # signature.
    prediction = getattr(predictor, "best_pred", None)
    if prediction is None:
        predictor.model.eval()
        with torch.no_grad():
            prediction = predictor.get_prediction()

    logits = _find_node_logits(prediction, predictor.n_nodes)
    if logits is None:
        raise RuntimeError(
            "Could not find an [num_nodes, ...] prediction tensor in "
            "predictor.get_prediction(...). The predictor should return its final "
            "logits as the first item or under a 'logits'/'final_logits' key."
        )
    return logits.detach().cpu()


def _as_probabilities(logits):
    logits = logits.float()
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        values = logits.reshape(-1)
        if torch.all((values >= 0.0) & (values <= 1.0)):
            return values
        return torch.sigmoid(values)

    if logits.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D logits, got shape {tuple(logits.shape)}.")
    row_sums = logits.sum(dim=1)
    already_probabilities = bool(
        torch.all((logits >= 0.0) & (logits <= 1.0))
        and torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1.0e-5, rtol=1.0e-5
        )
    )
    return logits if already_probabilities else torch.softmax(logits, dim=1)


def calculate_test_metrics(logits, labels, test_indices):
    labels = labels.detach().cpu().reshape(-1).long()
    if torch.is_tensor(test_indices):
        test_indices = test_indices.detach().cpu()
    else:
        test_indices = torch.as_tensor(test_indices)
    if test_indices.dtype == torch.bool:
        test_indices = torch.flatnonzero(test_indices)
    test_indices = test_indices.reshape(-1).long()

    test_logits = logits[test_indices]
    test_labels = labels[test_indices]
    probabilities = _as_probabilities(test_logits)

    if probabilities.ndim == 1:
        predictions = (probabilities >= 0.5).long()
        auc_scores = probabilities.numpy()
    else:
        predictions = probabilities.argmax(dim=1)
        if probabilities.shape[1] == 2:
            auc_scores = probabilities[:, 1].numpy()
        else:
            auc_scores = probabilities.numpy()

    accuracy = float((predictions == test_labels).float().mean().item())
    labels_numpy = test_labels.numpy()
    try:
        if probabilities.ndim == 1 or probabilities.shape[1] == 2:
            roc_auc = float(roc_auc_score(labels_numpy, auc_scores))
        else:
            roc_auc = float(
                roc_auc_score(
                    labels_numpy,
                    auc_scores,
                    labels=np.arange(probabilities.shape[1]),
                    multi_class="ovr",
                    average="macro",
                )
            )
    except ValueError as error:
        warnings.warn(f"ROC-AUC is undefined for this test split: {error}")
        roc_auc = float("nan")
    return accuracy, roc_auc


def get_test_indices(data):
    for name in ("idx_test", "test_masks", "test_mask"):
        if hasattr(data, name):
            return getattr(data, name)
    raise AttributeError("Dataset exposes no test indices/mask.")


def split_size(data, subset):
    for name in (f"idx_{subset}", f"{subset}_masks", f"{subset}_mask"):
        if hasattr(data, name):
            value = getattr(data, name)
            if torch.is_tensor(value) and value.dtype == torch.bool:
                return int(value.sum().item())
            array = np.asarray(value.detach().cpu() if torch.is_tensor(value) else value)
            return int(array.sum()) if array.dtype == bool else int(array.size)
    raise AttributeError(f"Dataset exposes no {subset} indices/mask.")


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return float("nan"), float("nan")
    return float(valid.mean()), float(valid.std())


def main():
    args = parse_args()
    data_path = "dataset"

    conf_path = f"./config/{args.method}/{args.method}_{args.dataset}.yaml"
    conf = load_conf(conf_path)

    base_seed = get_seed_from_conf(conf)
    predictor_cls = PREDICTOR_REGISTRY[args.method]

    office_split_bundle = None
    manifest_path = None
    manifest_record = None
    protocol_fingerprint = None
    if is_office_split(conf):
        split_path = resolve_office_split_path(args.dataset, conf)
        office_split_bundle = load_office_splits(split_path)
        split_dir = resolve_office_split_dir(args.dataset, conf)
        manifest_path, manifest_record, protocol_fingerprint = load_split_manifest(
            split_dir,
            args.dataset,
            office_split_bundle["num_splits"],
        )
        run_items = resolve_office_split_ids(
            conf,
            office_split_bundle,
            split_id=args.split_id,
            runs=args.runs,
        )
    else:
        if args.split_id is not None:
            raise ValueError("--split-id can only be used with split_type: office_split.")
        number_of_runs = 1 if args.runs is None else args.runs
        if number_of_runs < 1:
            raise ValueError("--runs must be at least 1.")
        run_items = list(range(number_of_runs))

    print("=" * 100)
    print("Experiment parameters")
    print(args)
    print(f"Config: {conf_path}")
    if office_split_bundle is not None:
        print_split_protocol(
            args.dataset,
            conf,
            office_split_bundle,
            manifest_path,
            manifest_record,
            protocol_fingerprint,
            run_items,
        )
    print("=" * 100)

    test_acc_results = []
    test_roc_auc_results = []
    per_split_results = []

    for run_id, item_id in enumerate(run_items):
        split_id = item_id if office_split_bundle is not None else None
        run_seed = base_seed + item_id
        setup_seed(run_seed)

        print("=" * 100)
        run_description = f"Run {run_id + 1}/{len(run_items)} | seed={run_seed}"
        if split_id is not None:
            run_description += f" | released_split={split_id}"
        print(run_description)
        print("=" * 100)

        # Rebuild the dataset after setting the run seed.  In office_split mode
        # Dataset loads the requested released partition directly.
        data = build_dataset(
            dataset_name=args.dataset,
            conf=conf,
            data_path=data_path,
            device=args.device,
            office_split_bundle=office_split_bundle,
            split_id=split_id,
        )

        # Dataset-dependent dimensions are filled automatically.
        conf.model["n_feat"] = data.dim_feats
        conf.model["n_class"] = data.n_classes

        print(
            f"Nodes: {data.n_nodes} | Features: {data.dim_feats} | "
            f"Classes: {data.n_classes}"
        )
        print(
            f"Train/Val/Test: {split_size(data, 'train')}/"
            f"{split_size(data, 'val')}/{split_size(data, 'test')}"
        )
        if office_split_bundle is not None:
            info = data.split_info
            print(
                f"Released split verification: id={info['split_id']}/"
                f"{office_split_bundle['num_splits'] - 1} | "
                f"official={info.get('official', True)} | "
                f"fingerprint={info.get('fingerprint', 'unavailable')}"
            )

        predictor = predictor_cls(conf, data, args.device)
        if office_split_bundle is not None:
            verify_predictor_office_split(
                predictor, office_split_bundle, split_id
            )
        result = predictor.train()

        test_indices = get_test_indices(data)
        logits = get_final_logits(predictor)
        test_acc, test_roc_auc = calculate_test_metrics(
            logits, data.labels, test_indices
        )
        result["test_acc"] = test_acc
        result["test_roc_auc"] = test_roc_auc

        selected_metric = str(
            conf.training.get("select_metric", "accuracy")
        ).lower()
        recalculated = test_acc if selected_metric == "accuracy" else test_roc_auc
        if not np.isclose(float(result["test"]), recalculated, atol=1.0e-7):
            raise RuntimeError(
                f"Predictor test {selected_metric}={float(result['test']):.8f} "
                f"does not match the runner value={recalculated:.8f}; the two "
                "components are not evaluating the same split or predictions."
            )

        test_acc_results.append(test_acc)
        test_roc_auc_results.append(test_roc_auc)
        per_split_results.append(
            {
                "split_id": split_id,
                "seed": run_seed,
                "acc": test_acc,
                "roc_auc": test_roc_auc,
            }
        )

        print(
            f"Run {run_id + 1} test ACC: {test_acc:.4f} | "
            f"ROC-AUC: {test_roc_auc:.4f}"
        )

        del predictor, data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    acc_mean, acc_std = summarize(test_acc_results)
    auc_mean, auc_std = summarize(test_roc_auc_results)

    print("=" * 100)
    print("Final test metrics")
    print(f"ACC:     {acc_mean:.4f} +/- {acc_std:.4f}")
    print(f"ROC-AUC: {auc_mean:.4f} +/- {auc_std:.4f}")
    print("=" * 100)

    return {
        "per_split": per_split_results,
        "acc": {"mean": acc_mean, "std": acc_std},
        "roc_auc": {"mean": auc_mean, "std": auc_std},
    }


if __name__ == "__main__":
    main()
