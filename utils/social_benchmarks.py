"""Download the small attributed social graphs and preserve CMGNN's masks.

The Flickr here has 7,575 nodes (AttributedGraphDataset), rather than the
89,250-node GraphSAINT dataset. No partitions are randomly regenerated.
Only NumPy is required for exporting the released CPU Boolean masks.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import os
import pickle
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


CMGNN_COMMIT = "580dcb42542ba52119187501beeb7b5879cd05b0"
CMGNN_BASE = f"https://raw.githubusercontent.com/zzn-promax/CMGNN/{CMGNN_COMMIT}"
CMGNN_SPLIT_SOURCE = f"https://github.com/zzn-promax/CMGNN/tree/{CMGNN_COMMIT}/data/splits"
SOCIAL_BENCHMARKS = {
    "blogcatalog": {
        "num_nodes": 5196, "num_features": 8189, "num_classes": 6,
        "num_stored_edges": 343486,
        "google_id": "178PqGqh67RUYMMP6-SoRHDoIBh8ku5FS",
        "archive_sha256": "6d587781f714e19649699bff8ac2857c1b730bdc2b6c222eed63db1321fd76e7",
        "source_masks_sha256": "58eaf355175a99f9bc50612c2e0dd17da246a9c5e20951b23ae89baf9f58dd74",
        "labels_sha256": "39aa04b9434d31a173f1a5e601e612559520d7ffbead3050e4bc7cc21f5032a7",
        "raw_sha256": {
            "attrs.npz": "b5481affc451ff5dfed8b229fa5dea011e343edfe9019a6517eb096b0091cfcd",
            "edgelist.txt": "cccbf085dca7e4de2c643b7794f792e96d04ce3f2b2f27d17fdbf5630f864cd8",
            "labels.txt": "7166e9c4dc888dc85e644b57c19babcc1c8645bb5793a10e24e72450cbaa4393",
        },
    },
    "flickr": {
        "num_nodes": 7575, "num_features": 12047, "num_classes": 9,
        "num_stored_edges": 479476,
        "google_id": "1tZp3EB20fAC27SYWwa-x66_8uGsuU62X",
        "archive_sha256": "5ac0c1166417ab13b8677d523832bc74c90f9298ad14a5230ab4242e325eea15",
        "source_masks_sha256": "2702bb3a98d892cf8ed41948c3e642380059935a17fab9d4c2464f3a4ab01a6e",
        "labels_sha256": "d002987b72f506fca33a14b0ac8d3f7f34cb1a005d0726e03469bdfdd000c86e",
        "raw_sha256": {
            "attrs.npz": "c590fd87bb00cf6a300e98b7a79da62e6f0dc18e50117455c125dcb88184a4c1",
            "edgelist.txt": "311c3d66cc710ac5412a03004d0a87042128c36de6657e92d67af6fc2d50c8cf",
            "labels.txt": "264d0a43458933bce89dfd78f28864f97fbfc3fce999c09f16d0548bfd45693f",
        },
    },
}
MASK_KEYS = ("train_masks", "val_masks", "test_masks")
RAW_NAMES = ("attrs.npz", "edgelist.txt", "labels.txt")


def canonical_social_name(name):
    name = str(name).strip().lower().replace("_", "-")
    name = {"blog-catalog": "blogcatalog", "blogcatalog": "blogcatalog",
            "flickr": "flickr"}.get(name, name)
    if name not in SOCIAL_BENCHMARKS:
        raise ValueError(f"Expected blogcatalog or flickr, received {name!r}.")
    return name


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def labels_sha256(labels):
    return hashlib.sha256(np.asarray(labels).reshape(-1).astype("<i8").tobytes()).hexdigest()


def split_fingerprint(parts):
    # Must match utils.datasplit._split_fingerprint exactly.
    digest = hashlib.sha256()
    for name, indices in zip(("train", "valid", "test"), parts):
        indices = np.sort(np.asarray(indices, dtype=np.int64).reshape(-1))
        digest.update(name.encode("ascii"))
        digest.update(np.asarray([len(indices)], dtype="<i8").tobytes())
        digest.update(indices.astype("<i8").tobytes())
    return digest.hexdigest()


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def download_verified(urls, destination, expected_sha256):
    """Reuse verified files and atomically download missing source files."""
    destination = Path(destination)
    if destination.is_file():
        if file_sha256(destination) != expected_sha256:
            raise ValueError(f"Source checksum mismatch: {destination}. Remove the corrupt cache and retry.")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        fd, temp = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                    dir=destination.parent)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with os.fdopen(fd, "wb") as output:
                with urllib.request.urlopen(request, timeout=60) as response:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            if file_sha256(temp) != expected_sha256:
                raise ValueError("Downloaded bytes do not match the verified source checksum.")
            os.replace(temp, destination)
            return destination
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
        finally:
            if os.path.exists(temp):
                os.unlink(temp)
    raise RuntimeError(f"Could not download {destination.name}: {'; '.join(errors)}")


def default_source_dir():
    return Path(__file__).resolve().parents[1] / "data_splits" / "_sources"


def read_social_labels(path, name):
    name = canonical_social_name(name)
    meta = SOCIAL_BENCHMARKS[name]
    rows = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if rows.shape != (meta["num_nodes"], 2):
        raise ValueError(f"{name}: expected one class ID per original node.")
    if not np.array_equal(rows[:, 0], np.arange(meta["num_nodes"])):
        raise ValueError(f"{name}: node IDs are not in the original 0..N-1 order.")
    labels = rows[:, 1] - 1  # Exactly the conversion used by PyG.
    if not np.array_equal(np.unique(labels), np.arange(meta["num_classes"])):
        raise ValueError(f"{name}: unexpected class IDs.")
    if labels_sha256(labels) != meta["labels_sha256"]:
        raise ValueError(f"{name}: source labels or node ordering do not match.")
    return labels


def ensure_social_raw(name, root, source_dir=None):
    """Prepare PyG-compatible root/name/raw files using its official archive."""
    name = canonical_social_name(name)
    meta = SOCIAL_BENCHMARKS[name]
    raw_dir = Path(root).expanduser().resolve() / name / "raw"
    raw_paths = {key: raw_dir / key for key in RAW_NAMES}
    for key, path in raw_paths.items():
        if path.exists() and file_sha256(path) != meta["raw_sha256"][key]:
            raise ValueError(f"Unexpected existing {name} raw file: {path}. Preserve original node numbering.")
    if not all(path.is_file() for path in raw_paths.values()):
        cache = Path(source_dir) if source_dir is not None else default_source_dir()
        google_id = meta["google_id"]
        archive = download_verified([
            f"https://drive.usercontent.google.com/download?id={google_id}&export=download&confirm=t",
            f"https://drive.google.com/uc?export=download&id={google_id}&confirm=t",
        ], cache / f"{name}.attr.zip", meta["archive_sha256"])
        # Extract only the three known members, rather than arbitrary zip paths.
        with zipfile.ZipFile(archive) as packed:
            payloads = {key: packed.read(f"{name}.attr/{key}") for key in RAW_NAMES}
        for key, payload in payloads.items():
            if hashlib.sha256(payload).hexdigest() != meta["raw_sha256"][key]:
                raise ValueError(f"Invalid {name} archive member: {key}")
        for key, path in raw_paths.items():
            if not path.is_file():
                atomic_write(path, payloads[key])
    read_social_labels(raw_paths["labels.txt"], name)
    return raw_dir


class _BoolStorage:
    pass


class _ReleasedMaskReader(pickle.Unpickler):
    """Read ONLY the pinned release's one-dimensional CPU bool tensors.

    No arbitrary pickle globals, code, PyG objects, GPU tensors or tensor
    views are permitted. This avoids requiring torch to export these masks.
    """
    def __init__(self, stream, archive):
        super().__init__(stream)
        self.archive = archive

    def find_class(self, module, name):
        allowed = {
            ("torch", "BoolStorage"): _BoolStorage,
            ("torch._utils", "_rebuild_tensor_v2"): self.rebuild_tensor,
            ("collections", "OrderedDict"): collections.OrderedDict,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported mask-payload global: {module}.{name}")
        return allowed[(module, name)]

    def persistent_load(self, storage_id):
        if not isinstance(storage_id, tuple) or len(storage_id) != 5:
            raise pickle.UnpicklingError("Invalid released-mask storage.")
        kind, dtype, key, location, size = storage_id
        if kind != "storage" or dtype is not _BoolStorage or location != "cpu":
            raise pickle.UnpicklingError("Only CPU Boolean storage is supported.")
        if not isinstance(key, str) or not key.isdigit() or not isinstance(size, int):
            raise pickle.UnpicklingError("Invalid mask storage key/size.")
        payload = self.archive.read(f"archive/data/{key}")
        if len(payload) != size:
            raise ValueError("Released-mask storage length mismatch.")
        return np.frombuffer(payload, dtype=np.bool_)

    @staticmethod
    def rebuild_tensor(storage, offset, shape, stride, requires_grad, hooks):
        if (offset != 0 or tuple(stride) != (1,) or len(shape) != 1
                or shape[0] != len(storage) or requires_grad or hooks):
            raise ValueError("Unsupported released-mask tensor layout.")
        return storage.copy()


def read_released_social_masks(name, source_dir=None):
    name = canonical_social_name(name)
    meta = SOCIAL_BENCHMARKS[name]
    cache = Path(source_dir) if source_dir is not None else default_source_dir()
    relative = f"data/splits/{name}_splits.pt"
    path = download_verified([f"{CMGNN_BASE}/{relative}"],
                             cache / f"{name}_splits.pt", meta["source_masks_sha256"])
    with zipfile.ZipFile(path) as archive:
        payload = _ReleasedMaskReader(io.BytesIO(archive.read("archive/data.pkl")), archive).load()
    if not isinstance(payload, dict) or set(payload) != {"train", "val", "test"}:
        raise ValueError(f"Unexpected {name} author mask keys.")
    matrices = []
    for key in ("train", "val", "test"):
        columns = payload[key]
        if (not isinstance(columns, list) or len(columns) != 10 or
                any(column.dtype != np.bool_ or column.shape != (meta["num_nodes"],)
                    for column in columns)):
            raise ValueError(f"{name}: invalid released {key} masks.")
        matrices.append(np.column_stack(columns))
    if not np.all(sum(matrix.astype(np.int8) for matrix in matrices) == 1):
        raise ValueError(f"{name}: every split must assign every node exactly once.")
    expected = (2494, 1662, 1040) if name == "blogcatalog" else (3636, 2424, 1515)
    if any(not np.all(matrix.sum(axis=0) == size) for matrix, size in zip(matrices, expected)):
        raise ValueError(f"{name}: counts do not match the released 48/32/20 protocol.")
    return tuple(matrices)


def validate_social_mask_bank(path, name):
    name = canonical_social_name(name)
    meta = SOCIAL_BENCHMARKS[name]
    with np.load(path, allow_pickle=False) as archive:
        expected_keys = {"schema_version", "dataset", "num_nodes", "num_splits",
                         *MASK_KEYS, "node_labels"}
        if not expected_keys.issubset(archive.files):
            raise ValueError(f"Missing Actor-compatible fields: {path}")
        if (int(archive["schema_version"].item()) != 1 or
                str(archive["dataset"].item()) != name or
                int(archive["num_nodes"].item()) != meta["num_nodes"] or
                int(archive["num_splits"].item()) != 10):
            raise ValueError(f"Unexpected mask-bank metadata: {path}")
        matrices = tuple(archive[key].copy() for key in MASK_KEYS)
        labels = archive["node_labels"].copy()
    if any(matrix.dtype != np.bool_ or matrix.shape != (meta["num_nodes"], 10)
           for matrix in matrices):
        raise ValueError(f"Invalid mask bank shape/dtype: {path}")
    if not np.all(sum(matrix.astype(np.int8) for matrix in matrices) == 1):
        raise ValueError(f"Overlapping or incomplete masks: {path}")
    if labels.dtype.kind not in "iu" or labels.shape != (meta["num_nodes"],):
        raise ValueError(f"Invalid original labels in {path}")
    if labels_sha256(labels) != meta["labels_sha256"]:
        raise ValueError(f"Source labels or node indexing differ in {path}")
    return matrices, labels


def make_social_manifest_record(path, name, matrices, labels):
    meta = SOCIAL_BENCHMARKS[name]
    splits = []
    for split_id in range(10):
        parts = tuple(np.flatnonzero(matrix[:, split_id]) for matrix in matrices)
        sizes = [len(part) for part in parts]
        splits.append({
            "split_id": split_id, "sizes": sizes,
            "ratios": [size / meta["num_nodes"] for size in sizes],
            "fingerprint": split_fingerprint(parts),
        })
    return {
        "file": f"{name}.npz", "schema_version": 1,
        "num_nodes": meta["num_nodes"], "num_splits": 10,
        "num_features": meta["num_features"], "num_classes": meta["num_classes"],
        "kind": "local_author_masks", "official": True,
        "ratios": [0.48, 0.32, 0.20], "nominal_ratios": [0.48, 0.32, 0.20],
        "reference": "CMGNN author-released 10 random partitions (48/32/20)",
        "source": CMGNN_SPLIT_SOURCE, "source_commit": CMGNN_COMMIT,
        "source_file": f"data/splits/{name}_splits.pt",
        "source_masks_sha256": meta["source_masks_sha256"],
        "dataset_source": "PyG AttributedGraphDataset / Scaling Attributed Network Embedding to Massive Graphs",
        "dataset_google_id": meta["google_id"],
        "file_sha256": file_sha256(path), "labels_sha256": labels_sha256(labels),
        "mask_layout": "num_nodes,num_splits", "splits": splits,
    }


def merge_social_manifest(split_dir, records):
    """Update just the selected social graph entries; retain other datasets."""
    root = Path(split_dir)
    path = root / "manifest.json"
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or not isinstance(manifest.get("datasets"), dict):
            raise ValueError(f"Unsupported existing manifest: {path}")
    else:
        manifest = {"schema_version": 1, "datasets": {}}
    changed = any(manifest["datasets"].get(name) != record for name, record in records.items())
    manifest["datasets"].update(records)
    if changed or not path.is_file():
        if path.is_file():
            backup = root / "manifest.before_blogcatalog_flickr.json"
            if not backup.exists():
                atomic_write(backup, path.read_bytes())
        atomic_write(path, (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return path


def prepare_social_benchmarks(names=("blogcatalog", "flickr"), data_path=None,
                              split_dir=None, overwrite=False):
    """Download raw graphs, export original masks, and merge manifest.json."""
    project = Path(__file__).resolve().parents[1]
    data_path = Path(data_path) if data_path is not None else project / "dataset"
    split_dir = Path(split_dir) if split_dir is not None else project / "data_splits"
    names = list(dict.fromkeys(canonical_social_name(name) for name in names))
    if not names:
        raise ValueError("At least one social benchmark is required.")
    records = {}
    cache = split_dir / "_sources"
    for name in names:
        meta = SOCIAL_BENCHMARKS[name]
        raw_dir = ensure_social_raw(name, data_path, source_dir=cache)
        labels = read_social_labels(raw_dir / "labels.txt", name)
        matrices = read_released_social_masks(name, source_dir=cache)
        destination = split_dir / f"{name}.npz"
        reused = False
        if destination.exists() and not overwrite:
            existing, old_labels = validate_social_mask_bank(destination, name)
            if (not np.array_equal(old_labels, labels) or
                    any(not np.array_equal(old, new) for old, new in zip(existing, matrices))):
                raise ValueError(f"{destination} differs from the CMGNN release; use --overwrite to replace it.")
            reused = True
        if not reused:
            payload = io.BytesIO()
            np.savez_compressed(payload, schema_version=np.int64(1), dataset=np.array(name),
                                num_nodes=np.int64(meta["num_nodes"]), num_splits=np.int64(10),
                                node_labels=labels.astype(np.int64),
                                **dict(zip(MASK_KEYS, matrices)))
            atomic_write(destination, payload.getvalue())
            validate_social_mask_bank(destination, name)
        records[name] = make_social_manifest_record(destination, name, matrices, labels)
        sizes = [int(matrix[:, 0].sum()) for matrix in matrices]
        print(f"[{name}] {'REUSE' if reused else 'EXPORT'} {destination}; "
              f"shape=({meta['num_nodes']}, 10), train/val/test={sizes}")
    manifest_path = merge_social_manifest(split_dir, records)
    print(f"Manifest merged: {manifest_path}")
    return records
