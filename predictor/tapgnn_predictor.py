"""TAPGNN adapter for the existing Base_Predictor interface."""

from __future__ import annotations

import math
import time
from copy import deepcopy

import torch
from sklearn.metrics import roc_auc_score

from .Base_Predictor import Predictor
from predictor.model.tapgnn import TAPGNN
from utils.tapgnn_utils import (
    as_bool_mask,
    build_row_normalized_adjacency,
    node_cross_entropy,
    tapgnn_objective,
)


class TAPGNNPredictor(Predictor):
    """Self pretraining followed by supervised propagation correction.

    n_epochs is the total budget, including warmup_epochs. Checkpoint
    selection and early stopping begin only in the joint stage. Evaluation
    uses final CE and the configured metric, never the joint training loss.
    """

    def method_init(self, conf, data):
        self.feats = self.feats.to(device=self.device, dtype=torch.float32)
        self.labels = self.labels.to(self.device).long().reshape(-1)
        self.n_nodes = self.feats.size(0)
        if self.labels.numel() != self.n_nodes:
            raise ValueError("Expected one class label per graph node.")
        self.n_class = max(2, int(data.n_classes))
        conf.model["n_feat"] = self.feats.size(1)
        conf.model["n_class"] = self.n_class
        self.mask_train = self._to_bool_mask(self.mask_train)
        self.mask_val = self._to_bool_mask(self.mask_val)
        self.mask_test = self._to_bool_mask(self.mask_test)
        for name, mask in (("train", self.mask_train), ("val", self.mask_val),
                           ("test", self.mask_test)):
            if not bool(mask.any()):
                raise ValueError(f"The {name} split must not be empty.")
        if bool(((self.mask_train & self.mask_val) |
                 (self.mask_train & self.mask_test) |
                 (self.mask_val & self.mask_test)).any()):
            raise ValueError("Train, validation, and test splits must be disjoint.")

        # Always build the method's own row normalization from raw data.adj.
        # A loader-provided symmetric normalized_adj is not used here.
        self.propagation = build_row_normalized_adjacency(
            data.adj, device=self.device, dtype=self.feats.dtype,
        )
        self.model = TAPGNN(conf).to(self.device)
        loss_cfg = getattr(conf, "loss", {})
        self.lambda_self = float(loss_cfg.get("lambda_self", 1.0))
        self.lambda_harm = float(loss_cfg.get("lambda_harm", 1.0))
        if min(self.lambda_self, self.lambda_harm) < 0:
            raise ValueError("Loss coefficients must be nonnegative.")
        training = conf.training
        self.num_epochs = int(training.get("n_epochs", 500))
        self.warmup_epochs = int(training.get("warmup_epochs", 20))
        if not 0 <= self.warmup_epochs < self.num_epochs:
            raise ValueError("Require 0 <= warmup_epochs < n_epochs.")
        self.metric_name = str(training.get("select_metric", "accuracy")).lower()
        if self.metric_name not in {"accuracy", "roc_auc"}:
            raise ValueError("select_metric must be accuracy or roc_auc.")
        if self.metric_name == "roc_auc" and self.n_class != 2:
            raise ValueError("The configured ROC-AUC requires two softmax classes.")
        self.metric = self._configured_metric
        self.criterion = str(training.get("criterion", "metric")).lower()
        if self.criterion not in {"loss", "metric"}:
            raise ValueError("criterion must be loss or metric.")
        if str(training.get("optimizer", "Adam")).lower() != "adam":
            raise ValueError("This implementation uses Adam.")
        self.optim = torch.optim.Adam(
            self.model.parameters(),
            lr=float(training.get("lr", 0.01)),
            weight_decay=float(training.get("weight_decay", 5e-4)),
        )
        self._warmup_active = False
        self.last_loss_components = {}
        self.history = []
        self.best_epoch = None

    def _to_bool_mask(self, values):
        return as_bool_mask(values, self.n_nodes, self.device)

    def _configured_metric(self, labels_np, logits_np):
        if self.metric_name == "accuracy":
            return float((logits_np.argmax(axis=1) == labels_np).mean())
        # logits[:, 1] alone is not the score of a two-logit softmax model.
        logits = torch.as_tensor(logits_np)
        probability = logits.softmax(dim=-1)[:, 1].numpy()
        if len(set(labels_np.tolist())) != 2:
            raise ValueError("ROC-AUC is undefined for a split containing one class.")
        return float(roc_auc_score(labels_np, probability))

    def input_distributer(self):
        return {"x": self.feats, "adj": self.propagation,
                "self_only": self._warmup_active}

    def get_prediction(self, label=None, mask=None):
        joint_training = self.model.training and not self._warmup_active
        result = self.model(**self.input_distributer(), return_aux=joint_training)
        logits = result["logits"] if joint_training else result
        loss, metric = None, None
        if label is not None and mask is not None:
            selected = self._to_bool_mask(mask)
            targets = label.to(self.device).long().reshape(-1)
            if not bool(selected.any()):
                raise ValueError("Cannot evaluate an empty split.")
            if joint_training:
                if not torch.equal(selected, self.mask_train):
                    raise ValueError("Joint training may use only the training split.")
                loss, parts = tapgnn_objective(
                    logits, result["logits_self"], targets, selected,
                    self.lambda_self, self.lambda_harm,
                )
                self.last_loss_components = {k: float(v.item()) for k, v in parts.items()}
                self.last_loss_components.update(
                    eta=float(result["eta"][selected].detach().mean().item()),
                    kappa=float(result["kappa"][selected].detach().mean().item()),
                )
            else:
                loss = node_cross_entropy(logits[selected], targets[selected]).mean()
                if self.model.training:
                    self.last_loss_components = {"self": float(loss.detach().item())}
            metric = self.metric(targets[selected].detach().cpu().numpy(),
                                 logits[selected].detach().cpu().numpy())
        return logits, loss, metric

    def train(self):
        training = self.conf.training
        patience = int(training.get("patience", 200))
        print_every = max(1, int(training.get("print_every", 50)))
        debug = bool(training.get("debug", True))
        min_joint_epochs = max(0, int(training.get("min_joint_epochs", 0)))
        self.history = []
        self.weights = None
        self.best_epoch = None
        best_value = math.inf if self.criterion == "loss" else -math.inf
        stale_epochs = 0
        started = time.time()

        for epoch in range(self.num_epochs):
            self._warmup_active = epoch < self.warmup_epochs
            self.model.train()
            self.optim.zero_grad(set_to_none=True)
            _, train_loss, train_metric = self.get_prediction(self.labels, self.mask_train)
            if not bool(torch.isfinite(train_loss)):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch + 1}.")
            train_loss.backward()
            self.optim.step()
            record = {"epoch": epoch + 1,
                      "phase": "warmup" if self._warmup_active else "joint",
                      "loss": float(train_loss.detach().item()),
                      "train_metric": train_metric,
                      **self.last_loss_components}
            stop = False
            improved = False
            if not self._warmup_active:
                val_loss, val_metric = self.evaluate(self.labels, self.mask_val)
                val_loss = float(val_loss.item())
                if not math.isfinite(val_loss) or not math.isfinite(val_metric):
                    raise FloatingPointError("Validation loss or metric is non-finite.")
                value = val_loss if self.criterion == "loss" else val_metric
                improved = value < best_value if self.criterion == "loss" else value > best_value
                if improved:
                    best_value = value
                    stale_epochs = 0
                    self.best_epoch = epoch + 1
                    self.best_val_loss = val_loss
                    self.result["valid"] = val_metric
                    self.result["train"] = train_metric
                    self.weights = deepcopy(self.model.state_dict())
                else:
                    stale_epochs += 1
                joint_epochs = epoch + 1 - self.warmup_epochs
                stop = patience > 0 and stale_epochs >= patience and joint_epochs >= min_joint_epochs
                record.update(val_loss=val_loss, val_metric=val_metric, improved=improved)
            self.history.append(record)
            if debug and (epoch == 0 or (epoch + 1) % print_every == 0 or
                          epoch + 1 in {self.warmup_epochs, self.warmup_epochs + 1,
                                       self.num_epochs} or stop):
                message = (f"Epoch {epoch + 1:04d} | {record['phase']} | "
                           f"loss={record['loss']:.4f} | {self.metric_name}(train)={train_metric:.4f}")
                if not self._warmup_active:
                    message += (f" | cls={record['cls']:.4f} | self={record['self']:.4f}"
                                f" | harm={record['harm']:.4f} | eta={record['eta']:.3f}"
                                f" | kappa={record['kappa']:.3f} | val={val_metric:.4f}")
                print(message + (" *" if improved else ""))
            if stop:
                break

        self._warmup_active = False
        if self.weights is None:
            raise RuntimeError("No joint-stage checkpoint was selected.")
        self.model.load_state_dict(self.weights)
        self.model.eval()
        with torch.no_grad():
            output, _, _ = self.get_prediction()
        self.best_pred = output.detach()
        for name, mask in (("train", self.mask_train), ("valid", self.mask_val),
                           ("test", self.mask_test)):
            self.result[name] = self.metric(
                self.labels[mask].cpu().numpy(), output[mask].cpu().numpy(),
            )
        self.total_time = time.time() - started
        if debug:
            print(f"Best epoch: {self.best_epoch} | "
                  f"{self.metric_name}(test)={self.result['test']:.4f}")
        return self.result
