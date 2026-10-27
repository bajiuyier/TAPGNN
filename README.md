# TAPGNN: Task-Aligned Node-Specific Propagation for Heterophilic Graphs

Official implementation of **TAPGNN (Task-Aligned Node-Specific Propagation for Heterophilic Graphs)**.

**Status:** The associated manuscript is currently under review. This repository provides the core implementation and selected experimental configurations. Additional configurations and experimental scripts are planned for release after the review process.

## Overview

Graph neural networks (GNNs) typically learn node representations through neighborhood aggregation. However, under heterophily, neighboring nodes may belong to different classes, and propagation can introduce information that is detrimental to node classification.

TAPGNN addresses this issue by learning node-specific propagation adjustments guided by classification objectives. Instead of relying solely on a shared propagation rule or adjusting propagation strength, TAPGNN jointly controls the direction and magnitude of representation updates.

The framework consists of three main components:

1. **Pre-propagation Representations and Multi-hop States:** TAPGNN encodes node attributes and constructs multi-hop neighborhood representations to capture structural information at different propagation depths.

2. **Node-Specific Component Correction and Strength Control:** TAPGNN learns a node-specific correction axis to adjust propagation components and an adaptive coefficient to control update magnitude.

3. **Task Supervision and Training:** TAPGNN combines classification supervision, auxiliary pre-propagation supervision, and a harm penalty to discourage propagation updates that increase node-level classification loss.

Through these components, TAPGNN aims to improve node classification across both homophilic and heterophilic graphs.

## Code Availability

The current release includes the core TAPGNN implementation and selected configuration files.

To support reproducible research, we plan to progressively extend this repository with:

- Additional dataset-specific hyperparameter configurations.
- Complete experimental reproduction scripts.
- Ablation study configurations and scripts.
- Mechanism analysis and visualization scripts.
- Further implementation details and experimental documentation.

These materials are planned for release following the completion of the paper review process.

## Datasets

The datasets are **not included** in this repository due to their size and external distribution requirements.

TAPGNN is evaluated on publicly available graph node classification benchmarks covering both homophilic and heterophilic settings.

Relevant benchmark sources include:

- [Heterophilous Graph Benchmark (ICLR 2023)](https://github.com/yandex-research/heterophilous-graphs)
- [LINKX / Non-Homophily Large-Scale Benchmark (NeurIPS 2021)](https://github.com/CUAI/Non-Homophily-Large-Scale)
- [PyTorch Geometric Datasets](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html)

Please obtain the required datasets from their original sources and prepare them according to the data-loading interfaces provided in the repository.

**Note:** Dataset preprocessing, train/validation/test partitions, and evaluation metrics must be consistent with the corresponding experimental protocols. Not all benchmark datasets use the same split strategy or evaluation metric.

## Configuration

Dataset-specific configurations are managed through configuration files.

The current repository provides selected configurations for the TAPGNN model. These configurations are intended to facilitate inspection and preliminary experimentation with the implementation.

The complete set of experimental hyperparameters will be made available in a subsequent update.

## Reproducibility

This repository is currently a partial research code release. Although the core implementation is available, not all configurations and scripts required to reproduce the complete experimental results have been released.

The repository will be updated with additional reproducibility materials after the review process.

## Citation

Citation information will be provided when the associated paper becomes publicly available.

## Updates

This repository will be maintained and progressively updated with additional experimental configurations, scripts, and documentation.
