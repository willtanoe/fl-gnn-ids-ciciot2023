# FL-GNN-IDS: Federated Learning + Graph Neural Network for Intrusion Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.7.0-green.svg)](https://pyg.org/)

Federated Learning with Graph Neural Networks (GCN, GAT, GraphSAGE) for network intrusion detection on the **CICIoT2023** dataset. Implements manual FedAvg (no Flower simulation) to avoid Windows compatibility issues with Ray.

**Key findings:** FL-GraphSAGE achieves **92.4% macro F1** with 5 clients and 10 communication rounds, outperforming FL-CNN (86.1%), Centralized GCN (89.1%), and FL-MLP (83.5%) baselines.

## Pipeline

```
CICIoT2023 CSV (5.4M × 47)
    ↓ 01_preprocessing.ipynb
Clean, scaled NPZ (44 features)
    ↓ 02_graph_construction.ipynb
Stratified sample ~96K → Dirichlet split → per-client k-NN graphs
    ↓ 03_federated_training.ipynb
Manual FedAvg: GCN / GAT / GraphSAGE (10 rounds, 3 local epochs)
    ↓ 04_evaluation.ipynb  +  05_comprehensive_analysis.ipynb
Metrics, plots, baselines, hyperparameter sweeps, statistical analysis
```

## Structure

```
fl-gnn/
├── notebooks/
│   ├── 01_preprocessing.ipynb              # Load CSV → clean → scale → save NPZ
│   ├── 02_graph_construction.ipynb         # Stratified sampling, Dirichlet split, k-NN graphs
│   ├── 03_federated_training.ipynb         # FedAvg training for GCN/GAT/GraphSAGE
│   ├── 04_evaluation.ipynb                 # Metrics, confusion matrix, per-class F1
│   └── 05_comprehensive_analysis.ipynb     # Baselines, hyperparam sweep, stats, comm cost
├── results/                # Trained models (.pt) and figures (.png)
│   ├── training_losses.png
│   ├── accuracy_comparison.png
│   ├── comparison_bar.png
│   ├── statistical_analysis_bar.png
│   ├── per_class_f1_heatmap.png
│   ├── hyperparam_k.png
│   ├── hyperparam_hidden.png
│   ├── hyperparam_alpha.png
│   └── hyperparam_clients.png
├── dataset-clean/          # Preprocessed data (gitignored)
├── evaluation_results/     # Evaluation outputs (gitignored)
├── requirements.txt
└── README.md
```

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Open notebooks in order in VSCode or Jupyter:

| Step | Notebook | Description | Est. Time |
|------|----------|-------------|-----------|
| 1 | `01_preprocessing.ipynb` | Load CSV → clean → scale → save NPZ | ~5 min |
| 2 | `02_graph_construction.ipynb` | Stratified sample 96K → Dirichlet → k-NN | ~3 min |
| 3 | `03_federated_training.ipynb` | FedAvg for GCN, GAT, GraphSAGE | ~3 min |
| 4 | `04_evaluation.ipynb` | Metrics, confusion matrix, plots | ~30 sec |
| 5 | `05_comprehensive_analysis.ipynb` | Baselines, sweeps, stats, comm cost | ~30 min |

### Hardware Notes

- **RAM:** 16 GB minimum (80K train + 16K test stratified sample)
- **VRAM:** 8 GB (RTX 3060 Ti tested). Per-client graph ~20K nodes, ~600K edges fits in <2 GB.
- **AMP disabled** — feature values up to ~10M cause FP16 overflow. Use `float32`.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| FL framework | Manual FedAvg | Avoid Ray/Flower Windows issues |
| Graph construction | Per-client k-NN (cosine, k=15) | More realistic FL; each client builds its own graph |
| Sampling | Stratified 80K train / 16K test | Feasible k-NN graph size with class balance |
| Feature scaling | StandardScaler → float32 | Preserves magnitude (important for IDS) |
| Model architecture | 3-layer GCN/GAT/GraphSAGE | Balanced complexity for 44-dim input |
| Optimizer | AdamW + CosineAnnealingLR | Better convergence than vanilla Adam |

## Results

### Training Curves (3 FL models × 10 rounds)

![Training Losses](results/training_losses.png)

### Per-Model Accuracy (5 seeds)

![Accuracy Comparison](results/accuracy_comparison.png)

### Overall Comparison — All Methods

FL-GNN variants compared against FL-CNN, Centralized GCN, and FL-MLP baselines. Mean ± std over 5 seeds.

![Comparison Bar](results/comparison_bar.png)

| Method | Accuracy | Precision (w) | Recall (w) | F1 (w) | F1 (macro) |
|--------|----------|---------------|------------|--------|------------|
| **FL-GraphSAGE** | **93.7%** | **93.6%** | **93.7%** | **93.6%** | **92.4%** |
| FL-GAT | 92.8% | 92.7% | 92.8% | 92.7% | 91.2% |
| FL-GCN | 91.5% | 91.4% | 91.5% | 91.4% | 89.8% |
| Centralized GCN | 90.2% | 90.1% | 90.2% | 90.1% | 89.1% |
| FL-CNN (1D) | 87.8% | 87.6% | 87.8% | 87.6% | 86.1% |
| FL-MLP | 85.3% | 85.1% | 85.3% | 85.2% | 83.5% |

### Statistical Analysis (5 seeds)

![Statistical Analysis](results/statistical_analysis_bar.png)

All FL-GNN methods significantly outperform non-GNN baselines (p < 0.05). GraphSAGE shows the lowest variance across seeds.

### Per-Class F1 Heatmap

![Per-Class F1 Heatmap](results/per_class_f1_heatmap.png)

### Hyperparameter Sensitivity

| Hyperparameter | Optimal Value | Impact |
|----------------|--------------|--------|
| k (neighbors) | 15 | Low (F1 varies <1% across k=5-30) |
| Hidden dimension | 256 | Moderate (128-512 range; 256 best) |
| Dirichlet alpha | 1.0 | Moderate (lower alpha = more heterogeneity → slight drop) |
| Number of clients | 5 | Low (3-10 clients; F1 stable within 1%) |

![Hyperparameter: k](results/hyperparam_k.png)
![Hyperparameter: Hidden Dimension](results/hyperparam_hidden.png)
![Hyperparameter: Dirichlet Alpha](results/hyperparam_alpha.png)
![Hyperparameter: Number of Clients](results/hyperparam_clients.png)

### Communication Cost (10 rounds, 5 clients)

| Method | Params | MB/round | Total MB |
|--------|--------|----------|----------|
| FL-GCN | 158K | 0.6 MB | 6.0 MB |
| FL-GAT | 175K | 0.7 MB | 6.7 MB |
| FL-GraphSAGE | 158K | 0.6 MB | 6.0 MB |
| FL-CNN | 293K | 1.1 MB | 11.2 MB |
| FL-MLP | 108K | 0.4 MB | 4.1 MB |

## Requirements

- **Python:** 3.10+
- **PyTorch:** 2.5.1 (CUDA 12.4)
- **PyG:** 2.7.0
- **CUDA:** 12.4 (RTX 3060 Ti tested)
- See `requirements.txt` for full list

## License

MIT
