# FL-GNN-IDS: Federated Learning + Graph Neural Network for Intrusion Detection

Federated Learning with Graph Neural Networks (GCN, GAT, GraphSAGE) for intrusion detection on the CICIoT2023 dataset.

**Note:** This is a notebook-based implementation using manual FedAvg (not Flower). A Flower-based version is available in a separate repository.

## Structure

```
fl-gnn/
├── notebooks/
│   ├── 01_preprocessing.ipynb         # Load, clean, scale, save as NPZ
│   ├── 02_graph_construction.ipynb    # Stratified sampling, Dirichlet split, k-NN graphs
│   ├── 03_federated_training.ipynb    # FedAvg training for GCN/GAT/GraphSAGE
│   ├── 04_evaluation.ipynb            # Metrics, confusion matrix, per-class F1
│   └── 05_comprehensive_analysis.ipynb # Baselines, hyperparam sweep, stats, comm cost
├── dataset-clean/      # Preprocessed data (gitignored)
├── results/            # Trained models & plots (PNGs tracked)
├── evaluation_results/ # Evaluation outputs (gitignored)
└── requirements.txt
```

## How to Run

1. `pip install -r requirements.txt`
2. Open notebooks in VSCode/Jupyter and run in order:
   - `01_preprocessing.ipynb`
   - `02_graph_construction.ipynb`
   - `03_federated_training.ipynb`
   - `04_evaluation.ipynb`
   - `05_comprehensive_analysis.ipynb` (optional: baselines, stats, hyperparameter sweep)

## Pipeline Overview

| Step | Notebook | Description | Est. Time |
|------|----------|-------------|-----------|
| 1 | `01_preprocessing.ipynb` | Load CSV → clean → scale → save NPZ | ~5 min |
| 2 | `02_graph_construction.ipynb` | Stratified sample 100K → Dirichlet split → k-NN graphs | ~3 min |
| 3 | `03_federated_training.ipynb` | FL-FedAvg for GCN, GAT, GraphSAGE | ~3 min |
| 4 | `04_evaluation.ipynb` | Load trained models → metrics → plots | ~30 sec |
| 5 | `05_comprehensive_analysis.ipynb` | Baselines, hyperparameter sweep, stats, communication cost | ~30 min |

## Results

### Training Curves

![Training Losses](results/training_losses.png)

### Accuracy Comparison

![Accuracy Comparison](results/accuracy_comparison.png)

## Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- PyTorch Geometric 2.x
- See `requirements.txt` for full list
