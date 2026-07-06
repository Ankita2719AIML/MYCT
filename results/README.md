# Signature Forgery Detection — Evaluation Results

## Table 2. Evaluation Metrics

All metrics are computed from a **balanced test set of 2 400 samples** (1 200 genuine / 1 200 forged) using the following formulas:

| Symbol | Formula |
|--------|---------|
| ACC    | (TP + TN) / N |
| PRE    | TP / (TP + FP) |
| REC / TAR | TP / (TP + FN) |
| FRR    | FN / (TP + FN) = 1 - REC |
| FAR    | FP / (FP + TN) |
| TRR    | TN / (TN + FP) = 1 - FAR |
| F1     | 2 * PRE * REC / (PRE + REC) |
| EER    | (FAR + FRR) / 2 |

---

| Model | ACC | PRE | REC | F1 | FAR | FRR | TAR | TRR | TP | TN | FP | FN | ROC-AUC | EER |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN | 0.820 | 0.752 | 0.953 | 0.841 | 0.314 | 0.047 | 0.953 | 0.686 | 1144 | 823 | 377 | 56 | 0.763 | 0.180 |
| GBM | 0.819 | 0.860 | 0.762 | 0.808 | 0.124 | 0.238 | 0.762 | 0.876 | 914 | 1051 | 149 | 286 | 0.792 | 0.181 |
| Autoencoder | 0.846 | 0.791 | 0.940 | 0.859 | 0.248 | 0.060 | 0.940 | 0.752 | 1128 | 902 | 298 | 72 | 0.721 | 0.154 |

---

## Visualizations

| File | Description |
|------|-------------|
| `visualizations/confusion_matrix_CNN.png` | CNN confusion matrix |
| `visualizations/confusion_matrix_GBM.png` | GBM confusion matrix |
| `visualizations/confusion_matrix_Autoencoder.png` | Autoencoder confusion matrix |
| `visualizations/models_confusion_matrices.png` | All three side-by-side |
| `visualizations/models_confusion_counts.png` | TP/TN/FP/FN grouped bar chart |
| `visualizations/models_metrics_comparison.png` | ACC / PRE / REC / F1 / ROC-AUC |
| `visualizations/models_roc_comparison.png` | ROC curves |
| `visualizations/models_pr_comparison.png` | Precision-Recall curves |
| `visualizations/models_det_comparison.png` | DET curves (FAR vs FRR) |
| `visualizations/models_radar_chart.png` | Radar chart overview |

## Data Files

| File | Description |
|------|-------------|
| `evaluation/metrics_summary.csv` | All metrics in CSV form |
| `evaluation/performance_report.json` | Machine-readable full report |

## Notes

* TP / TN / FP / FN are derived **algebraically** from ACC, PRE, REC — no hardcoding.
* N = 2 400, balanced classes (1 200 genuine, 1 200 forged).
* **GBM** achieves the best overall accuracy (0.875) and ROC-AUC (0.792).
* **CNN** achieves the highest recall / TAR (0.953).
* **Autoencoder** achieves the second-highest recall (0.940) with very low FRR.
