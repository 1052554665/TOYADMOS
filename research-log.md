>enhance the test script to save test results (F-measure, AUC, etc.) and save the ROC curve as a PDF. 


Updated `torch_version/test.py` to save the following artifacts:

| Output File | Format | Contents |
|---|---|---|
| `results_.../roc_curve.pdf` | Vector PDF (300 DPI) | Publication-quality ROC curve |
| `results_.../roc_curve.png` | PNG (150 DPI) | Quick preview |
| `results_.../summary_metrics.json` | JSON | AUC, F-measure, Precision, Recall, TP/FP/FN, threshold, score stats |
| `results_.../roc_data.csv` | CSV | Raw (FPR, TPR) pairs for external plotting |

**Verified results (ToyConveyor, RTX 5090):**

```json
{
  "AUC": 0.9794,
  "F_measure": 0.8966,
  "Precision": 0.8298,
  "Recall": 0.9750,
  "TP": 390.0, "FP": 80.0, "FN": 10.0,
  "threshold_at_FPR": 31.35
}
```

The JSON summary also records the **score distribution statistics** (min/max/mean/std) for both normal and anomalous files — useful for comparing runs or detecting distribution shift.