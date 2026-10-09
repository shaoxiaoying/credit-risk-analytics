# Analysis summary

## Sample

- Customers: **30,000**
- Observed next-month default rate: **22.1%**
- Duplicate rows removed: **0**

## Baseline model

Stratified 80/20 holdout; logistic regression; 0.50 classification threshold.

| Metric | Holdout |
|---|---:|
| ROC AUC | 0.700 |
| Accuracy | 0.804 |
| Default precision | 0.697 |
| Default recall | 0.205 |
| Default F1 | 0.317 |

This model is an educational benchmark only. See the README for limitations and responsible-use notes.
