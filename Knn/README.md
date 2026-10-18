# K-Nearest Neighbours (KNN) from Scratch

A from-scratch implementation of the **k-nearest neighbours classifier** using only NumPy, applied to the
[Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) dataset.

scikit-learn is used **only** to load the data and split it into train and test sets. The feature scaler, distance functions,
neighbour search, voting and evaluation metrics are all written by hand.

## Result

| Metric (malignant = positive class) | Value |
|---|---|
| Accuracy | **98.2%** (56 / 57 test samples) |
| Precision | 1.000 |
| Recall | 0.952 (20 of 21 malignant tumours found) |

Confusion matrix (rows = actual, columns = predicted):

|  | Predicted malignant | Predicted benign |
|---|---|---|
| **Actually malignant** | 20 | 1 |
| **Actually benign** | 0 | 36 |

Settings: k = 5, Euclidean distance, standardised features, 90/10 stratified split (`random_state=42`).

## What's implemented

- **Standardisation** (`z = (x - mean) / std`), fitted on the training set only to avoid data leakage
- **Distance functions:** Euclidean, Manhattan, Minkowski
- **Neighbour search:** indices of the k nearest training points
- **Majority-vote prediction** for a single point and for a whole test set
- **Evaluation:** accuracy, confusion matrix, precision and recall

## How KNN works

KNN is a lazy learner: there is no training step, the training data is simply stored. To classify a new point:

1. Compute its distance to every training point.
2. Take the k closest points (the neighbours).
3. Predict the most common class among them.

Since everything depends on distances, features must be on a comparable scale, which is why standardisation is part of the pipeline.

| Metric | Formula |
|---|---|
| Euclidean | `sqrt(sum((x_j - y_j)^2))` |
| Manhattan | `sum(abs(x_j - y_j))` |
| Minkowski | `sum(abs(x_j - y_j)^p) ^ (1/p)` (p=1 is Manhattan, p=2 is Euclidean) |

## Dataset

- 569 samples, 30 numeric features computed from images of breast mass cell nuclei (radius, texture, perimeter, area, smoothness, etc.)
- Target: `0` = malignant (212 samples), `1` = benign (357 samples)
- No missing values

## Limitations

- **Small test set:** with 57 samples, one extra mistake changes accuracy by about 1.8 percentage points.
- **Single split:** the result comes from one train/test split, so another `random_state` could give a different score.
- **k not tuned:** k = 5 was chosen by hand, not selected by validation.
- **Prediction cost:** each prediction computes distances to every training point, `O(n · d)` per query.

## Possible next steps

- [ ] Choose k with k-fold cross-validation and plot accuracy against k
- [ ] Compare Euclidean, Manhattan and Minkowski distances
- [ ] Add distance-weighted voting
- [ ] Show the effect of not scaling the features
- [ ] Wrap the code in a `KNN` class with `fit`, `predict` and `predict_proba`
- [ ] Validate against `sklearn.neighbors.KNeighborsClassifier`
- [ ] Vectorise prediction across all test points and speed up the neighbour search

## License

Add a license of your choice (for example MIT) before publishing.
