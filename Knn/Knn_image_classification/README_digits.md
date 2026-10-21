# K-Nearest Neighbours (KNN) from Scratch: Handwritten Digit Recognition

A from-scratch implementation of the **k-nearest neighbours classifier** using only NumPy, applied to the
[scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
(8×8 images of handwritten digits 0 to 9).

scikit-learn is used **only** to load the data and split it into train and test sets. The distance functions, the KNN class,
the metrics and the confusion matrix are written by hand.

## Result

| Metric | Value |
|---|---|
| Test accuracy (k = 5) | **98.3%** (354 / 360 images) |
| Digits with perfect precision and recall | 0, 2, 3, 5 |
| Weakest digit | 8 (precision 0.941, recall 0.914) |
| Best k in the sweep (evaluated on the test set) | 1 to 3 (98.6%) |

Settings: k = 5, Euclidean distance, raw pixel values (no scaling), 80/20 stratified split (`random_state=42`).
### In case of MNIST dataset accuracy is around 98%(with deskew and setting k = 5) but it is slow takes around 40 min in intel i5 210h.

## What's implemented

- **Distance functions:** Euclidean, Manhattan, Minkowski (vectorised with NumPy broadcasting)
- **`KNN` class** with `fit`, `get_neighbours`, `predict_lable` (majority vote) and `predict`
- **Evaluation:** accuracy, a 10×10 confusion matrix, and per-class precision and recall
- **k sweep:** accuracy for k = 1 to 99, plotted against k

## How it works

KNN is a lazy learner: there is no training step, the training data is simply stored. To classify a new image:

1. Compute its distance to every training image (each image is a vector of 64 pixel values).
2. Take the k closest images.
3. Predict the most common digit among them.

## Findings

- **No feature scaling.** All 64 features are pixel intensities on the same 0 to 16 range. In a separate check on this split,
  standardising the pixels lowered accuracy from about 98.3% to 96.4%, because dividing by the standard deviation inflates
  rarely-used border pixels. Scaling is worth testing, not applying by default.
- **Effect of k.** Accuracy is highest for very small k (about 98.6%), stays above 96% until k = 24, and falls to about 91% at k = 99
  as the vote is averaged over increasingly distant images.
- **Confusions.** Errors concentrate on the digits 8 and 9; most other digits are classified almost perfectly.

Requires Python 3.9 or newer. Run the notebook cells from top to bottom (the k sweep in section 9 runs 99 full predictions, so it takes a little while).

## Repository contents

```
.
├── knn_digits_from_scratch.ipynb   # full walkthrough: code, explanations and results
└── README.md
```

## Limitations

- **Small test set:** with 360 images, each extra mistake changes accuracy by about 0.28 percentage points.
- **Single split:** all numbers come from one train/test split.
- **k sweep on the test set:** it shows the trend but is not a valid way to choose k; use cross-validation on the training data for that.
- **Speed:** every prediction computes distances to all 1,437 training images, and predictions run in a Python loop.
- **Easy dataset:** 8×8 images are small and clean; larger datasets such as MNIST are much harder for plain pixel-distance KNN.

## Possible next steps

- [ ] Choose k with k-fold cross-validation
- [ ] Display the misclassified digits to see why 8 and 9 are confused
- [ ] Compare Euclidean, Manhattan and Minkowski distances, and distance-weighted voting
- [ ] Reduce the 64 dimensions with PCA before running KNN
- [ ] Validate against `sklearn.neighbors.KNeighborsClassifier`
- [ ] Vectorise prediction across all test images
