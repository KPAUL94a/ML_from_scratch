# Logistic Regression from Scratch: Stars vs Galaxies

Classifies astronomical objects (0 = star, 1 = galaxy) with logistic regression implemented in pure NumPy: sigmoid, polynomial feature expansion, L2 regularization and batch gradient descent. Metrics are also computed by hand.

## Result
Test accuracy ≈ **95.7%**, F1 ≈ **0.956** (precision 0.950, recall 0.962, log-loss 0.167).

## Files
- `Logistic_regression.ipynb`: full notebook with explanations
- `astronomical_objects.csv`: dataset (add it to the same folder; not included here)

## Run
```bash
pip install numpy pandas matplotlib jupyter
jupyter notebook astc_final_2.ipynb
```

## Pipeline
1. Load data (7209 rows, 10 features + label), drop rows with missing values and duplicates (5111 rows remain)
2. Seeded 80/20 train/test split
3. Standardize features using training statistics only
4. Expand features with squares and pairwise products (10 → 65), scale again, add bias column
5. Train with gradient descent (learning rate 0.5, L2 λ = 1.0, 5001 iterations)
6. Evaluate with a confusion matrix, accuracy, precision, recall, F1 and log-loss

## Possible improvements
Tune λ, learning rate and iterations with cross-validation, impute missing values instead of dropping them, add ROC-AUC, and compare against `sklearn.linear_model.LogisticRegression`.
