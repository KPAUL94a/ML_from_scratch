# Multiple Linear Regression from Scratch (California Housing)

Predicts `median_house_value` using multiple linear regression implemented with NumPy (Normal Equation: θ = (XᵀX)⁻¹Xᵀy).

## Result
Test R² ≈ **0.64**

## Files
- `linear_regression.ipynb`: full notebook with explanations
- `housing.csv`: dataset (add it to the same folder; not included here)

## Run
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook linear_regression.ipynb
```

## Pipeline
1. Load data and drop rows with missing `total_bedrooms`
2. 80/20 train/test split (`random_state=42`)
3. Standardize features using training statistics only
4. Add bias column and solve the Normal Equation
5. Evaluate with R²

## Possible improvements
One-hot encode `ocean_proximity`, engineer ratio features, log-transform skewed columns, remove the capped target values, try Ridge/Lasso and tree-based models.
