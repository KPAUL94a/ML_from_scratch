# Machine Learning Algorithms from Scratch

Implementations of core machine learning algorithms using only **NumPy** for the maths, written to understand how they work under the hood. Libraries like scikit-learn are used only for utilities such as data splitting or metric checks, never for the models themselves.

Each notebook walks through the theory, the implementation and the evaluation, with comments and explanations.

## Algorithms

| Algorithm | Dataset | Technique | Result |
|---|---|---|---|
| [Multiple Linear Regression](linear_regression.ipynb) | California Housing | Normal Equation | R² ≈ 0.64 |
| [Logistic Regression](astc_final_2.ipynb) | Astronomical objects (star vs galaxy) | Gradient descent, L2 regularization, polynomial features | Accuracy ≈ 95.7% |

More algorithms coming soon.

## Getting started

```bash
git clone https://github.com/<KPAUL94a>/<ML_from_scratch>.git
cd <ML_from_scratch>
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook
```

Datasets are included.

## Planned

- [ ] k-Nearest Neighbors
- [ ] Decision Trees
- [ ] k-Means Clustering
- [ ] Naive Bayes
- [ ] Neural Network (backpropagation)

## Contributing

Suggestions and improvements are welcome. Open an issue or a pull request.
