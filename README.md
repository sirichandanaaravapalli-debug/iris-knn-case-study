# Iris Species Classification with k-Nearest Neighbors

This repository contains the runnable experiment for the CLA-3 case study. It loads the Iris dataset bundled with scikit-learn, makes a stratified 80/20 train/test split (random_state=42), standardizes features, trains 5-nearest-neighbor classification, and prints accuracy, confusion matrix, class metrics, and mistakes.

## Run

```bash
python -m pip install scikit-learn
python iris_knn.py
```

Python 3.10+ is recommended. The code downloads no files; `load_iris()` uses the dataset shipped with scikit-learn.

## Reproducibility

The split and model settings are fixed in the script. Expected test accuracy is 0.9333 (28/30) with confusion matrix `[[10,0,0],[0,10,0],[0,2,8]]`, with rows and columns ordered setosa, versicolor, virginica.
