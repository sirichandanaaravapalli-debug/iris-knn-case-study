from pathlib import Path
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.20, random_state=42, stratify=iris.target
    )
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Dataset: {len(iris.data)} rows; train={len(X_train)}, test={len(X_test)}")
    print("Features:", ", ".join(iris.feature_names))
    print("Classes:", ", ".join(iris.target_names))
    print("Accuracy:", f"{accuracy_score(y_test, predictions):.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions))
    print("Classification report:")
    print(classification_report(y_test, predictions, target_names=iris.target_names, digits=3))
    print("Misclassified test examples:")
    for i, (actual, predicted) in enumerate(zip(y_test, predictions)):
        if actual != predicted:
            print(f"  test-row={i}, measurements={X_test[i].tolist()}, "
                  f"actual={iris.target_names[actual]}, predicted={iris.target_names[predicted]}")


if __name__ == "__main__":
    main()
