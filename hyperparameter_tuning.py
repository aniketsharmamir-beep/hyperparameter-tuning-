"""
Hyperparameter tuning for a Random Forest on the engineered Titanic features.

Steps:
1. Load the data and split it 80/20
2. Train a baseline Random Forest with default settings
3. Tune it with Grid Search (tries every combination)
4. Tune it with Randomized Search (tries random combinations)
5. Compare baseline vs tuned models on the unseen test set
"""

import os
import time

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

DATA_PATH = "data/engineered_features.csv"
RANDOM_STATE = 42


def evaluate(name, model, X_test, y_test, cv_score, seconds=0.0):
    """Collect the scores of one model in a dictionary."""
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "CV Accuracy": round(cv_score, 3),
        "Test Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Test F1-score": round(f1_score(y_test, y_pred), 3),
        "Search Time (s)": round(seconds, 1),
    }


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns="Survived")
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 5-fold cross-validation acts as our validation set during tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    # 1. Baseline: default hyperparameters
    baseline = RandomForestClassifier(random_state=RANDOM_STATE)
    base_cv = cross_val_score(baseline, X_train, y_train, cv=cv).mean()
    baseline.fit(X_train, y_train)
    results.append(evaluate("Baseline (default)", baseline, X_test, y_test, base_cv))

    # 2. Grid Search: tries every combination in the grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    start = time.time()
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    grid_time = time.time() - start
    print("Grid Search best parameters:", grid.best_params_)
    results.append(
        evaluate("Grid Search", grid.best_estimator_, X_test, y_test,
                 grid.best_score_, grid_time)
    )

    # 3. Randomized Search: tries 30 random combinations from wider ranges
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 15),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
    }
    start = time.time()
    rand = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_dist, n_iter=30, cv=cv, scoring="accuracy",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rand.fit(X_train, y_train)
    rand_time = time.time() - start
    print("Randomized Search best parameters:", rand.best_params_)
    results.append(
        evaluate("Randomized Search", rand.best_estimator_, X_test, y_test,
                 rand.best_score_, rand_time)
    )

    # Compare everything
    results = pd.DataFrame(results).set_index("Model")
    print("\n", results.to_string())

    os.makedirs("images", exist_ok=True)
    results.to_csv("data/tuning_results.csv")

    # How does max_depth affect the score? (from the grid search results)
    grid_df = pd.DataFrame(grid.cv_results_)
    grid_df["max_depth"] = grid_df["param_max_depth"].map(
        lambda v: "None" if pd.isna(v) else str(int(v))
    )
    depth_scores = grid_df.groupby("max_depth")["mean_test_score"].mean()
    depth_scores = depth_scores.reindex(["3", "5", "7", "None"])
    print("\nAverage CV accuracy by max_depth:")
    print(depth_scores.round(3))

    depth_scores.plot(kind="bar", figsize=(6, 4))
    plt.title("Effect of max_depth on CV Accuracy")
    plt.xlabel("max_depth")
    plt.ylabel("Average CV Accuracy")
    plt.ylim(0.75, 0.85)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("images/max_depth_effect.png")


if __name__ == "__main__":
    main()
