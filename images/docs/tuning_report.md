# Hyperparameter Tuning Report

## Objective
Improve the performance of a Random Forest classifier on the engineered Titanic dataset using Grid Search and Randomized Search.

## Method
- Data: engineered_features.csv, split 80% training and 20% test (random_state=42, stratified)
- Model: Random Forest Classifier
- Validation: 5-fold cross-validation on the training data
- Metric used for tuning: accuracy

## Hyperparameters Tuned
- n_estimators: number of trees in the forest
- max_depth: how deep each tree can grow
- min_samples_split: minimum samples needed to split a node
- min_samples_leaf: minimum samples required in a leaf

## Search Strategies
1. Grid Search: tried all 72 combinations from a fixed grid (n_estimators 100/200, max_depth 3/5/7/None, min_samples_split 2/5/10, min_samples_leaf 1/2/4).
2. Randomized Search: tried 30 random combinations from wider ranges.

## Results
| Model | CV Accuracy | Test Accuracy | Test F1 | Search Time |
|---|---|---|---|---|
| Baseline (default) | 0.812 | 0.816 | 0.756 | - |
| Grid Search | 0.834 | 0.816 | 0.740 | 64 s |
| Randomized Search | 0.829 | 0.827 | 0.763 | 51 s |

Best parameters:
- Grid Search: max_depth=None, min_samples_leaf=2, min_samples_split=10, n_estimators=100
- Randomized Search: max_depth=14, min_samples_leaf=2, min_samples_split=11, n_estimators=369

## Effect of max_depth
Average cross-validation accuracy was 0.822 for depth 3, 0.815 for depth 5, 0.818 for depth 7, and 0.824 for no limit (see images/max_depth_effect.png). The differences are small, so depth alone does not change the result much.

## Conclusion
Tuning raised the cross-validation accuracy from 0.812 to about 0.83, so both searches found better settings than the defaults. On the unseen test set the gain was small: Randomized Search improved accuracy from 0.816 to 0.827, while Grid Search stayed at 0.816. The cross-validation score after tuning is a little optimistic, because the best setting was picked from many tries. Randomized Search is the chosen model: it has the best test accuracy and F1-score and was also faster. With only 179 test passengers, a difference of about 1% is within normal noise, so the real gain from tuning is modest. The default Random Forest was already a strong model for this dataset.
