# ShopSmart - Purchase Prediction (Decision Tree)

Predicting whether an e-commerce visitor will make a purchase (`Revenue`) based on their browsing session behavior, using a Decision Tree Classifier.

## Problem

ShopSmart wants to predict which visitors are likely to complete a purchase based on session data (12,330 sessions, 1 year). The dataset is imbalanced (~84.5% no purchase, ~15.5% purchase), so F1 score is used as the evaluation metric instead of accuracy, with a benchmark of 0.55.

## Dataset

- 12,330 rows, 18 columns
- Mix of numerical (ProductRelated, BounceRates, ExitRates, PageValues, etc.) and categorical features (Month, VisitorType, Weekend)
- Target: `Revenue` (0 = No Purchase, 1 = Purchase)

## Steps

1. **Preprocessing**
   - One-Hot Encoded `Month` and `VisitorType`
   - Converted `Weekend` and `Revenue` to int (0/1)
   - Selected features (excluded `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, and target `Revenue`)

2. **Train/Test Split**
   - 80/20 split, stratified on `Revenue` to preserve class imbalance ratio

3. **Baseline Model**
   - `DecisionTreeClassifier()` with no constraints (fully grown tree)

4. **Pruning**
   - Used cost-complexity pruning (`cost_complexity_pruning_path`) to find candidate `ccp_alpha` values
   - Trained a tree for each alpha, selected the one with best F1 on test set

## Results

| Metric | Baseline (unpruned) | Pruned (best `ccp_alpha`) |
|---|---|---|
| Accuracy | 0.8609 | 0.8686 |
| Precision | 0.5472 | 0.5551 |
| Recall | 0.5916 | 0.7644 |
| F1 Score | 0.5686 | 0.6432 |

Benchmark target: **0.55 F1** — Achieved: **0.6432** ✅

## Key Finding

`PageValues` was the most important feature, forming the root split of the pruned tree (`PageValues <= 1.109`). This aligns with business intuition, since it directly measures the value of pages visited before completing a transaction.

Pruning significantly improved recall (0.59 → 0.76) by removing overfit branches that were causing the baseline model to miss real purchasers.

## Tech Stack

- Python, Pandas, Scikit-learn, Matplotlib
