# Lab 2: Classification Using KNN and RNN Algorithms

**Student:** Minh Le  
**Course:** 2026 Fall - Advanced Big Data and Data Mining (MSCS-634-M50) - Full Term

## Purpose

This lab evaluates K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) classifiers on scikit-learn's Wine dataset. It explores the dataset, creates a reproducible stratified 80/20 train-test split, measures test accuracy across the assigned parameter values, visualizes both accuracy trends, and compares when each neighbor-based method may be preferable.

## Files

- `MSCS_634_Lab_2.ipynb` — complete analysis with executed outputs and plots
- `requirements.txt` — pinned Python dependencies needed to reproduce the notebook

## Method

- Dataset: scikit-learn Wine dataset (178 observations, 13 numeric features, 3 classes)
- Split: 80% training and 20% testing, stratified by class
- Reproducibility: `random_state=42`
- KNN values: 1, 5, 11, 15, and 21
- RNN radii: 350, 400, 450, 500, 550, and 600
- Metric: test-set accuracy

The features are intentionally left on their original scales because the assigned RNN radii correspond to those raw distances. This means high-magnitude features—especially `proline`—have greater influence on Euclidean distance.

## Results and key insights

| Model | Parameter | Test accuracy |
|---|---:|---:|
| KNN | k = 1 | 77.78% |
| KNN | k = 5 | 80.56% |
| KNN | k = 11 | 80.56% |
| KNN | k = 15 | 80.56% |
| KNN | k = 21 | 80.56% |
| RNN | radius = 350 | 72.22% |
| RNN | radius = 400 | 69.44% |
| RNN | radius = 450 | 69.44% |
| RNN | radius = 500 | 69.44% |
| RNN | radius = 550 | 66.67% |
| RNN | radius = 600 | 66.67% |

KNN's maximum test accuracy was 80.56%. Values from k = 5 through k = 21 tied for that maximum, so k = 5 is identified as the first and smallest best-performing tested value. RNN performed best at radius 350 with 72.22%. Its accuracy declined as the radius grew because larger neighborhoods included increasingly distant observations and blurred class boundaries. No tested RNN radius left a test observation without neighbors.

For this split and parameter grid, the best KNN result exceeded the best RNN result by 8.33 percentage points. KNN is attractive when every observation should receive a prediction and local sample density varies. RNN can be preferable when the problem supplies a meaningful distance threshold and an adaptive neighbor count is desirable, but it is especially sensitive to radius selection and feature scale.

## Challenges and decisions

- The feature ranges differ substantially, but scaling would make the required RNN radii of 350–600 inappropriate. The lab therefore uses raw features and documents that choice.
- RNN can encounter test observations with no neighbors. The notebook records minimum neighbor counts and zero-neighbor counts for every radius and includes a safe fallback label; the fallback was not needed for these experiments.
- A single test split is useful for the requested comparison but does not establish general performance. A stronger follow-up would combine `StandardScaler` with each model in a pipeline, retune the parameters through stratified cross-validation on the training set, and evaluate the selected model once on the untouched test set.

## Run the notebook

Using Python 3.13 or another current Python 3 version:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
jupyter notebook MSCS_634_Lab_2.ipynb
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.
