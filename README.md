# Unsupervised Learning

## Objective
This repository contains analyses of two datasets using unsupervised learning techniques, including dimensionality reduction methods (PCA, ICA, RP, and Isomap) and clustering algorithms (K-Means and Expectation Maximization). The goal is to explore and visualize data, identify meaningful clusters, and evaluate the effectiveness of different techniques.

## Technologies Used
- Python 3.11.9

## Analyses Performed
- Dimensionality Reduction:
  - Principal Component Analysis (PCA)
  - Independent Component Analysis (ICA)
  - Randomized Projection (RP)
  - Isomap for visualization
- Clustering Analysis:
  - K-Means Clustering
  - Expectation Maximization (EM)
- Evaluation of cluster quality using metrics such as silhouette scores.
- Visualization of high-dimensional data and clustering results.

## Libraries Used
The following libraries are required and can be installed via `pip`:
- `ucimlrepo` (to retrieve datasets from the UCI Machine Learning Repository)
- `numpy`
- `pandas`
- `random`
- `matplotlib`
- `sklearn`
- `scipy`
- `time`

## Datasets
The following datasets from the UCI Machine Learning Repository were used in this analysis:
- [Drug Consumption (Quantified)](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)
- [Iranian Churn Dataset](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)

## How to Replicate
1. Clone the repository:
   ```bash
   git clone https://github.com/AdrianKuklaPL/unsupervised-learning.git
