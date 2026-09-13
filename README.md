# Hierarchical Clustering

## Overview

This project applies **Hierarchical Clustering** to segment mall customers into distinct groups based on their **Annual Income** and **Spending Score**. It is part of the *Machine Learning A-Z* course (Part 4 – Clustering, Section 25).

---

## Dataset

**File:** `Mall_Customers.csv`

The dataset contains information about mall customers, including:

| Column | Description |
|---|---|
| CustomerID | Unique customer identifier |
| Genre | Customer gender |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousands of dollars |
| Spending Score (1-100) | Score assigned by the mall based on customer behavior and spending |

The model uses columns 4 and 5 — **Annual Income** and **Spending Score** — as features.

---

## Workflow

### 1. Import Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### 2. Load the Dataset
```python
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values
```

### 3. Find the Optimal Number of Clusters (Dendrogram)
A **dendrogram** is built using Ward's linkage method to determine the optimal number of clusters by identifying the largest vertical gap:
```python
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
```

### 4. Train the Hierarchical Clustering Model
Using `AgglomerativeClustering` from scikit-learn with **5 clusters**, Euclidean affinity, and Ward linkage:
```python
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)
```

### 5. Visualise the Clusters
Each cluster is plotted in a distinct colour:
```python
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red',     label='Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='blue',    label='Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='green',   label='Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan',    label='Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
```

---

## Results

The dendrogram suggests **5 optimal clusters**, grouping customers by their spending behaviour relative to income:

| Cluster | Income Level | Spending Score | Profile |
|---|---|---|---|
| 1 | Medium | Medium | Average customers |
| 2 | High | High | Target customers (big spenders) |
| 3 | Low | High | Spending beyond means |
| 4 | High | Low | Careful high earners |
| 5 | Low | Low | Budget-conscious customers |

---

## Requirements

| Library | Purpose |
|---|---|
| `numpy` | Numerical operations |
| `matplotlib` | Data visualisation |
| `pandas` | Data loading and manipulation |
| `scipy` | Dendrogram construction |
| `scikit-learn` | Agglomerative clustering model |

Install dependencies:
```bash
pip install numpy matplotlib pandas scipy scikit-learn
```

---

## How to Run

1. Open `hierarchical_clustering.ipynb` in Jupyter Notebook or Google Colab.
2. Ensure `Mall_Customers.csv` is in the same directory.
3. Run all cells in order.

---

## Files

```
.
├── hierarchical_clustering.ipynb   # Main notebook
├── Mall_Customers.csv              # Dataset
└── README.md                       # Project documentation
```
