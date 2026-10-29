# Week 6: Integrative Data Science Capstone

This end-to-end project uses the public Wine Recognition dataset distributed with scikit-learn. It demonstrates data acquisition, cleaning, exploratory data analysis, supervised classification, unsupervised clustering, model evaluation, and practical interpretation.

## Files
- `Week6_Capstone_Report.docx` — comprehensive capstone report
- `capstone_analysis.py` — reproducible analysis pipeline
- `wine_dataset.csv` — dataset with readable class labels
- `wine_clustering_results.csv` — K-Means cluster assignments
- `evaluation_metrics.csv` — supervised and unsupervised metrics
- `test_predictions.csv` — held-out predictions
- `visualizations/` — EDA and model evaluation figures
- `requirements.txt` — dependencies

## Run on Windows PowerShell
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python capstone_analysis.py
```

## Methods
- EDA: distributions, descriptive statistics, missing/duplicate checks, correlation analysis
- Supervised: Logistic Regression and Random Forest compared using stratified 5-fold cross-validation; best mean CV accuracy selected and evaluated on a held-out test set
- Unsupervised: StandardScaler + K-Means, elbow and silhouette diagnostics, PCA visualization, cluster profiling
- Reproducibility: fixed random seed (42)

## Dataset source
The Wine Recognition dataset is distributed with scikit-learn and is based on a public UCI dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html

Note: Cluster labels are arbitrary. ARI is computed only after clustering for comparison with known labels; the labels are not used to train K-Means.
