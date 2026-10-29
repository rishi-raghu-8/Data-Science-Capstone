"""
Week 6 Integrative Capstone: End-to-end data science pipeline.
Dataset: Wine recognition dataset distributed with scikit-learn.
Includes EDA, preprocessing, supervised model comparison, evaluation, and K-Means clustering.
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

SEED = 42
random.seed(SEED); np.random.seed(SEED)
OUT = Path("outputs")
PLOTS = OUT / "visualizations"
PLOTS.mkdir(parents=True, exist_ok=True)

wine = load_wine(as_frame=True)
X = wine.data.copy()
X.columns = [c.replace("/", "_per_").replace(" ", "_") for c in X.columns]
y = pd.Series(wine.target, name="wine_class")
target_names = list(wine.target_names)
data = X.copy()
data["wine_class"] = y.map(dict(enumerate(target_names)))
data.to_csv(OUT / "wine_dataset.csv", index=False)

print("Dataset shape:", X.shape)
print("Missing values:", int(X.isna().sum().sum()))
print("Duplicate feature rows:", int(X.duplicated().sum()))
print("\\nClass counts:\\n", data["wine_class"].value_counts())
print("\\nDescriptive statistics:\\n", X.describe().round(2).T)

# EDA: class counts and correlation
data["wine_class"].value_counts().reindex(target_names).plot(kind="bar")
plt.title("Wine Class Distribution"); plt.xlabel("Class"); plt.ylabel("Samples")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(PLOTS / "01_class_distribution.png", dpi=180); plt.close()

corr = X.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, label="Pearson correlation")
ax.set_xticks(range(len(corr.columns)), [c.replace("_", " ") for c in corr.columns], rotation=90, fontsize=7)
ax.set_yticks(range(len(corr.index)), [c.replace("_", " ") for c in corr.index], fontsize=7)
ax.set_title("Feature Correlation Matrix")
fig.tight_layout(); fig.savefig(PLOTS / "02_correlation_heatmap.png", dpi=180); plt.close(fig)

# Supervised: split first, then use CV only on training partition.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, random_state=SEED))
    ]),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced")
}
cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    cv_results[name] = scores
    print(f"{name} CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

best_name = max(cv_results, key=lambda name: cv_results[name].mean())
best_model = models[best_name]
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)
acc = accuracy_score(y_test, pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
cm = confusion_matrix(y_test, pred, labels=[0, 1, 2])
print("Selected model:", best_name)
print(f"Test accuracy: {acc:.4f}; macro precision: {precision:.4f}; macro recall: {recall:.4f}; macro F1: {f1:.4f}")

names = list(cv_results.keys())
means = [cv_results[n].mean() for n in names]
stds = [cv_results[n].std() for n in names]
plt.figure(figsize=(7, 4.5)); plt.bar(names, means, yerr=stds, capsize=5)
plt.ylim(0, 1.05); plt.ylabel("5-fold CV accuracy"); plt.title("Supervised Model Comparison")
plt.tight_layout(); plt.savefig(PLOTS / "04_model_comparison.png", dpi=180); plt.close()

fig, ax = plt.subplots(figsize=(5.5, 4.8))
ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix — {best_name}")
fig.tight_layout(); fig.savefig(PLOTS / "05_confusion_matrix.png", dpi=180); plt.close(fig)

# Unsupervised: standardize all features for exploratory segmentation.
Xs = StandardScaler().fit_transform(X)
ks = range(2, 8)
inertias, silhouettes = [], []
for k in ks:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=20)
    labels = km.fit_predict(Xs)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(Xs, labels))
kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=20)
clusters = kmeans.fit_predict(Xs)
sil = silhouette_score(Xs, clusters)
ari = adjusted_rand_score(y, clusters)  # post-hoc only; labels are not used to fit KMeans
result = data.copy()
result["cluster"] = clusters
result.to_csv(OUT / "wine_clustering_results.csv", index=False)

fig, ax = plt.subplots()
ax.plot(list(ks), inertias, marker="o")
ax.set(xlabel="Number of clusters (k)", ylabel="Inertia", title="K-Means Elbow Analysis")
fig.tight_layout(); fig.savefig(PLOTS / "06_elbow_method.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots()
ax.plot(list(ks), silhouettes, marker="o")
ax.set(xlabel="Number of clusters (k)", ylabel="Silhouette score", title="Silhouette Score by Cluster Count")
fig.tight_layout(); fig.savefig(PLOTS / "07_silhouette_scores.png", dpi=180); plt.close(fig)

pca = PCA(n_components=2, random_state=SEED)
Xp = pca.fit_transform(Xs)
fig, ax = plt.subplots()
for cl in range(3):
    mask = clusters == cl
    ax.scatter(Xp[mask, 0], Xp[mask, 1], label=f"Cluster {cl}", alpha=.8)
ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
       ylabel=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)",
       title="K-Means Clusters in PCA Projection")
ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "08_cluster_pca.png", dpi=180); plt.close(fig)

features_plot = ["alcohol", "malic_acid", "color_intensity", "flavanoids"]
result.groupby("cluster")[features_plot].mean().T.plot(kind="bar", figsize=(8, 4.5))
plt.title("Selected Feature Means by Cluster"); plt.ylabel("Mean value")
plt.xticks(rotation=20, ha="right"); plt.tight_layout()
plt.savefig(PLOTS / "09_cluster_profiles.png", dpi=180); plt.close()

metrics = {
    "selected_supervised_model": best_name,
    "test_accuracy": acc,
    "macro_precision": precision,
    "macro_recall": recall,
    "macro_f1": f1,
    "cv_accuracy_mean": cv_results[best_name].mean(),
    "cv_accuracy_std": cv_results[best_name].std(),
    "kmeans_k": 3,
    "kmeans_silhouette": sil,
    "kmeans_adjusted_rand_index_posthoc": ari,
    "missing_values": int(X.isna().sum().sum()),
    "duplicate_feature_rows": int(X.duplicated().sum())
}
pd.DataFrame([metrics]).to_csv(OUT / "evaluation_metrics.csv", index=False)
pred_out = X_test.copy()
pred_out["actual_class"] = y_test.map(dict(enumerate(target_names)))
pred_out["predicted_class"] = pd.Series(pred, index=y_test.index).map(dict(enumerate(target_names)))
pred_out.to_csv(OUT / "test_predictions.csv", index=True, index_label="row_index")
print("Clustering silhouette:", round(sil, 4))
print("Post-hoc ARI:", round(ari, 4))
print("Results saved to:", OUT.resolve())
