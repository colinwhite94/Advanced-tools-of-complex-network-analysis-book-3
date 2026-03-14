# robustness test: refit model dropping 2 features
# dropped: tenure (rank 1, 0.44) and clustering (rank 3, 0.08)
# kept: attendance, pagerank, betweenness, degree
# perform same R=50 stratified 80/20 splits and compare to full model

# run with: python question_02_robustness.py

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# load graph and labels
G = nx.karate_club_graph()
nodes = list(G.nodes())

labels = [G.nodes[n]["club"] for n in nodes]
y = np.array([0 if lab == "Mr. Hi" else 1 for lab in labels])

# all structural features
degree = dict(G.degree())
clustering = nx.clustering(G)
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)

# synthetic features
np.random.seed(0)
attendance = np.where(
    y == 0,
    np.random.normal(7, 1.5, size=len(nodes)),
    np.random.normal(4, 1.5, size=len(nodes))
)
tenure = np.where(
    y == 0,
    np.random.normal(3, 1.0, size=len(nodes)),
    np.random.normal(6, 1.0, size=len(nodes))
)

# full feature matrix (6 features)
X_full = np.array([
    [degree[n], clustering[n], betweenness[n], pagerank[n], attendance[i], tenure[i]]
    for i, n in enumerate(nodes)
])

# reduced feature matrix (drop tenure + clustering)
# dropped:tenure (rank 1), clustering (rank 3)
# kept: attendance, pagerank, betweenness, degree
feature_names_reduced = ["degree", "betweenness", "pagerank", "attendance"]

X_reduced = np.array([
    [degree[n], betweenness[n], pagerank[n], attendance[i]]
    for i, n in enumerate(nodes)
])

print("dropped features: tenure (rank 1), clustering (rank 3)")
print("kept features:", feature_names_reduced)
print(f"reduced matrix shape: {X_reduced.shape}\n")

# R=50 splits for both models
R = 50

acc_full = []
f1_full = []
acc_reduced = []
f1_reduced = []

for seed in range(R):
    # full model
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y, test_size=0.20, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc_full.append(clf.score(X_te, y_te))
    f1_full.append(f1_score(y_te, y_pred, average="macro"))

    # reduced model
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reduced, y, test_size=0.20, random_state=seed, stratify=y)
    clf_r = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf_r.fit(X_tr_r, y_tr_r)
    y_pred_r = clf_r.predict(X_te_r)
    acc_reduced.append(clf_r.score(X_te_r, y_te_r))
    f1_reduced.append(f1_score(y_te_r, y_pred_r, average="macro"))

#comparison report
print("=" * 58)
print(f"{'model':<28} {'acc mean':>8} {'acc std':>8} {'f1 mean':>8} {'f1 std':>8}")
print("=" * 58)
print(f"{'full (6 features)':<28} {np.mean(acc_full):>8.3f} {np.std(acc_full):>8.3f} {np.mean(f1_full):>8.3f} {np.std(f1_full):>8.3f}")
print(f"{'reduced (4 features)':<28} {np.mean(acc_reduced):>8.3f} {np.std(acc_reduced):>8.3f} {np.mean(f1_reduced):>8.3f} {np.std(f1_reduced):>8.3f}")
print("=" * 58)

# side by side accuracy distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

bins = np.linspace(0.60, 1.01, 14)

axes[0].hist(acc_full, bins=bins, color="black", edgecolor="white")
axes[0].axvline(np.mean(acc_full), color="gray", linestyle="--", linewidth=1.5,
                label=f"mean={np.mean(acc_full):.3f}")
axes[0].set_title("full model\n(degree, clustering, betweenness, pagerank, attendance, tenure)",
                  fontsize=10, fontfamily="serif")
axes[0].set_xlabel("test accuracy", fontsize=10, fontfamily="serif")
axes[0].set_ylabel("count", fontsize=10, fontfamily="serif")
axes[0].legend(fontsize=9)
axes[0].set_xlim(0.60, 1.01)

axes[1].hist(acc_reduced, bins=bins, color="gray", edgecolor="white")
axes[1].axvline(np.mean(acc_reduced), color="black", linestyle="--", linewidth=1.5,
                label=f"mean={np.mean(acc_reduced):.3f}")
axes[1].set_title("reduced model\n(dropped tenure rank 1, clustering rank 3)",
                  fontsize=10, fontfamily="serif")
axes[1].set_xlabel("test accuracy", fontsize=10, fontfamily="serif")
axes[1].set_ylabel("count", fontsize=10, fontfamily="serif")
axes[1].legend(fontsize=9)
axes[1].set_xlim(0.60, 1.01)

plt.tight_layout()
plt.savefig("question_02_robustness.png", dpi=300, bbox_inches="tight")
print("\nsaved question_02_robustness.png")