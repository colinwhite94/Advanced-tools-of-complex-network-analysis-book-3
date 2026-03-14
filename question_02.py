# load zachary karate club graph
# extract club labels as target variable y
# build node level feature matrix X with 6 features:
#  four structural: degree, clustering coefficient, betweenness centrality, pagerank
#  two synthetic: attendance (sampled from gaussian conditional on label), tenure (same)
# perform R=50 stratified 80/20 train/test splits with different random seeds
# for each split: fit a random forest classifier and record test accuracy and macro-f1
# report mean and std of accuracy and macro-f1 across all 50 splits
# show confusion matrix for one representative split (seed=42)

# run with: python question_02.py

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

# load graph and labels
G = nx.karate_club_graph()
nodes = list(G.nodes())

labels = [G.nodes[n]["club"] for n in nodes]
y = np.array([0 if lab == "Mr. Hi" else 1 for lab in labels])
label_names = ["Mr. Hi", "Officer"]

# structural features
degree = dict(G.degree())
clustering = nx.clustering(G)
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)

# synthetic features
# attendance: Mr. Hi:  N(7, 1.5),  Officer: N(4, 1.5)
# tenure: Mr. Hi: N(3, 1.0),  Officer: N(6, 1.0)
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

# assemble feature matrix
feature_names = [
    "degree",
    "clustering",
    "betweenness",
    "pagerank",
    "attendance",
    "tenure"
]

X = np.array([
    [
        degree[n],
        clustering[n],
        betweenness[n],
        pagerank[n],
        attendance[i],
        tenure[i]
    ]
    for i, n in enumerate(nodes)
])

print(f"feature matrix shape: {X.shape}")
print(f"class distribution  : Mr. Hi={sum(y==0)}, Officer={sum(y==1)}\n")

# R=50 stratified 80/20 splits
R = 50
REPRESENTATIVE_SEED = 42

accuracies = []
f1_scores  = []
rep_result = {}

for seed in range(R):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = clf.score(X_test, y_test)
    f1  = f1_score(y_test, y_pred, average="macro")

    accuracies.append(acc)
    f1_scores.append(f1)

    # store representative split results
    if seed == REPRESENTATIVE_SEED:
        rep_result["y_test"]      = y_test
        rep_result["y_pred"]      = y_pred
        rep_result["importances"] = clf.feature_importances_
        rep_result["acc"]         = acc
        rep_result["f1"]          = f1

# report mean and std
print(f"results across R={R} splits (80/20 stratified):")
print(f"  accuracy  : mean={np.mean(accuracies):.3f}, std={np.std(accuracies):.3f}")
print(f"  macro-f1  : mean={np.mean(f1_scores):.3f}, std={np.std(f1_scores):.3f}")
print(f"\nrepresentative split (seed={REPRESENTATIVE_SEED}):")
print(f"  accuracy  : {rep_result['acc']:.3f}")
print(f"  macro-f1  : {rep_result['f1']:.3f}\n")

# plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

# plot 1: accuracy distribution across 50 splits
axes[0].hist(accuracies, bins=10, color="gray", edgecolor="black")
axes[0].axvline(np.mean(accuracies), color="black", linestyle="--", linewidth=1.5,
                label=f"mean={np.mean(accuracies):.3f}")
axes[0].set_title("test accuracy across 50 splits", fontsize=12, fontfamily="serif")
axes[0].set_xlabel("accuracy", fontsize=10, fontfamily="serif")
axes[0].set_ylabel("count", fontsize=10, fontfamily="serif")
axes[0].legend(fontsize=9)

# plot 2: confusion matrix for representative split
cm = confusion_matrix(rep_result["y_test"], rep_result["y_pred"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(ax=axes[1], colorbar=False, cmap="Greys")
axes[1].set_title(f"confusion matrix (seed={REPRESENTATIVE_SEED})",
                  fontsize=12, fontfamily="serif")

# plot 3: feature importances for representative split
importances = pd.Series(rep_result["importances"], index=feature_names).sort_values()
importances.plot.barh(ax=axes[2], color="black", edgecolor="white")
axes[2].set_title(f"feature importances (seed={REPRESENTATIVE_SEED})",
                  fontsize=12, fontfamily="serif")
axes[2].set_xlabel("mean decrease in impurity", fontsize=10, fontfamily="serif")
axes[2].tick_params(labelsize=9)

plt.tight_layout()
plt.savefig("question_02.png", dpi=300, bbox_inches="tight")
print("saved question_02.png")