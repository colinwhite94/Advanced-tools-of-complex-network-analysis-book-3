# visualize the logistic distance-decay random geometric graph model
# p_ij = sigma(alpha - beta * d_ij), sigma(z) = 1 / (1 + exp(-z))
# show a grid of graphs across different (alpha, beta) combinations
# include a beta=0 panel to illustrate erdos-renyi equivalence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)
N = 300

# sample node positions uniformly from unit square
pos = np.random.uniform(0, 1, size=(N, 2))

# compute all pairwise distances
diff = pos[:, None, :] - pos[None, :, :] # n x n x 2
D = np.sqrt((diff ** 2).sum(axis=2)) # n x n euclidean distance matrix

# logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# sample adjacency matrix from logistic rgg
def sample_graph(alpha, beta):
    """sample adjacency matrix from logistic RGG"""
    P = sigmoid(alpha - beta * D)# n x n connection probabilities
    np.fill_diagonal(P, 0)# no self loops
    U = np.random.uniform(0, 1, size=(N, N))# uniform random draws
    A = (U < P).astype(int)
    A = np.triu(A, 1)# upper triangle only
    A = A + A.T# make symmetric
    return A

# extract edge list from adjacency matrix
def get_edges(A):
    rows, cols = np.where(np.triu(A, 1))
    return list(zip(rows, cols))

# parameter grid 
# rows: beta (distance sensitivity)
# cols: alpha (baseline density)
alphas = [-2,  0,  3]
betas  = [ 0,  5, 20]

alpha_labels = [f"α = {a}" for a in alphas]
beta_labels  = [f"β = {b}" for b in betas]

# plot 
fig = plt.figure(figsize=(13, 13), dpi=300)
gs  = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.15)

for row, beta in enumerate(betas):
    for col, alpha in enumerate(alphas):

        ax = fig.add_subplot(gs[row, col])
        A  = sample_graph(alpha, beta)
        edges = get_edges(A)
        n_edges = len(edges)
        density = n_edges / (N * (N - 1) / 2)

        # draw edges first (thin, gray)
        for i, j in edges:
            ax.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                color="gray", alpha=0.15, linewidth=0.3, zorder=1
            )

        # draw nodes on top
        ax.scatter(pos[:, 0], pos[:, 1],
                   s=4, color="black", zorder=2)

        # title with parameter values and stats
        if beta == 0 and col == 1:
            note = "\n(Erdős–Rényi: distance ignored)"
        else:
            note = ""

        ax.set_title(
            f"α={alpha}, β={beta}{note}\n"
            f"edges={n_edges}, density={density:.3f}",
            fontsize=8, fontfamily="serif"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        # row and column labels on outer edges
        if col == 0:
            ax.set_ylabel(beta_labels[row], fontsize=10,
                          fontfamily="serif", labelpad=8)
        if row == 0:
            ax.set_title(
                alpha_labels[col] + "\n\n" +
                f"α={alpha}, β={beta}{note}\n"
                f"edges={n_edges}, density={density:.3f}",
                fontsize=8, fontfamily="serif"
            )

# overall labels
fig.text(0.5, 0.98, "logistic distance-decay random geometric graph",
         ha="center", va="top", fontsize=14, fontfamily="serif", fontweight="bold")
fig.text(0.5, 0.01, "← increasing α (baseline density) →",
         ha="center", fontsize=11, fontfamily="serif")
fig.text(0.01, 0.5, "← increasing β (distance sensitivity) →",
         va="center", rotation="vertical", fontsize=11, fontfamily="serif")

plt.savefig("question_03_a.png", dpi=300, bbox_inches="tight")
print("saved question_03_a.png")