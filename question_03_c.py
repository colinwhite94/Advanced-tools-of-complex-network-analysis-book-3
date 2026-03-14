# sample two soft rggs with the same alpha but two different beta values
# compute for each graph:
# m = total number of edges (sum over i < j of a_ij)
# k= mean degree = 2m / n
# d_edge = mean edge length = (1/m) * sum over i<j of a_ij * d_ij
# visualize both graphs side by side with nodes at spatial positions

# run with: python question_03_c.py

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 300

# sample node positions uniformly from unit square
pos = np.random.uniform(0, 1, size=(N, 2))

# compute full pairwise distance matrix
diff = pos[:, None, :] - pos[None, :, :] # n x n x 2
D = np.sqrt((diff ** 2).sum(axis=2)) # n x n

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sample_soft_rgg(pos, alpha, beta):
    """
    sample a soft rgg using the logistic distance decay model.
    p_ij = sigma(alpha - beta * d_ij), sigma(z) = 1 / (1 + exp(-z))

    parameters
    alpha : controls baseline edge density
    beta : controls how strongly distance suppresses edges

    returns
    A : (n, n)  adjacency matrix with zeros on diagonal
    """
    N = len(pos)

    # (i) compute d_ij for all i < j
    diff = pos[:, None, :] - pos[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2))

    # (ii) convert distances to connection probabilities p_ij
    P = sigmoid(alpha - beta * D)
    np.fill_diagonal(P, 0)

    # (iii) sample edges
    U = np.random.uniform(0, 1, size=(N, N))
    A = (U < P).astype(int)
    A = np.triu(A, 1)
    A = A + A.T

    return A

def compute_stats(A, D, N):
    """compute m, mean degree, and mean edge length"""
    # upper triangle indices where edges exist
    rows, cols = np.where(np.triu(A, 1))

    m = len(rows)# total edges
    mean_k = 2 * m / N # mean degree
    mean_d = D[rows, cols].mean() if m > 0 else 0 # mean edge length

    return m, mean_k, mean_d

def get_edges(A):
    rows, cols = np.where(np.triu(A, 1))
    return list(zip(rows, cols))

# parameters
ALPHA   = 0 # same alpha for both graphs
BETA_LO = 2# small beta: weak distance suppression
BETA_HI = 15 # large beta: strong distance suppression

# sample graphs
A_lo = sample_soft_rgg(pos, ALPHA, BETA_LO)
A_hi = sample_soft_rgg(pos, ALPHA, BETA_HI)

# compute stats
m_lo, k_lo, d_lo = compute_stats(A_lo, D, N)
m_hi, k_hi, d_hi = compute_stats(A_hi, D, N)

print(f"alpha = {ALPHA} (same for both)\n")
print(f"{'metric':<20} {'beta='+str(BETA_LO):>12} {'beta='+str(BETA_HI):>12}")
print("-" * 46)
print(f"{'m (edges)':<20} {m_lo:>12d} {m_hi:>12d}")
print(f"{'<k> (mean degree)':<20} {k_lo:>12.3f} {k_hi:>12.3f}")
print(f"{'<d_edge> (mean len)':<20} {d_lo:>12.4f} {d_hi:>12.4f}")

# visualize both graphs side by side
fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=300)

for ax, A, beta, m, mean_k, mean_d in [
    (axes[0], A_lo, BETA_LO, m_lo, k_lo, d_lo),
    (axes[1], A_hi, BETA_HI, m_hi, k_hi, d_hi),
]:
    edges = get_edges(A)

    # draw edges
    for i, j in edges:
        ax.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            color="gray", alpha=0.2, linewidth=0.3, zorder=1
        )

    # draw nodes
    ax.scatter(pos[:, 0], pos[:, 1], s=5, color="black", zorder=2)

    ax.set_title(
        f"α = {ALPHA},  β = {beta}\n"
        f"m = {m},   ⟨k⟩ = {mean_k:.2f},   ⟨d_edge⟩ = {mean_d:.4f}",
        fontsize=11, fontfamily="serif"
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=9, fontfamily="serif")
    ax.set_ylabel("y", fontsize=9, fontfamily="serif")

axes[0].set_title(
    f"small β = {BETA_LO}  (weak distance decay)\n"
    f"α = {ALPHA},  m = {m_lo},   ⟨k⟩ = {k_lo:.2f},   ⟨d_edge⟩ = {d_lo:.4f}",
    fontsize=11, fontfamily="serif"
)
axes[1].set_title(
    f"large β = {BETA_HI}  (strong distance decay)\n"
    f"α = {ALPHA},  m = {m_hi},   ⟨k⟩ = {k_hi:.2f},   ⟨d_edge⟩ = {d_hi:.4f}",
    fontsize=11, fontfamily="serif"
)

fig.suptitle(
    f"soft RGG: same α = {ALPHA}, varying β",
    fontsize=14, fontfamily="serif", fontweight="bold", y=1.01
)

plt.tight_layout()
plt.savefig("question_03_c.png", dpi=300, bbox_inches="tight")
print("\nsaved question_03_c.png")