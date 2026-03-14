# load polblogs dataset from netzschleuder
# extract largest connected component
# initialize blockstate with b=2 for both models
# run mcmc sweep for 1000 iterations, collecting description length at each step
# visualize both curves in a single plot

# in terminal: conda activate gtcnet5052
# then terminal: python question_01_b.py

import graph_tool.all as gt
import matplotlib.pyplot as plt

# load polblogs dataset from netzschleuder
g = gt.collection.ns["polblogs"]

# extract the largest connected component
g = gt.extract_largest_component(g, directed=False)

# initialize blockstate with b=2 for ndc
state_ndc = gt.BlockState(g, B=2, deg_corr=False)

# initialize blockstate with b=2 for dc
state_dc = gt.BlockState(g, B=2, deg_corr=True)

# run mcmc sweep for 1000 iterations, collecting description length at each step
ndc_lengths = []
dc_lengths = []

for i in range(1000):
    state_ndc.mcmc_sweep(niter=1)
    ndc_lengths.append(state_ndc.entropy())

    state_dc.mcmc_sweep(niter=1)
    dc_lengths.append(state_dc.entropy())

# visualize both curves in a single plot
plt.figure(figsize=(8, 5), dpi=300)
plt.plot(ndc_lengths, color="black", linewidth=1.2, linestyle="-", label="non-degree-corrected")
plt.plot(dc_lengths, color="gray", linewidth=1.2, linestyle="--", label="degree-corrected")
plt.xlabel("iteration", fontsize=12, fontfamily="serif")
plt.ylabel("description length", fontsize=12, fontfamily="serif")
plt.title("mcmc description length over iterations", fontsize=13, fontfamily="serif")
plt.legend(fontsize=11, frameon=True, edgecolor="black")
plt.tick_params(axis="both", labelsize=10)
plt.tight_layout()
plt.savefig("question_01_b.png", dpi=300, bbox_inches="tight")
print("done")

