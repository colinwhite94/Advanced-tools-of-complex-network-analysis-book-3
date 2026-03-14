# load polblogs dataset from netzschleuder
# extract largest connected component
# fit ndc sbm with 2 blocks
# fit dc sbm with 2 blocks
# visualize both graphs, colored by each partition

# in terminal: conda activate gtcnet5052
# then terminal: python question_01_a.py
import graph_tool.all as gt

#load polblogs dataset from netzschleuder
g = gt.collection.ns["polblogs"]

#extract the largest connected component
g = gt.extract_largest_component(g, directed=False)

#fit the ndc sbm
state_ndc = gt.minimize_blockmodel_dl(
    g,
    state_args={"deg_corr": False},
    multilevel_mcmc_args={"B_min": 2, "B_max": 2}
)

#fit the dc sbm
state_dc = gt.minimize_blockmodel_dl(
    g,
    state_args={"deg_corr": True},
    multilevel_mcmc_args={"B_min": 2, "B_max": 2}
)

#visualize plot 1 (ndc)
state_ndc.draw(output="question_01_a_ndc.png")

#visualize plot 2 (dc)
state_dc.draw(output="question_01_a_dc.png")

#  description lengths from fitted models
print("ndc description length:", state_ndc.entropy())
print("dc description length:", state_dc.entropy())

print("done")