# road network analysis using osmnx
# download and visualize boston city hall road network
# download and visualize tucson road network
# dissimilarity function D(G, H) using:
#  intersection density (nodes / area)
#  mean degree
#  mean edge length
# compute D(boston, tucson)
# compute pairwise distance matrix for list of cities

# graph representation: undirected simple graph (used consistently throughout)
# caching: graphs saved as graphml and reloaded if already downloaded

# run with: python question_04.py

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import osmnx as ox

# settings 
RADIUS = 3219 # meters 
NET_TYPE = "drive"
CACHE_DIR = "graphml_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# helper: download or load from cache
def load_graph(query, tag, point=None):
    path = os.path.join(CACHE_DIR, f"{tag}.graphml")
    # if graphml file exists, load it; otherwise download and save
    if os.path.exists(path):
        G = ox.load_graphml(path)
    else:
        if point is None:
            # geocode place name to lat/lon, then use graph_from_point
            point = ox.geocode(query)
        G = ox.graph_from_point(point, dist=RADIUS, network_type=NET_TYPE)
        ox.save_graphml(G, path)

    # project to utm for metric calculations
    G = ox.project_graph(G)

    # convert to undirected simple graph (osmnx 2.0 compatible)
    G = nx.Graph(G.to_undirected())

    return G

# helper: compute area of graph footprint (km squared) 
def graph_area_km2(G):
    #approximate area using bounding box of node coordinates
    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]
    area_m2 = (max(xs) - min(xs)) * (max(ys) - min(ys))
    return area_m2 / 1e6

# helper: extract features
def feature_vector(G):
    #compute three size-comparable features:
    #f1: intersection density = N / area_km2
    #f2: mean degree
    #f3: mean edge length (meters)

    N    = G.number_of_nodes()
    area = graph_area_km2(G)

    # intersection density
    f1 = N / area if area > 0 else 0

    # mean degree
    degrees = [d for _, d in G.degree()]
    f2 = np.mean(degrees)

    # mean edge length
    lengths = [d.get("length", 0) for _, _, d in G.edges(data=True)]
    f3 = np.mean(lengths) if lengths else 0

    return np.array([f1, f2, f3])

# dissimilarity function 
def D(fv_G, fv_H, mu, sigma):
    # normalize features using provided means and stdevs
    z_g = (fv_G - mu) / sigma
    z_h = (fv_H - mu) / sigma
    return float(np.linalg.norm(z_g - z_h))

# helper: visualize a single graph
def visualize_graph(G, title, filename):
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    xs = [d["x"] for _, d in G.nodes(data=True)]
    ys = [d["y"] for _, d in G.nodes(data=True)]

    # draw edges
    for u, v, data in G.edges(data=True):
        xu, yu = G.nodes[u]["x"], G.nodes[u]["y"]
        xv, yv = G.nodes[v]["x"], G.nodes[v]["y"]
        ax.plot([xu, xv], [yu, yv],
                color="white", linewidth=0.4, alpha=0.7, zorder=1)

    # draw nodes
    ax.scatter(xs, ys, s=1.5, color="white", alpha=0.6, zorder=2)

    ax.set_title(title, color="white", fontsize=12,
                 fontfamily="serif", pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight",
                facecolor="black")
    plt.close()
    print(f"  saved {filename}")

#parts (a) and (b): boston and tucson
print("\nboston city hall")
G_boston = load_graph(
    query="boston city hall, boston, ma, usa",
    tag="boston",
    point=(42.3601, -71.0589)
)
N_b, M_b = G_boston.number_of_nodes(), G_boston.number_of_edges()
print(f"  N = {N_b} nodes,  M = {M_b} edges")
visualize_graph(
    G_boston,
    title="boston, ma — drive network\ncenter: boston city hall, radius: 3219 m",
    filename="question_04_boston.png"
)

print("\ntucson")
G_tucson = load_graph(
    query="tucson, arizona, usa",
    tag="tucson",
    point=(32.335660407, -110.97908793)
)
N_t, M_t = G_tucson.number_of_nodes(), G_tucson.number_of_edges()
print(f"  N = {N_t} nodes,  M = {M_t} edges")
visualize_graph(
    G_tucson,
    title="tucson, az — drive network\ncenter: (32.3357, −110.9791), radius: 3219 m",
    filename="question_04_tucson.png"
)

# part (f): D(boston, tucson)
print("\nD(boston, tucson)")
fv_boston = feature_vector(G_boston)
fv_tucson = feature_vector(G_tucson)

print(f" boston features: density={fv_boston[0]:.2f}, mean_k={fv_boston[1]:.3f}, mean_len={fv_boston[2]:.2f}m")
print(f" tucson features: density={fv_tucson[0]:.2f}, mean_k={fv_tucson[1]:.3f}, mean_len={fv_tucson[2]:.2f}m")

# for part f, normalize using boston and tucson as reference
fv_both = np.vstack([fv_boston, fv_tucson])
mu_bt    = fv_both.mean(axis=0)
sigma_bt = fv_both.std(axis=0)
sigma_bt[sigma_bt == 0] = 1
d_bt = D(fv_boston, fv_tucson, mu_bt, sigma_bt)
print(f"  D(boston, tucson) = {d_bt:.4f}")

# g: city distance matrix
print("\npart (g): 25-city distance matrix ")

CITIES = [
    ("Atlanta, Georgia, USA",                   "atlanta"),
    ("Boston, MA, USA",                          "boston_g"),
    ("Buffalo, NY, USA",                         "buffalo"),
    ("Charlotte, NC, USA",                       "charlotte"),
    ("Chicago, IL, USA",                         "chicago"),
    ("Cleveland, OH, USA",                       "cleveland"),
    ("Dallas, TX, USA",                          "dallas"),
    ("Houston, TX, USA",                         "houston"),
    ("Denver, CO, USA",                          "denver"),
    ("Detroit, MI, USA",                         "detroit"),
    ("Las Vegas, NV, USA",                       "las_vegas"),
    ("Los Angeles, CA, USA",                     "los_angeles"),
    ("Manhattan, NYC, NY, USA",                  "manhattan"),
    ("Miami, FL, USA",                           "miami"),
    ("Minneapolis, MN, USA",                     "minneapolis"),
    ("Orlando, FL, USA",                         "orlando"),
    ("Philadelphia, PA, USA",                    "philadelphia"),
    ("Phoenix, AZ, USA",                         "phoenix"),
    ("Portland, OR, USA",                        "portland"),
    ("Sacramento, CA, USA",                      "sacramento"),
    ("San Francisco, CA, USA",                   "san_francisco"),
    ("Seattle, WA, USA",                         "seattle"),
    ("St. Louis, MO, USA",                       "st_louis"),
    ("Tampa, FL, USA",                           "tampa"),
    ("Washington, District of Columbia, USA",    "washington_dc"),
]

# download all cities and compute feature vectors
city_labels = [tag for _, tag in CITIES]
feature_vectors = {}

for query, tag in CITIES:
    print(f"  processing {tag}")
    try:
        G_city = load_graph(query=query, tag=tag)
        feature_vectors[tag] = feature_vector(G_city)
    except Exception as e:
        print(f" failed for {tag}: {e}")
        feature_vectors[tag] = None

# filter out any failed cities
valid = [(tag, fv) for tag, fv in feature_vectors.items() if fv is not None]
valid_labels = [tag for tag, _ in valid]
valid_fvs    = [fv  for _, fv  in valid]
K = len(valid_labels)

# convert to numpy array and compute global normalization
valid_fvs = np.array(valid_fvs)
global_mu = valid_fvs.mean(axis=0)
global_sigma = valid_fvs.std(axis=0)
global_sigma[global_sigma == 0] = 1
print(f' global means  : density={global_mu[0]:.2f}, mean_k={global_mu[1]:.3f}, mean_len={global_mu[2]:.2f}m')
print(f' global stdevs : density={global_sigma[0]:.2f}, mean_k={global_sigma[1]:.3f}, mean_len={global_sigma[2]:.2f}m')

# compute pairwise distance matrix
dist_matrix = np.zeros((K, K))
# since D is symmetric and D(G,G)=0, we only need to compute upper triangle
for i in range(K):
    for j in range(K):
        if i == j:
            dist_matrix[i, j] = 0.0
        elif j > i:
            d = D(valid_fvs[i], valid_fvs[j], global_mu, global_sigma)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

# visualize distance matrix
short_labels = [l.replace("_", "\n") for l in valid_labels]

fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
im = ax.imshow(dist_matrix, cmap="Greys", aspect="auto")
plt.colorbar(im, ax=ax, label="dissimilarity D(G,H)", shrink=0.8)

ax.set_xticks(range(K))
ax.set_yticks(range(K))
ax.set_xticklabels(short_labels, rotation=90, fontsize=7, fontfamily="serif")
ax.set_yticklabels(short_labels, fontsize=7, fontfamily="serif")
ax.set_title("pairwise road network dissimilarity — 25 u.s. cities",
             fontsize=13, fontfamily="serif", pad=14)

plt.tight_layout()
plt.savefig("question_04_distance_matrix.png", dpi=300, bbox_inches="tight")
print("saved question_04_distance_matrix.png")

print("\ndone.")