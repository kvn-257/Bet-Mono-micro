import networkx as nx
import numpy as np
import random



# ----------------------------
# Weight assignment
# ----------------------------
def assign_weights(G, mean=0.5, sigma=1.0):
    for u, v in G.edges():
        G[u][v]['weight'] = float(np.random.lognormal(mean, sigma))
    return G

# ----------------------------
# Enforce DAG 
# ----------------------------
def enforce_dag(G):
    DG = nx.DiGraph()
    nodes = list(G.nodes())

    ordering = list(range(len(nodes)))
    random.shuffle(ordering)
    order_map = {nodes[i]: ordering[i] for i in range(len(nodes))}

    for u, v in G.edges():
        if order_map[u] < order_map[v]:
            DG.add_edge(u, v)
        elif order_map[v] < order_map[u]:
            DG.add_edge(v, u)

    return DG

# ----------------------------
# Enforce density
# ----------------------------
def enforce_density(G, target_density=0.02):
    n = G.number_of_nodes()
    max_edges = int(target_density * n * (n - 1))

    edges = list(G.edges())
    if len(edges) > max_edges:
        keep_edges = random.sample(edges, max_edges)
        DG = nx.DiGraph()
        DG.add_edges_from(keep_edges)
        return DG

    return G

# ----------------------------
# Scale-free DAG
# ----------------------------
def generate_scale_free_dag(n=200, m=3):
    G = nx.barabasi_albert_graph(n, m)
    DG = nx.DiGraph()

    for u, v in G.edges():
        if random.random() < 0.5:
            DG.add_edge(u, v)
        else:
            DG.add_edge(v, u)

    return DG

# ----------------------------
# Layered DAG
# ----------------------------
def generate_layered_dag(n=200, layers=4, p_forward=0.1, p_skip=0.03):
    DG = nx.DiGraph()

    layer_sizes = [n // layers] * layers
    for i in range(n % layers):
        layer_sizes[i] += 1

    nodes = list(range(n))
    random.shuffle(nodes)

    layer_nodes = []
    idx = 0
    for size in layer_sizes:
        layer_nodes.append(nodes[idx:idx+size])
        idx += size

    for i in range(layers):
        for u in layer_nodes[i]:
            for j in range(i+1, layers):
                prob = p_forward if j == i+1 else p_skip
                for v in layer_nodes[j]:
                    if random.random() < prob:
                        DG.add_edge(u, v)

    return DG

# ----------------------------
# Weak modular graph
# ----------------------------
def generate_weakly_modular_graph(n=200, blocks=4, p_in=0.04, p_out=0.008):
    sizes = [n // blocks] * blocks
    for i in range(n % blocks):
        sizes[i] += 1

    probs = [[p_in if i == j else p_out for j in range(blocks)] for i in range(blocks)]
    G = nx.stochastic_block_model(sizes, probs, directed=True)

    DG = nx.DiGraph()
    DG.add_edges_from(G.edges())

    return DG

# ----------------------------
# Controlled bottleneck injection
# ----------------------------
def inject_bottleneck(G, strength=0.05):
    nodes = list(G.nodes())
    center = random.choice(nodes)

    for u in nodes:
        if u == center:
            continue

        if random.random() < strength:
            G.add_edge(u, center)

        if random.random() < strength:
            G.add_edge(center, u)

    return G

def inject_hub(G, hub_strength=0.1):
    nodes = list(G.nodes())
    hub = random.choice(nodes)

    for u in nodes:
        if u == hub:
            continue

        if random.random() < hub_strength:
            G.add_edge(u, hub)

    return G
# ----------------------------
# Dataset generator (FIXED)
# ----------------------------
def generate_dataset(num_graphs=10):
    graphs = []

    for _ in range(num_graphs):
        r = random.random()

        if r < 0.7:
            G = generate_scale_free_dag(
                n=random.randint(100, 300),
                m=random.randint(2, 4)
            )

        elif r < 0.9:
            G = generate_layered_dag(
                n=random.randint(100, 300),
                layers=random.randint(3, 6)
            )

        else:
            G = generate_weakly_modular_graph(
                n=random.randint(100, 300),
                blocks=random.randint(3, 5)
            )


        # 1. inject structure FIRST
        if random.random() < 0.6:
            G = inject_hub(G)

        if random.random() < 0.4:
            G = inject_bottleneck(G)

        # 2. control density
        G = enforce_density(G)

        # 3. NOW enforce DAG (critical)
        if random.random() < 0.85:
            G = enforce_dag(G)

        # 4. THEN assign weights
        G = assign_weights(G)

        graphs.append(G)

    return graphs

# ----------------------------
# Stats
# ----------------------------
def print_stats(G):
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Density:", round(nx.density(G), 4))
    print("Is DAG:", nx.is_directed_acyclic_graph(G))
    print("-" * 40)

# ----------------------------
# Run
# ----------------------------
graphs = generate_dataset(50)

for G in graphs:
    print_stats(G)