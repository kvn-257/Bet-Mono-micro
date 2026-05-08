"""
Phase 2 — Edge Betweenness Centrality Estimation
=================================================
Trains a Graph Neural Network (GNN) to estimate the Edge Betweenness Centrality
for the structural dependency graphs produced in Phase 1.
"""

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data, Batch
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import kendalltau


class EdgeBetweennessGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_layers=3, learnable_features=False, max_nodes=10000):
        super(EdgeBetweennessGNN, self).__init__()
        torch.manual_seed(42)
        
        self.learnable_features = learnable_features
        if self.learnable_features:
            self.node_embedding = nn.Embedding(max_nodes, in_channels)
            # Randomly initialize embeddings
            nn.init.uniform_(self.node_embedding.weight, -1.0, 1.0)
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # Edge Regressor
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 1, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, edge_weight, n_id=None):
        # 1. Compute node embeddings using GraphSAGE
        # If learnable_features is enabled, ignore static x and use embeddings
        if self.learnable_features and n_id is not None:
            x = self.node_embedding(n_id)
            
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        # 2. Create edge embeddings by concatenating source and target node embeddings
        # along with the edge weight
        src, dst = edge_index
        edge_embeds = torch.cat([x[src], x[dst], edge_weight], dim=1)

        # 3. Predict edge betweenness centrality
        out = self.regressor(edge_embeds)
        return out.squeeze()


def margin_pair_loss(y_out, true_val, sample_size=10, margin=1):
    """
    Margin ranking loss to train the model to output relative rankings of edges 
    instead of absolute betweenness scores.
    """
    num_edges = len(true_val)
    if num_edges < 2:
        return torch.tensor(0.0, requires_grad=True).to(y_out.device)
        
    _, order_y_true = torch.sort(-true_val)
    
    sample_num = num_edges * sample_size
    ind_1 = torch.randint(0, num_edges, (sample_num,)).long()
    ind_2 = torch.randint(0, num_edges, (sample_num,)).long()
    
    y = torch.sign(-1 * (ind_1 - ind_2)).float()
    
    x1 = y_out[order_y_true[ind_1]]
    x2 = y_out[order_y_true[ind_2]]

    true_w1 = true_val[order_y_true[ind_1]]
    true_w2 = true_val[order_y_true[ind_2]]
    
    raw_weights = true_w1 + true_w2
    
    # Normalize so the average weight is 1.0
    # This prevents the vanishing gradient problem you correctly identified
    importance_weights = raw_weights / (raw_weights.mean() + 1e-8)
    
    base_loss = torch.clamp(-y * (x1 - x2) + margin, min=0)
    loss = (base_loss * importance_weights).mean()
    return loss


def kendall_tau_metric(predictions, true_values):
    if len(predictions) < 2:
        return 0.0
    tau, _ = kendalltau(predictions.detach().cpu().numpy(), true_values.detach().cpu().numpy())
    return tau if not np.isnan(tau) else 0.0


def precision_at_k(predictions, true_values, k=10):
    if len(predictions) < k:
        k = len(predictions)
    if k == 0:
        return 0.0
        
    _, top_k_pred_idx = torch.topk(predictions, k)
    _, top_k_true_idx = torch.topk(true_values, k)
    
    pred_set = set(top_k_pred_idx.tolist())
    true_set = set(top_k_true_idx.tolist())
    
    return len(pred_set.intersection(true_set)) / k


def extract_node_features(G: nx.DiGraph) -> torch.Tensor:
    """
    Extract basic graph-theoretic node centralities as input features for the GNN.
    """
    # 1. Degree Centrality
    x_deg = np.array(list(nx.degree_centrality(G).values())).reshape(-1, 1)
    
    # 2. In/Out Degree Centrality
    x_in_deg = np.array(list(nx.in_degree_centrality(G).values())).reshape(-1, 1)
    x_out_deg = np.array(list(nx.out_degree_centrality(G).values())).reshape(-1, 1)

    # 3. Closeness Centrality
    x_cc = np.array(list(nx.closeness_centrality(G).values())).reshape(-1, 1)

    # 4. PageRank
    try:
        x_pr = np.array(list(nx.pagerank(G, weight='weight').values())).reshape(-1, 1)
    except:
        x_pr = np.array(list(nx.pagerank(G).values())).reshape(-1, 1)

    # Combine features (5 features total)
    x = np.hstack((x_deg, x_in_deg, x_out_deg, x_cc, x_pr))

    # Scale to [0, 1]
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    return torch.tensor(x_scaled, dtype=torch.float32)


def graph_to_pyg_data(G: nx.DiGraph) -> Data:
    """
    Converts a NetworkX DiGraph to a PyTorch Geometric Data object.
    Computes the ground-truth Edge Betweenness Centrality for the labels.
    """
    # Map nodes to integer indices
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Compute exact edge betweenness centrality (normalized)
    # Using 'weight' if available, as these are weighted graphs
    edge_bwc = nx.edge_betweenness_centrality(G, weight='weight')

    edges = []
    labels = []
    weights = []
    
    for u, v in G.edges():
        edges.append([node_to_idx[u], node_to_idx[v]])
        labels.append(edge_bwc.get((u, v), 0.0))
        weights.append([G[u][v].get('weight', 1.0)])

    # normalize edge weights to [0, 1] to prevent overpowering node embeddings
    if len(weights) > 0:
        scaler = MinMaxScaler()
        weights_scaled = scaler.fit_transform(weights)
    else:
        weights_scaled = weights

    # edge_index shape [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # targets shape [num_edges]
    y = torch.tensor(labels, dtype=torch.float32)

    # edge weights shape [num_edges, 1]
    edge_weight = torch.tensor(weights_scaled, dtype=torch.float32)

    # input features
    x = extract_node_features(G)
    
    # assign local node ids (will be offset during batching or used directly)
    n_id = torch.arange(len(nodes), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight, n_id=n_id)
    
    # Store original edge tuples for reference mapping later in Phase 3/4
    data.edge_tuples = list(G.edges())
    return data


def prepare_dataset(graphs: list[nx.DiGraph], learnable_features=False):
    print(f"Preparing dataset from {len(graphs)} graphs...")
    dataset = []
    global_node_count = 0
    for G in graphs:
        data = graph_to_pyg_data(G)
        if learnable_features:
            data.n_id = data.n_id + global_node_count
            global_node_count += data.x.size(0)
        dataset.append(data)
    return dataset, global_node_count


def train_edge_bwc_model(dataset, global_node_count, epochs=100, lr=0.01, learnable_features=False, hidden_channels=64, num_layers=3):
    """
    Trains the EdgeBetweennessGNN on a list of graphs.
    """
    # Split into train/test
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # All graphs have the same feature dimension (5)
    in_channels = train_data[0].x.size(1)
    
    model = EdgeBetweennessGNN(
        in_channels=in_channels, 
        hidden_channels=hidden_channels, 
        num_layers=num_layers, 
        learnable_features=learnable_features,
        max_nodes=global_node_count if learnable_features else 10000
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training for {epochs} epochs (Learnable Features: {learnable_features})...")
    
    best_val_tau = -1.0
    best_metrics = {}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data in train_data:
            optimizer.zero_grad()
            n_id = data.n_id if learnable_features else None
            out = model(data.x, data.edge_index, data.edge_weight, n_id)
            # Use Pairwise Margin Loss instead of MSE
            loss = margin_pair_loss(out, data.y, sample_size=5, margin=0.1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_data)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            model.eval()
            total_val_loss = 0
            total_tau = 0
            total_p10 = 0
            total_p20 = 0
            total_p30 = 0
            total_p50 = 0
            
            with torch.no_grad():
                for data in test_data:
                    n_id = data.n_id if learnable_features else None
                    out = model(data.x, data.edge_index, data.edge_weight, n_id)
                    val_loss = margin_pair_loss(out, data.y, sample_size=5, margin=0.1)
                    total_val_loss += val_loss.item()
                    
                    total_tau += kendall_tau_metric(out, data.y)
                    total_p10 += precision_at_k(out, data.y, k=10)
                    total_p20 += precision_at_k(out, data.y, k=20)
                    total_p30 += precision_at_k(out, data.y, k=30)
                    total_p50 += precision_at_k(out, data.y, k=50)
                    
            avg_val_loss = total_val_loss / len(test_data)
            avg_tau = total_tau / len(test_data)
            avg_p10 = total_p10 / len(test_data)
            avg_p20 = total_p20 / len(test_data)
            avg_p30 = total_p30 / len(test_data)
            avg_p50 = total_p50 / len(test_data)
            
            if avg_tau > best_val_tau:
                best_val_tau = avg_tau
                best_metrics = {
                    "val_tau": avg_tau,
                    "p@10": avg_p10,
                    "p@20": avg_p20,
                    "p@30": avg_p30,
                    "p@50": avg_p50
                }
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:03d} | Val Loss: {avg_val_loss:.5f} | Val Tau: {avg_tau:.4f} | P@10: {avg_p10:.4f} | P@20: {avg_p20:.4f} | P@30: {avg_p30:.4f} | P@50: {avg_p50:.4f}")

    print("Training complete.")
    return model, best_metrics


if __name__ == "__main__":
    import argparse
    import sys
    import json
    import itertools
    import pickle
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Phase 2: Edge Betweenness Centrality Estimation")
    parser.add_argument("--learnable-features", action="store_true", help="Use randomly initialized learnable parameters as node features instead of static centralities")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num-graphs", type=int, default=500, help="Number of synthetic graphs to generate for training/testing")
    args = parser.parse_args()

    # Add project root to sys.path
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    # -----------------------------------------------------------------------
    # Load pre-built real-world graphs (built by build_real_graphs.py)
    # -----------------------------------------------------------------------
    real_world_dir = project_root / "real_world_graphs"
    real_world_data: list = []   # list of (repo_name, Data)

    if real_world_dir.exists():
        pickle_files = sorted(real_world_dir.glob("*.gpickle"))
        if pickle_files:
            print(f"\nLoading {len(pickle_files)} real-world graphs for evaluation...")
            for pf in pickle_files:
                try:
                    with open(pf, "rb") as f:
                        G_real = pickle.load(f)
                    if G_real.number_of_edges() > 0:
                        rw_data = graph_to_pyg_data(G_real)
                        real_world_data.append((pf.stem, rw_data))
                        print(f"  Loaded {pf.stem}: {G_real.number_of_nodes()} nodes, "
                              f"{G_real.number_of_edges()} edges")
                    else:
                        print(f"  [SKIP] {pf.stem}: no edges after filtering")
                except Exception as e:
                    print(f"  [WARN] Could not load {pf.name}: {e}")
        else:
            print("\n[WARN] No .gpickle files found in real_world_graphs/. "
                  "Run build_real_graphs.py first for real-world evaluation.")
    else:
        print("\n[WARN] real_world_graphs/ directory not found. "
              "Run build_real_graphs.py first for real-world evaluation.")

    def evaluate_on_real_graphs(model: nn.Module) -> dict:
        """Run inference on every pre-loaded real-world graph and return per-repo metrics."""
        model.eval()
        results = {}
        with torch.no_grad():
            for repo_name, rw_data in real_world_data:
                out = model(rw_data.x, rw_data.edge_index, rw_data.edge_weight, None)
                tau = kendall_tau_metric(out, rw_data.y)
                p10 = precision_at_k(out, rw_data.y, k=10)
                p20 = precision_at_k(out, rw_data.y, k=20)
                p30 = precision_at_k(out, rw_data.y, k=30)
                p50 = precision_at_k(out, rw_data.y, k=50)
                results[repo_name] = {
                    "tau":  round(tau, 4),
                    "p@10": round(p10, 4),
                    "p@20": round(p20, 4),
                    "p@30": round(p30, 4),
                    "p@50": round(p50, 4),
                }
                print(f"    [{repo_name}] Tau: {tau:.4f} | P@10: {p10:.4f} | "
                      f"P@20: {p20:.4f} | P@30: {p30:.4f} | P@50: {p50:.4f}")

        # Macro-average across all repos
        if results:
            repo_vals = list(results.values())
            results["avg"] = {
                "tau":  round(sum(r["tau"]  for r in repo_vals) / len(repo_vals), 4),
                "p@10": round(sum(r["p@10"] for r in repo_vals) / len(repo_vals), 4),
                "p@20": round(sum(r["p@20"] for r in repo_vals) / len(repo_vals), 4),
                "p@30": round(sum(r["p@30"] for r in repo_vals) / len(repo_vals), 4),
                "p@50": round(sum(r["p@50"] for r in repo_vals) / len(repo_vals), 4),
            }
            avg = results["avg"]
            print(f"    [AVG] Tau: {avg['tau']:.4f} | P@10: {avg['p@10']:.4f} | "
                  f"P@20: {avg['p@20']:.4f} | P@30: {avg['p@30']:.4f} | P@50: {avg['p@50']:.4f}")
        return results

    try:
        from synthetic_dataset.syn_graphs import generate_dataset
        print(f"\nGenerating {args.num_graphs} synthetic graphs...")
        graphs = generate_dataset(num_graphs=args.num_graphs)

        dataset, global_node_count = prepare_dataset(graphs, args.learnable_features)

        # Hyperparameter Tuning
        hidden_channels_opts = [16, 32, 64]
        num_layers_opts      = [2, 3, 4]
        lr_opts              = [0.001, 0.005, 0.01]

        best_tau         = -1.0
        best_model_state = None
        best_hparams     = {}

        for hc, nl, lr in itertools.product(hidden_channels_opts, num_layers_opts, lr_opts):
            print(f"\n--- Tuning Config: hidden_channels={hc}, num_layers={nl}, lr={lr} ---")
            model, metrics = train_edge_bwc_model(
                dataset,
                global_node_count,
                epochs=args.epochs,
                lr=lr,
                learnable_features=args.learnable_features,
                hidden_channels=hc,
                num_layers=nl
            )

            # Evaluate on real-world graphs after each run
            rw_metrics = {}
            if real_world_data:
                print(f"  >> Real-world evaluation (hc={hc}, nl={nl}, lr={lr}):")
                rw_metrics = evaluate_on_real_graphs(model)

            if metrics.get("val_tau", -1) > best_tau:
                best_tau         = metrics.get("val_tau", -1)
                best_model_state = model.state_dict()
                best_hparams = {
                    "hidden_channels":      hc,
                    "num_layers":           nl,
                    "lr":                   lr,
                    "synthetic_val_tau":    metrics.get("val_tau", 0),
                    "synthetic_p@10":       metrics.get("p@10", 0),
                    "synthetic_p@20":       metrics.get("p@20", 0),
                    "synthetic_p@30":       metrics.get("p@30", 0),
                    "synthetic_p@50":       metrics.get("p@50", 0),
                    "real_world_avg":       rw_metrics.get("avg", {}),
                    "real_world":           {k: v for k, v in rw_metrics.items() if k != "avg"},
                }

        print(f"\nBest Hyperparameters found: {best_hparams}")

        # Save best model
        torch.save(best_model_state, "best_edge_bwc_gnn.pth")
        print("Best model saved to best_edge_bwc_gnn.pth")

        # Save best hyperparameters
        with open("best_hparams.json", "w") as f:
            json.dump(best_hparams, f, indent=4)
        print("Best hyperparameters saved to best_hparams.json")

    except ImportError as e:
        print(f"Could not import module: {e}")
        print("Make sure you are running from the project root or PYTHONPATH is set correctly.")


