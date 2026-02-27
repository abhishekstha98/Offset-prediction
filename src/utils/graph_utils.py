import torch

def to_dense_adj(edge_index, num_nodes):
    """
    Convert sparse edge_index (2, E) to dense adjacency matrix (N, N).
    """
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1
    return adj
