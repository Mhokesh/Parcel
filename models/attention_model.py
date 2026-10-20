"""
models/attention_model.py
Multi-Head Attention Encoder-Decoder Policy for MO-CVRP and MO-CVRPTW.

Architecture:
- Node Encoder: Linear Projection + Multi-Head Self-Attention layers with residual connections & normalization.
- Preference Conditioned Context: Incorporates user preference weights (w_dist, w_fuel, w_tw),
  vehicle position, remaining capacity, and time elapsed.
- Dynamic Pointer Decoder: Single-head/multi-head attention pointing mechanism with dynamic feasibility masking.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List


class MultiHeadAttentionLayer(nn.Module):
    """
    Standard Transformer Multi-Head Self-Attention Layer with Skip Connection and Normalization.
    """
    def __init__(self, embedding_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_nodes, embedding_dim)
        batch_size, num_nodes, _ = x.size()

        # Multi-Head Attention
        q = self.q_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)

        # Transpose back and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.embedding_dim)
        out1 = self.norm1(x + self.out_proj(attn_out))

        # Feed Forward Network
        out2 = self.norm2(out1 + self.ffn(out1))
        return out2


class GraphAttentionEncoder(nn.Module):
    """
    Encodes graph nodes (depot + customers) into context-aware embeddings.
    """
    def __init__(self, node_dim: int = 4, embedding_dim: int = 128, num_layers: int = 3, num_heads: int = 8):
        super().__init__()
        # node_dim: (x, y, demand, time_window_early, time_window_late) -> e.g. 4 or 5
        self.node_embed = nn.Linear(node_dim, embedding_dim)
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(embedding_dim=embedding_dim, num_heads=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        # node_features: (batch_size, num_nodes, node_dim)
        h = self.node_embed(node_features)
        for layer in self.layers:
            h = layer(h)
        return h # (batch_size, num_nodes, embedding_dim)


class PointerDecoder(nn.Module):
    """
    Autoregressive Pointer Decoder with dynamic state context and preference conditioning.
    """
    def __init__(self, embedding_dim: int = 128, context_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        # Context projection: embeds (current_node_embed + remaining_cap + preference_weights)
        # Context features: current_node_embed (128) + graph_mean (128) + remaining_capacity (1) + preference_weights (3)
        self.context_proj = nn.Linear(embedding_dim * 2 + 1 + 3, embedding_dim)

        # Pointer attention projections
        self.w_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.w_k = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        current_node_idx: torch.Tensor,
        remaining_capacity: torch.Tensor,
        preference_weights: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        node_embeddings: (batch_size, num_nodes, embedding_dim)
        current_node_idx: (batch_size, 1) long tensor
        remaining_capacity: (batch_size, 1) float tensor in [0, 1]
        preference_weights: (batch_size, 3) float tensor
        mask: (batch_size, num_nodes) bool tensor (True = masked out / invalid action)
        """
        batch_size, num_nodes, emb_dim = node_embeddings.size()

        # Gather current node embedding
        idx_expanded = current_node_idx.unsqueeze(-1).expand(batch_size, 1, emb_dim)
        curr_emb = torch.gather(node_embeddings, 1, idx_expanded).squeeze(1) # (batch_size, emb_dim)

        # Graph-level pooled representation
        graph_mean = node_embeddings.mean(dim=1) # (batch_size, emb_dim)

        # Form context query
        context_input = torch.cat([curr_emb, graph_mean, remaining_capacity, preference_weights], dim=-1)
        query = self.context_proj(context_input).unsqueeze(1) # (batch_size, 1, emb_dim)

        # Scaled dot-product pointer attention
        q = self.w_q(query) # (batch_size, 1, emb_dim)
        k = self.w_k(node_embeddings) # (batch_size, num_nodes, emb_dim)

        scores = torch.matmul(q, k.transpose(-2, -1)).squeeze(1) / math.sqrt(emb_dim) # (batch_size, num_nodes)

        # Clip logits for numerical stability (tanh clipping)
        scores = 10.0 * torch.tanh(scores)

        # Apply feasibility mask: set masked out positions to large negative value
        scores = scores.masked_fill(mask, -1e9)

        # Return log probabilities and softmax probabilities
        log_probs = F.log_softmax(scores, dim=-1)
        probs = F.softmax(scores, dim=-1)

        return log_probs, probs


class MultiObjectiveAttentionModel(nn.Module):
    """
    Complete Neural Combinatorial Optimization policy network for MO-CVRP and MO-CVRPTW.
    """
    def __init__(
        self,
        node_dim: int = 4,
        embedding_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8
    ):
        super().__init__()
        self.encoder = GraphAttentionEncoder(
            node_dim=node_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
        self.decoder = PointerDecoder(
            embedding_dim=embedding_dim,
            context_dim=embedding_dim,
            num_heads=num_heads
        )
        self.embedding_dim = embedding_dim

    def forward(
        self,
        node_features: torch.Tensor,
        demands: torch.Tensor,
        preference_weights: Optional[torch.Tensor] = None,
        decode_type: str = "greedy"
    ) -> Dict[str, Any]:
        """
        Performs full autoregressive tour generation for a batch of instances.

        node_features: (batch_size, num_nodes, node_dim) -> index 0 is depot
        demands: (batch_size, num_nodes) -> normalized demand [0, 1]
        preference_weights: (batch_size, 3) -> (w_dist, w_fuel, w_tw)
        decode_type: 'greedy' or 'sample'
        """
        batch_size, num_nodes, _ = node_features.size()
        device = node_features.device
        if not torch.isfinite(demands).all() or (demands < 0).any() or (demands[:, 1:] > 1).any():
            raise ValueError("Normalized customer demand must be finite and between 0 and 1")

        if preference_weights is None:
            # Default equal weights if not specified
            preference_weights = torch.full((batch_size, 3), 1.0 / 3.0, device=device)

        # 1. Encode all nodes
        node_embeddings = self.encoder(node_features)

        # 2. Initialize decoding state
        current_node = torch.zeros((batch_size, 1), dtype=torch.long, device=device) # Start at depot 0
        remaining_capacity = torch.ones((batch_size, 1), dtype=torch.float32, device=device) # Full capacity = 1.0

        visited_mask = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=device)
        visited_mask[:, 0] = False # Depot can be revisited

        # Track completed customers
        customers_served = torch.zeros((batch_size, num_nodes), dtype=torch.bool, device=device)
        customers_served[:, 0] = True # Depot doesn't count as unserved customer

        actions_list = []
        log_probs_list = []

        max_steps = num_nodes * 4 # Upper bound on route steps

        for step in range(max_steps):
            # Check if all customers are served
            all_served = customers_served[:, 1:].all(dim=-1) # (batch_size,)
            if all_served.all() and (current_node == 0).all():
                break

            # Build dynamic mask:
            # 1. Mask customers already served
            mask = customers_served.clone()

            # 2. Mask customers exceeding current remaining capacity
            exceeds_cap = demands > remaining_capacity # (batch_size, num_nodes)
            mask = mask | exceeds_cap

            # 3. Handle depot logic:
            # - If at depot and not all served: mask depot (must visit at least one customer)
            # - If all customers served: ONLY depot is allowed
            at_depot = (current_node.squeeze(-1) == 0)
            mask[:, 0] = False # unmask depot by default

            for b in range(batch_size):
                if all_served[b]:
                    mask[b, :] = True
                    mask[b, 0] = False # Only depot allowed
                elif at_depot[b]:
                    # At depot: can't immediately stay at depot
                    mask[b, 0] = True

            # If all customer actions are masked out for some batch item (e.g. no capacity fits), must go to depot
            no_feasible_cust = mask[:, 1:].all(dim=-1)
            for b in range(batch_size):
                if no_feasible_cust[b] and not all_served[b]:
                    mask[b, :] = True
                    mask[b, 0] = False # Force return to depot

            # 4. Decoder step
            log_probs, probs = self.decoder(
                node_embeddings=node_embeddings,
                current_node_idx=current_node,
                remaining_capacity=remaining_capacity,
                preference_weights=preference_weights,
                mask=mask
            )

            # 5. Select action
            if decode_type == "greedy":
                action = torch.argmax(probs, dim=-1, keepdim=True)
            elif decode_type == "sample":
                # Sample from categorical distribution
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().unsqueeze(-1)
            else:
                raise ValueError(f"Unknown decode_type: {decode_type}")

            selected_log_prob = log_probs.gather(1, action)

            actions_list.append(action)
            log_probs_list.append(selected_log_prob)

            # 6. Update state
            # Next node index
            next_node = action
            current_node = next_node

            for b in range(batch_size):
                chosen_idx = next_node[b, 0].item()
                if chosen_idx == 0:
                    # Returned to depot -> refill vehicle capacity
                    remaining_capacity[b, 0] = 1.0
                else:
                    # Served customer -> decrement capacity and mark served
                    remaining_capacity[b, 0] = max(0.0, remaining_capacity[b, 0].item() - demands[b, chosen_idx].item())
                    customers_served[b, chosen_idx] = True

        # Stack actions: (batch_size, seq_len)
        actions = torch.cat(actions_list, dim=-1) if actions_list else torch.zeros((batch_size, 0), dtype=torch.long)
        total_log_probs = torch.cat(log_probs_list, dim=-1).sum(dim=-1) if log_probs_list else torch.zeros(batch_size)

        return {
            "actions": actions,
            "total_log_probs": total_log_probs,
            "seq_len": actions.size(1) if actions.dim() > 1 else 0
        }
