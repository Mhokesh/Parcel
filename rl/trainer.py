"""
rl/trainer.py
Multi-Objective Deep Reinforcement Learning Trainer for MO-CVRP and MO-CVRPTW.

Algorithm:
- REINFORCE with Rollout / Exponential Baseline
- Multi-Objective Preference Sampling (w ~ Simplex)
- Dynamic Feasibility Masking
"""

import os
import time
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple

from models.attention_model import MultiObjectiveAttentionModel
from rl.environment import compute_batch_costs
from rl.baseline_policy import RolloutBaseline, ExponentialBaseline
from data.cvrp_generator import sample_cvrp_batch


def sample_preference_weights(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Samples random preference vectors (w_dist, w_fuel, w_tw) on the 3D simplex (sum = 1).
    """
    # Sample from Dirichlet distribution with alpha = (1, 1, 1) or uniform simplex
    # Training data has no time windows; do not claim a learned third objective.
    raw = np.random.exponential(scale=1.0, size=(batch_size, 2))
    weights = raw / np.sum(raw, axis=-1, keepdims=True)
    return torch.tensor(np.column_stack([weights, np.zeros(batch_size)]), dtype=torch.float32, device=device)


class RLTrainer:
    """
    Multi-Objective Reinforcement Learning Trainer.
    """
    def __init__(
        self,
        model: Optional[MultiObjectiveAttentionModel] = None,
        num_customers: int = 20,
        vehicle_capacity: int = 30,
        learning_rate: float = 1e-4,
        grad_clip_norm: float = 1.0,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_customers = num_customers
        self.vehicle_capacity = vehicle_capacity

        if model is None:
            self.model = MultiObjectiveAttentionModel(
                node_dim=4, # (x, y, demand, time_window_proxy)
                embedding_dim=128,
                num_layers=3,
                num_heads=8
            ).to(self.device)
        else:
            self.model = model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.grad_clip_norm = grad_clip_norm
        self.baseline = ExponentialBaseline(beta=0.8)

    def prepare_batch(self, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a synthetic CVRP batch formatted as model input tensors.
        """
        raw = sample_cvrp_batch(
            batch_size=batch_size,
            num_customers=self.num_customers,
            capacity=self.vehicle_capacity
        )

        depots = torch.tensor(raw["depots"], dtype=torch.float32, device=self.device) # (B, 1, 2)
        customers = torch.tensor(raw["customers"], dtype=torch.float32, device=self.device) # (B, N, 2)
        coords = torch.cat([depots, customers], dim=1) # (B, N + 1, 2)

        # Demands: depot has 0 demand
        demands_raw = torch.tensor(raw["norm_demands"], dtype=torch.float32, device=self.device) # (B, N)
        depot_demands = torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device)
        all_demands = torch.cat([depot_demands, demands_raw], dim=-1) # (B, N + 1)

        # Build node features: [x, y, demand, dummy_time_window]
        tw_feature = torch.zeros((batch_size, self.num_customers + 1, 1), dtype=torch.float32, device=self.device)
        node_features = torch.cat([coords, all_demands.unsqueeze(-1), tw_feature], dim=-1) # (B, N + 1, 4)

        return node_features, all_demands

    def train_epoch(
        self,
        num_batches: int = 50,
        batch_size: int = 64,
        fuel_type: str = "diesel"
    ) -> Dict[str, float]:
        """
        Runs one training epoch with preference sampling.
        """
        self.model.train()
        epoch_losses = []
        epoch_distances = []
        epoch_fuels = []
        epoch_costs = []

        for b_idx in range(num_batches):
            node_features, demands = self.prepare_batch(batch_size=batch_size)
            prefs = sample_preference_weights(batch_size, self.device)

            # Sample actions from policy distribution
            out = self.model(
                node_features=node_features,
                demands=demands,
                preference_weights=prefs,
                decode_type="sample"
            )

            # Compute costs for the sampled actions
            costs = compute_batch_costs(
                coords=node_features[:, :, :2],
                demands=demands,
                actions=out["actions"],
                preference_weights=prefs,
                fuel_type=fuel_type,
                vehicle_capacity=self.vehicle_capacity
            )

            scalarized_cost = costs["scalarized_cost"]

            # Compute baseline
            baseline_val = self.baseline.evaluate(scalarized_cost)

            # REINFORCE Advantage: (J - b)
            advantage = (scalarized_cost - baseline_val).detach()

            # Policy Gradient Loss: Advantage * sum(log_probs)
            log_probs = out["total_log_probs"]
            loss = torch.mean(advantage * log_probs)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            epoch_losses.append(loss.item())
            epoch_distances.append(costs["total_distance"].mean().item())
            epoch_fuels.append(costs["total_fuel"].mean().item())
            epoch_costs.append(scalarized_cost.mean().item())

        return {
            "loss": float(np.mean(epoch_losses)),
            "avg_distance": float(np.mean(epoch_distances)),
            "avg_fuel": float(np.mean(epoch_fuels)),
            "avg_scalarized_cost": float(np.mean(epoch_costs))
        }

    def evaluate_greedy(
        self,
        batch_size: int = 64,
        preference_weights: Optional[Tuple[float, float, float]] = None,
        fuel_type: str = "diesel"
    ) -> Dict[str, float]:
        """
        Evaluates the current model greedily on a test batch.
        """
        self.model.eval()
        with torch.no_grad():
            node_features, demands = self.prepare_batch(batch_size=batch_size)
            if preference_weights is not None:
                pw = torch.tensor([preference_weights] * batch_size, dtype=torch.float32, device=self.device)
            else:
                pw = sample_preference_weights(batch_size, self.device)

            out = self.model(
                node_features=node_features,
                demands=demands,
                preference_weights=pw,
                decode_type="greedy"
            )

            costs = compute_batch_costs(
                coords=node_features[:, :, :2],
                demands=demands,
                actions=out["actions"],
                preference_weights=pw,
                fuel_type=fuel_type,
                vehicle_capacity=self.vehicle_capacity
            )

            return {
                "eval_distance": costs["total_distance"].mean().item(),
                "eval_fuel": costs["total_fuel"].mean().item(),
                "eval_co2": costs["total_co2"].mean().item(),
                "eval_scalarized_cost": costs["scalarized_cost"].mean().item()
            }

    def save_checkpoint(self, path: str = "saved_checkpoints/mo_cvrp_model.pt"):
        """
        Saves model weights to checkpoint file.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_customers": self.num_customers,
            "vehicle_capacity": self.vehicle_capacity
        }, path)
        print(f"Checkpoint successfully saved to {path}")

    def load_checkpoint(self, path: str = "saved_checkpoints/mo_cvrp_model.pt"):
        """
        Loads model weights from checkpoint file.
        """
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Checkpoint successfully loaded from {path}")
        else:
            print(f"Warning: Checkpoint {path} not found.")
