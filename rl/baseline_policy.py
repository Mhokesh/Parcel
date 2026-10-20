"""
rl/baseline_policy.py
Rollout Baseline for REINFORCE with Greedy Policy Comparator (Kool et al., 2019).
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Any


class RolloutBaseline:
    """
    Evaluates instances using a frozen copy of the model with greedy decoding
    to compute an unbiased variance-reduction baseline b(s).
    """
    def __init__(self, model: nn.Module):
        self.baseline_model = copy.deepcopy(model)
        self.baseline_model.eval()

    def update_baseline(self, model: nn.Module):
        """
        Updates the baseline model weights with the current model weights.
        """
        self.baseline_model.load_state_dict(model.state_dict())
        self.baseline_model.eval()

    def evaluate(
        self,
        node_features: torch.Tensor,
        demands: torch.Tensor,
        preference_weights: torch.Tensor,
        fuel_type: str = "diesel"
    ) -> torch.Tensor:
        """
        Runs greedy evaluation and returns the scalarized baseline cost.
        """
        from rl.environment import compute_batch_costs

        with torch.no_grad():
            out = self.baseline_model(
                node_features=node_features,
                demands=demands,
                preference_weights=preference_weights,
                decode_type="greedy"
            )
            costs = compute_batch_costs(
                coords=node_features[:, :, :2],
                demands=demands,
                actions=out["actions"],
                preference_weights=preference_weights,
                fuel_type=fuel_type
            )
            return costs["scalarized_cost"]


class ExponentialBaseline:
    """
    Simple running exponential moving average baseline for quick convergence.
    """
    def __init__(self, beta: float = 0.8):
        self.beta = beta
        self.value = None

    def evaluate(self, costs: torch.Tensor) -> torch.Tensor:
        batch_mean = costs.mean().item()
        if self.value is None:
            self.value = batch_mean
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * batch_mean
        return torch.full_like(costs, self.value)
