"""
models/policy_solver.py
Neural Policy Solver Interface for MO-CVRP and MO-CVRPTW.

Allows running inference with a trained Attention Model policy
with customizable preference weights (w_dist, w_fuel, w_tw).
"""

import time
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from models.attention_model import MultiObjectiveAttentionModel
from rl.environment import parse_action_sequence_to_routes
from rl.rewards import evaluate_solution_metrics


class NeuralPolicySolver:
    """
    Solves CVRP / CVRPTW instances using a trained Multi-Objective Attention Model.
    """
    def __init__(
        self,
        checkpoint_path: Optional[str] = "saved_checkpoints/mo_cvrp_model.pt",
        device: Optional[str] = None
    ):
        import os
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError("A trained checkpoint is required for neural inference")
        attributes = getattr(os.stat(checkpoint_path), "st_file_attributes", 0)
        if attributes & (0x1000 | 0x40000 | 0x400000):
            raise FileNotFoundError("Neural checkpoint is an offline OneDrive placeholder. Choose 'Always keep on this device' for saved_checkpoints, then retry.")
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = MultiObjectiveAttentionModel(
            node_dim=4,
            embedding_dim=128,
            num_layers=3,
            num_heads=8
        ).to(self.device)

        import os
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise FileNotFoundError("A trained checkpoint is required for neural inference")
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        except OSError as exc:
            raise FileNotFoundError("Checkpoint cannot be read locally. Hydrate the OneDrive file or select another trained checkpoint.") from exc
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def solve(
        self,
        instance: Dict[str, Any],
        preference_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        fuel_type: str = "diesel",
        speed: float = 30.0,
        decode_type: str = "greedy"
    ) -> Dict[str, Any]:
        """
        Runs neural policy inference to find optimal routes.
        """
        start_time = time.perf_counter()

        coords = instance["coords"] # (N+1, 2)
        demands = instance["demands"] # (N+1,)
        capacity = instance["capacity"]
        dist_matrix = instance["distance_matrix"]
        num_nodes = len(demands)
        if capacity <= 0 or np.any(np.asarray(demands)[1:] > capacity) or np.any(np.asarray(demands) < 0):
            raise ValueError("All parcels must fit the neural decoder's vehicle capacity")

        # Normalize coordinates to [0, 1] range for neural net
        min_c = coords.min(axis=0)
        max_c = coords.max(axis=0)
        span = np.maximum(max_c - min_c, 1e-5)
        norm_coords = (coords - min_c) / span

        # Normalize demands
        norm_demands = demands.astype(np.float32) / float(capacity)

        # Build tensor features: [x, y, demand, 0]
        node_feats = np.zeros((1, num_nodes, 4), dtype=np.float32)
        node_feats[0, :, :2] = norm_coords
        node_feats[0, :, 2] = norm_demands

        t_node_features = torch.tensor(node_feats, dtype=torch.float32, device=self.device)
        t_demands = torch.tensor(norm_demands[np.newaxis, :], dtype=torch.float32, device=self.device)
        t_weights = torch.tensor([preference_weights], dtype=torch.float32, device=self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(
                node_features=t_node_features,
                demands=t_demands,
                preference_weights=t_weights,
                decode_type=decode_type
            )

        actions = out["actions"][0].cpu().numpy().tolist()
        routes = parse_action_sequence_to_routes(actions)
        visited = [n for r in routes for n in r if n != 0]
        if sorted(visited) != list(range(1, num_nodes)):
            raise RuntimeError("Neural decoder did not produce complete unique customer coverage")

        elapsed_sec = time.perf_counter() - start_time

        metrics = evaluate_solution_metrics(
            routes=routes,
            distance_matrix=dist_matrix,
            demands=demands,
            time_windows=instance.get("time_windows"),
            service_times=instance.get("service_times"),
            speed=speed,
            fuel_type=fuel_type
        )

        return {
            "algorithm": f"Neural Attention Policy (w={preference_weights})",
            "routes": routes,
            "solve_time_sec": round(elapsed_sec, 5),
            "preference_weights": preference_weights,
            "metrics": metrics
        }
