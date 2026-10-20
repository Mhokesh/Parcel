"""
rl/environment.py
Vectorized Multi-Objective Reward & Route Construction for PyTorch Batches.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from rl.rewards import FuelAndEmissionConfig, evaluate_solution_metrics


def compute_batch_costs(
    coords: torch.Tensor,
    demands: torch.Tensor,
    actions: torch.Tensor,
    preference_weights: torch.Tensor,
    fuel_type: str = "diesel",
    vehicle_capacity: float = 30.0
) -> Dict[str, torch.Tensor]:
    """
    Computes multi-objective costs and scalarized rewards for a batch of tours.

    coords: (batch_size, num_nodes, 2)
    demands: (batch_size, num_nodes) - normalized demand [0, 1]
    actions: (batch_size, seq_len) - sequence of node indices visited
    preference_weights: (batch_size, 3) - (w_dist, w_fuel, w_tw)
    """
    batch_size, num_nodes, _ = coords.size()
    seq_len = actions.size(1)
    device = coords.device

    # Prepend depot (0) to actions sequence: tour starts at depot
    full_tour = torch.cat([torch.zeros((batch_size, 1), dtype=torch.long, device=device), actions], dim=-1)

    # 1. Total Travel Distance
    # Gather node coordinates along the tour
    # full_tour: (batch_size, seq_len + 1)
    tour_coords = []
    for b in range(batch_size):
        idx = full_tour[b]
        tour_coords.append(coords[b, idx])
    tour_coords = torch.stack(tour_coords, dim=0) # (batch_size, seq_len + 1, 2)

    # Step-by-step Euclidean distances
    step_diffs = tour_coords[:, 1:, :] - tour_coords[:, :-1, :]
    step_dists = torch.linalg.vector_norm(step_diffs, dim=-1)
    total_distance = torch.sum(step_dists, dim=-1) # (batch_size,)

    # 2. CMEM Fuel Consumption & CO2 Emissions
    # Track payload on each leg: payload starts as sum of customer demands on the route
    total_fuel = torch.zeros(batch_size, device=device)
    total_co2 = torch.zeros(batch_size, device=device)

    # Base & load coefficients based on fuel type
    if fuel_type == "diesel":
        base_rate = FuelAndEmissionConfig.DIESEL_BASE_RATE
        load_coeff = FuelAndEmissionConfig.DIESEL_LOAD_COEFF * vehicle_capacity
        co2_factor = FuelAndEmissionConfig.DIESEL_CO2_PER_LITER
    elif fuel_type == "petrol":
        base_rate = FuelAndEmissionConfig.PETROL_BASE_RATE
        load_coeff = FuelAndEmissionConfig.PETROL_LOAD_COEFF * vehicle_capacity
        co2_factor = FuelAndEmissionConfig.PETROL_CO2_PER_LITER
    else: # ev
        base_rate = FuelAndEmissionConfig.EV_BASE_RATE_KWH
        load_coeff = FuelAndEmissionConfig.EV_LOAD_COEFF_KWH * vehicle_capacity
        co2_factor = FuelAndEmissionConfig.EV_CO2_PER_KWH

    for b in range(batch_size):
        sequence = actions[b].tolist()
        def outgoing_load(start):
            load = 0.0
            for node in sequence[start:]:
                if node == 0:
                    break
                load += demands[b, node].item()
            return load
        curr_load = outgoing_load(0)
        b_fuel = 0.0
        for s in range(seq_len):
            leg_dist = step_dists[b, s]
            next_node = actions[b, s].item()

            # The incoming leg carries the parcel until it is delivered.
            leg_fuel = leg_dist * (base_rate + load_coeff * curr_load)
            b_fuel += leg_fuel
            if next_node == 0:
                # Returned to depot
                curr_load = outgoing_load(s + 1)
            else:
                curr_load = max(0.0, curr_load - demands[b, next_node].item())


        total_fuel[b] = b_fuel
        total_co2[b] = b_fuel * co2_factor

    # 3. Normalization & Multi-Objective Scalarization
    # Reference scales
    dist_norm = total_distance.mean().detach() + 1e-5
    co2_norm = total_co2.mean().detach() + 1e-5

    w_dist = preference_weights[:, 0]
    w_co2 = preference_weights[:, 1]
    w_tw = preference_weights[:, 2]

    norm_d = total_distance / dist_norm
    norm_c = total_co2 / co2_norm

    # Scalarized objective to minimize: J(pi, w) = w1 * norm_dist + w2 * norm_co2
    scalarized_cost = w_dist * norm_d + w_co2 * norm_c
    reward = -scalarized_cost # Reward is negative cost (maximization)

    return {
        "reward": reward,
        "scalarized_cost": scalarized_cost,
        "total_distance": total_distance,
        "total_fuel": total_fuel,
        "total_co2": total_co2
    }


def parse_action_sequence_to_routes(action_sequence: List[int]) -> List[List[int]]:
    """
    Converts a flat sequence of actions (e.g. [3, 5, 0, 1, 4, 0])
    into a structured list of individual vehicle routes starting and ending with 0.
    """
    routes = []
    current_route = [0]

    for node in action_sequence:
        if node == 0:
            if len(current_route) > 1:
                current_route.append(0)
                routes.append(current_route)
                current_route = [0]
        else:
            current_route.append(node)

    if len(current_route) > 1:
        current_route.append(0)
        routes.append(current_route)

    return routes
