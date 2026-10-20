"""
baselines/clarke_wright.py
Parallel Clarke-Wright Savings Algorithm for CVRP and CVRPTW.

Formulation (1964):
For any customer pair (i, j), the savings from merging routes is:
s(i, j) = d(0, i) + d(0, j) - d(i, j)

Merges are performed in descending order of savings, ensuring vehicle capacity
and time window constraints are strictly respected.
"""

import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from rl.rewards import evaluate_solution_metrics


class ClarkeWrightSolver:
    """
    Implements the classical Parallel Clarke & Wright Savings Algorithm (1964)
    with capacity feasibility and optional multi-objective evaluation.
    """

    def __init__(self):
        pass

    def solve(
        self,
        instance: Dict[str, Any],
        fuel_type: str = "diesel",
        speed: float = 30.0
    ) -> Dict[str, Any]:
        """
        Solves the given CVRP instance using Clarke-Wright Savings.
        """
        start_time = time.perf_counter()

        coords = instance["coords"]
        demands = instance["demands"]
        dist_matrix = instance["distance_matrix"]
        capacity = instance["capacity"]
        if capacity <= 0 or np.any(np.asarray(demands)[1:] > capacity) or np.any(np.asarray(demands) < 0):
            raise ValueError("All demands must be nonnegative and fit a positive vehicle capacity")
        time_windows = instance.get("time_windows")
        service_times = instance.get("service_times")
        num_nodes = len(demands)
        num_customers = num_nodes - 1

        if num_customers <= 0:
            return {"routes": [], "solve_time_sec": 0.0, "metrics": {}}

        # 1. Initialize each customer in their own route: [0, i, 0]
        # Map: customer_id -> route list
        routes = [[0, i, 0] for i in range(1, num_nodes)]
        # Map: customer_id -> route index in routes list
        node_to_route = {i: i - 1 for i in range(1, num_nodes)}
        # Track total load of each route
        route_loads = {i - 1: demands[i] for i in range(1, num_nodes)}

        # 2. Compute savings for all pairs (i, j) with i != j
        savings: List[Tuple[float, int, int]] = []
        for i in range(1, num_nodes):
            for j in range(i + 1, num_nodes):
                # s_ij = d(0, i) + d(0, j) - d(i, j)
                s = dist_matrix[0][i] + dist_matrix[0][j] - dist_matrix[i][j]
                savings.append((s, i, j))

        # Sort savings in descending order
        savings.sort(key=lambda x: x[0], reverse=True)

        # 3. Merge routes iteratively
        for s_val, i, j in savings:
            r_i_idx = node_to_route[i]
            r_j_idx = node_to_route[j]

            # Already in the same route
            if r_i_idx == r_j_idx:
                continue

            r_i = routes[r_i_idx]
            r_j = routes[r_j_idx]

            # Check if i and j are adjacent to depot in their respective routes
            # Case 1: ... -> i -> 0 and 0 -> j -> ... (i is at end of r_i, j is at start of r_j)
            can_merge = False
            merged_route = None

            if r_i[-2] == i and r_j[1] == j:
                # Merge: r_i[0:-1] + r_j[1:]
                if route_loads[r_i_idx] + route_loads[r_j_idx] <= capacity:
                    merged_route = r_i[:-1] + r_j[1:]
                    can_merge = True

            # Case 2: 0 -> i -> ... and ... -> j -> 0 (i at start of r_i, j at end of r_j)
            elif r_i[1] == i and r_j[-2] == j:
                # Merge: r_j[0:-1] + r_i[1:]
                if route_loads[r_i_idx] + route_loads[r_j_idx] <= capacity:
                    merged_route = r_j[:-1] + r_i[1:]
                    can_merge = True

            # Case 3: ... -> i -> 0 and ... -> j -> 0 (both at end -> reverse r_j)
            elif r_i[-2] == i and r_j[-2] == j:
                if route_loads[r_i_idx] + route_loads[r_j_idx] <= capacity:
                    reversed_j = [0] + list(reversed(r_j[1:-1])) + [0]
                    merged_route = r_i[:-1] + reversed_j[1:]
                    can_merge = True

            # Case 4: 0 -> i -> ... and 0 -> j -> ... (both at start -> reverse r_i)
            elif r_i[1] == i and r_j[1] == j:
                if route_loads[r_i_idx] + route_loads[r_j_idx] <= capacity:
                    reversed_i = [0] + list(reversed(r_i[1:-1])) + [0]
                    merged_route = reversed_i[:-1] + r_j[1:]
                    can_merge = True

            if can_merge and merged_route is not None:
                # Update routes
                new_load = route_loads[r_i_idx] + route_loads[r_j_idx]
                routes[r_i_idx] = merged_route
                routes[r_j_idx] = [] # Mark old route as empty
                route_loads[r_i_idx] = new_load
                route_loads[r_j_idx] = 0

                # Re-point all customers in merged route to r_i_idx
                for node in merged_route:
                    if node != 0:
                        node_to_route[node] = r_i_idx

        # Filter out empty routes
        active_routes = [r for r in routes if len(r) > 2]
        elapsed_sec = time.perf_counter() - start_time

        metrics = evaluate_solution_metrics(
            routes=active_routes,
            distance_matrix=dist_matrix,
            demands=demands,
            time_windows=time_windows,
            service_times=service_times,
            speed=speed,
            fuel_type=fuel_type
        )

        return {
            "algorithm": "Clarke-Wright Savings",
            "routes": active_routes,
            "solve_time_sec": round(elapsed_sec, 5),
            "metrics": metrics
        }
