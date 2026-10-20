"""
baselines/nearest_neighbor.py
Greedy Nearest Neighbor Heuristic for CVRP and CVRPTW.

Progressively builds routes starting from the depot:
At each step, selects the closest unvisited feasible customer (within remaining capacity).
When no customer can be added without exceeding capacity, the vehicle returns to the depot
and a new route starts.
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np
from rl.rewards import evaluate_solution_metrics


class NearestNeighborSolver:
    """
    Greedy Nearest Neighbor Baseline.
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
        Solves CVRP instance greedily by nearest neighbor.
        """
        start_time = time.perf_counter()

        demands = instance["demands"]
        dist_matrix = instance["distance_matrix"]
        capacity = instance["capacity"]
        if capacity <= 0 or np.any(np.asarray(demands)[1:] > capacity) or np.any(np.asarray(demands) < 0):
            raise ValueError("All demands must be nonnegative and fit a positive vehicle capacity")
        time_windows = instance.get("time_windows")
        service_times = instance.get("service_times")
        num_nodes = len(demands)

        unvisited = set(range(1, num_nodes))
        routes = []

        while unvisited:
            current_route = [0]
            current_node = 0
            current_load = 0

            while True:
                # Find feasible unvisited customer that minimizes distance from current_node
                best_next = None
                best_dist = float("inf")

                for cand in unvisited:
                    if current_load + demands[cand] <= capacity:
                        d = dist_matrix[current_node][cand]
                        if d < best_dist:
                            best_dist = d
                            best_next = cand

                if best_next is not None:
                    current_route.append(best_next)
                    current_load += demands[best_next]
                    current_node = best_next
                    unvisited.remove(best_next)
                else:
                    # No feasible customer fits -> return to depot
                    current_route.append(0)
                    routes.append(current_route)
                    break

        elapsed_sec = time.perf_counter() - start_time

        metrics = evaluate_solution_metrics(
            routes=routes,
            distance_matrix=dist_matrix,
            demands=demands,
            time_windows=time_windows,
            service_times=service_times,
            speed=speed,
            fuel_type=fuel_type
        )

        return {
            "algorithm": "Nearest Neighbor",
            "routes": routes,
            "solve_time_sec": round(elapsed_sec, 5),
            "metrics": metrics
        }
