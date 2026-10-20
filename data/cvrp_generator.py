"""
data/cvrp_generator.py
Synthetic CVRP and CVRPTW Instance Generator.

Supports:
- Uniform spatial customer distribution
- Clustered spatial distribution (real-world suburban/urban clusters)
- Dynamic demand generation
- Soft and hard Time Window generation
"""

import math
import numpy as np
from typing import Dict, Any, Optional, Tuple


class CVRPInstanceGenerator:
    """
    Generates synthetic Capacitated Vehicle Routing Problem instances
    with optional Time Windows and realistic payload/fleet configurations.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        grid_size: float = 100.0,
        vehicle_capacity: int = 50,
        min_demand: int = 1,
        max_demand: int = 10,
        max_time_horizon: float = 480.0, # 8-hour shift in minutes
        avg_speed: float = 30.0 # km/h
    ):
        self.rng = np.random.RandomState(seed)
        self.grid_size = grid_size
        self.vehicle_capacity = vehicle_capacity
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.max_time_horizon = max_time_horizon
        self.avg_speed = avg_speed

    def generate_instance(
        self,
        num_customers: int = 20,
        distribution: str = "uniform",
        num_clusters: int = 3,
        include_time_windows: bool = True,
        instance_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates an instance containing:
        - depot coordinates (index 0)
        - customer coordinates (indices 1 to n)
        - demands for all nodes (depot demand = 0)
        - distance matrix (Euclidean or scaled)
        - time windows [ready_time, due_time] for each node
        - service times for each customer
        """
        if instance_name is None:
            instance_name = f"synthetic_n{num_customers}_{distribution}"

        # 1. Depot at center or random
        depot_coord = np.array([self.grid_size / 2.0, self.grid_size / 2.0])

        # 2. Customer coordinates
        if distribution == "uniform":
            customer_coords = self.rng.uniform(0.0, self.grid_size, size=(num_customers, 2))
        elif distribution == "clustered":
            # Cluster centers
            centers = self.rng.uniform(0.2 * self.grid_size, 0.8 * self.grid_size, size=(num_clusters, 2))
            customer_coords = []
            for i in range(num_customers):
                c = centers[self.rng.choice(num_clusters)]
                # Add Gaussian noise around cluster center
                noise = self.rng.normal(0, self.grid_size / 10.0, size=2)
                coord = np.clip(c + noise, 0.0, self.grid_size)
                customer_coords.append(coord)
            customer_coords = np.array(customer_coords)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        all_coords = np.vstack([depot_coord, customer_coords]) # Shape: (n + 1, 2)

        # 3. Demands: depot = 0, customers in [min_demand, max_demand]
        cust_demands = self.rng.randint(self.min_demand, self.max_demand + 1, size=num_customers)
        all_demands = np.concatenate([[0], cust_demands]).astype(int)

        # 4. Euclidean Distance Matrix
        diff = all_coords[:, np.newaxis, :] - all_coords[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

        # 5. Time Windows & Service Times
        time_windows = np.zeros((num_customers + 1, 2))
        time_windows[0] = [0.0, self.max_time_horizon] # Depot open whole shift

        service_times = np.zeros(num_customers + 1)

        if include_time_windows:
            for i in range(1, num_customers + 1):
                # Dist from depot to customer i
                travel_from_depot = (dist_matrix[0][i] / self.avg_speed) * 60.0 # in minutes
                # Random earliest start time
                earliest = self.rng.uniform(travel_from_depot, self.max_time_horizon * 0.6)
                # Window duration (e.g. 60 to 180 min)
                window_len = self.rng.uniform(60.0, 150.0)
                latest = min(self.max_time_horizon, earliest + window_len)

                time_windows[i] = [round(earliest, 1), round(latest, 1)]
                service_times[i] = round(self.rng.uniform(5.0, 15.0), 1) # 5 to 15 min unloading

        return {
            "name": instance_name,
            "num_customers": num_customers,
            "num_nodes": num_customers + 1,
            "capacity": self.vehicle_capacity,
            "coords": all_coords,
            "demands": all_demands,
            "distance_matrix": dist_matrix,
            "time_windows": time_windows if include_time_windows else None,
            "service_times": service_times if include_time_windows else None,
            "grid_size": self.grid_size,
            "max_time_horizon": self.max_time_horizon
        }


def sample_cvrp_batch(
    batch_size: int = 64,
    num_customers: int = 20,
    capacity: int = 30,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Fast tensor batch generator for PyTorch RL training environment.
    Coordinates are normalized to [0, 1] range, demands normalized by capacity.
    """
    rng = np.random.RandomState(seed)

    # Depot coords: (batch_size, 1, 2)
    depots = rng.uniform(0.0, 1.0, size=(batch_size, 1, 2)).astype(np.float32)

    # Customer coords: (batch_size, num_customers, 2)
    customers = rng.uniform(0.0, 1.0, size=(batch_size, num_customers, 2)).astype(np.float32)

    # Raw Demands: (batch_size, num_customers) between 1 and 9
    raw_demands = rng.randint(1, 10, size=(batch_size, num_customers)).astype(np.float32)
    # Normalized demands by vehicle capacity
    norm_demands = raw_demands / float(capacity)

    return {
        "depots": depots,
        "customers": customers,
        "raw_demands": raw_demands,
        "norm_demands": norm_demands,
        "capacity": capacity,
        "num_customers": num_customers,
        "batch_size": batch_size
    }
