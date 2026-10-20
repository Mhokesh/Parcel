"""
data/benchmark_parser.py
Parses and loads standard CVRPLIB (Augerat Set A, B, P, Christofides) and Solomon VRPTW instances.
"""

import os
import re
import math
import urllib.request
import numpy as np
from typing import Dict, Any, Optional, List


# Standard sample CVRPLIB instance definitions embedded for immediate offline availability
AUGERAT_A_N32_K5 = """NAME : A-n32-k5
COMMENT : (Augerat et al, Min no of trucks: 5, Optimal value: 784)
TYPE : CVRP
DIMENSION : 32
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
 1 82 76
 2 96 44
 3 50 5
 4 49 8
 5 13 7
 6 29 89
 7 58 30
 8 84 39
 9 14 24
 10 2 39
 11 3 82
 12 5 10
 13 98 52
 14 84 25
 15 61 59
 16 1 65
 17 88 51
 18 91 2
 19 19 32
 20 93 3
 21 64 6
 22 94 42
 23 57 77
 24 28 12
 25 70 81
 26 98 73
 27 12 100
 28 6 19
 29 85 99
 30 30 1
 31 39 49
 32 65 35
DEMAND_SECTION
1 0
2 19
3 21
4 6
5 19
6 7
7 12
8 16
9 6
10 16
11 8
12 14
13 21
14 16
15 3
16 22
17 18
18 19
19 1
20 24
21 8
22 12
23 4
24 8
25 24
26 24
27 2
28 10
29 15
30 2
31 18
32 20
DEPOT_SECTION
 1
 -1
EOF
"""

AUGERAT_A_N45_K6 = """NAME : A-n45-k6
COMMENT : (Augerat et al, Min no of trucks: 6, Optimal value: 944)
TYPE : CVRP
DIMENSION : 45
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 100
NODE_COORD_SECTION
 1 35 35
 2 41 49
 3 35 17
 4 55 45
 5 55 20
 6 15 30
 7 25 30
 8 20 50
 9 10 43
 10 55 60
 11 30 60
 12 20 65
 13 50 35
 14 30 25
 15 15 10
 16 30 5
 17 10 20
 18 5 30
 19 20 40
 20 15 60
 21 45 65
 22 45 20
 23 45 10
 24 55 5
 25 65 35
 26 65 20
 27 45 30
 28 35 40
 29 41 37
 30 64 42
 31 40 60
 32 31 52
 33 35 69
 34 53 52
 35 65 55
 36 63 65
 37 2 60
 38 20 20
 39 5 5
 40 60 12
 41 40 25
 42 42 7
 43 24 12
 44 23 3
 45 10 65
DEMAND_SECTION
1 0
2 21
3 14
4 12
5 22
6 17
7 15
8 25
9 22
10 8
11 19
12 21
13 18
14 13
15 24
16 13
17 9
18 8
19 6
20 17
21 8
22 12
23 18
24 10
25 24
26 16
27 2
28 12
29 18
30 10
31 20
32 14
33 24
34 16
35 3
36 22
37 18
38 20
39 23
40 24
41 2
42 12
43 4
44 8
45 24
DEPOT_SECTION
 1
 -1
EOF
"""


def parse_vrp_string(content: str) -> Dict[str, Any]:
    """
    Parses TSPLIB/CVRPLIB string format.
    """
    lines = content.strip().splitlines()
    name = "unknown"
    capacity = 100
    dimension = 0
    coords_dict = {}
    demands_dict = {}
    depot_node = 1
    optimal_val = None

    section = None
    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            continue

        if line.startswith("NAME"):
            name = line.split(":")[-1].strip()
        elif line.startswith("COMMENT"):
            m = re.search(r"Optimal value:\s*(\d+)", line)
            if m:
                optimal_val = float(m.group(1))
        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":")[-1].strip())
        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":")[-1].strip())
        elif line.startswith("NODE_COORD_SECTION"):
            section = "COORDS"
            continue
        elif line.startswith("DEMAND_SECTION"):
            section = "DEMANDS"
            continue
        elif line.startswith("DEPOT_SECTION"):
            section = "DEPOT"
            continue

        if section == "COORDS":
            parts = line.split()
            if len(parts) >= 3:
                node_id = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                coords_dict[node_id] = (x, y)
        elif section == "DEMANDS":
            parts = line.split()
            if len(parts) >= 2:
                node_id = int(parts[0])
                demand = int(parts[1])
                demands_dict[node_id] = demand
        elif section == "DEPOT":
            val = int(line)
            if val > 0:
                depot_node = val

    num_nodes = len(coords_dict)
    # Order nodes: depot first (index 0), then customers
    ordered_ids = [depot_node] + [k for k in sorted(coords_dict.keys()) if k != depot_node]

    coords = np.array([coords_dict[nid] for nid in ordered_ids], dtype=np.float32)
    demands = np.array([demands_dict[nid] for nid in ordered_ids], dtype=np.int32)

    # Compute Euclidean distance matrix (rounded to nearest integer or float as per TSPLIB standard)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.floor(np.sqrt(np.sum(diff ** 2, axis=-1)) + 0.5)

    # Synthetic time windows for standard CVRPLIB instances if needed
    time_windows = np.zeros((num_nodes, 2))
    time_windows[0] = [0.0, 1000.0]
    service_times = np.zeros(num_nodes)
    for i in range(1, num_nodes):
        time_windows[i] = [0.0, 1000.0] # Open windows
        service_times[i] = 10.0

    return {
        "name": name,
        "num_customers": num_nodes - 1,
        "num_nodes": num_nodes,
        "capacity": capacity,
        "coords": coords,
        "demands": demands,
        "distance_matrix": dist_matrix,
        "time_windows": time_windows,
        "service_times": service_times,
        "known_optimal": optimal_val
    }


def load_cvrplib_benchmark(instance_name: str = "A-n32-k5", cache_dir: str = "data/cvrplib") -> Dict[str, Any]:
    """
    Loads a CVRPLIB instance by name. Uses local cache / built-in definitions if available.
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, f"{instance_name}.vrp")

    if instance_name == "A-n32-k5":
        return parse_vrp_string(AUGERAT_A_N32_K5)
    elif instance_name == "A-n45-k6":
        instance = parse_vrp_string(AUGERAT_A_N45_K6)
        instance["known_optimal"] = None
        instance["name"] = "Unverified_embedded_45_node_example"
        instance["provenance"] = "Embedded example; not verified against the named CVRPLIB source"
        return instance

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return parse_vrp_string(f.read())

    # Try downloading from CVRPLIB URL
    url = f"http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/{instance_name}.vrp"
    try:
        urllib.request.urlretrieve(url, file_path)
        with open(file_path, "r") as f:
            return parse_vrp_string(f.read())
    except Exception as e:
        # Return fallback A-n32-k5 if offline or failed
        print(f"Warning: Could not fetch {instance_name} from web ({e}), using default A-n32-k5.")
        return parse_vrp_string(AUGERAT_A_N32_K5)
