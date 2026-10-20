"""
evaluation/pareto.py
Multi-Objective Pareto Frontier Analysis & Non-Dominated Sorting for MO-CVRP.
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def is_dominated(p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
    """
    Returns True if point p1 is dominated by point p2 (minimization problem).
    p2 dominates p1 if p2 is <= p1 in all objectives and strictly < in at least one.
    """
    return (p2[0] <= p1[0] and p2[1] <= p1[1]) and (p2[0] < p1[0] or p2[1] < p1[1])


def extract_pareto_front(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filters a list of solution points to only the non-dominated Pareto-optimal set.
    Each item must contain 'distance' and 'co2_kg' (or 'fuel_consumed').
    """
    pareto_set = []

    for i, p_cand in enumerate(points):
        cand_pt = (p_cand["distance"], p_cand["co2_kg"])
        dominated = False

        for j, p_other in enumerate(points):
            if i == j:
                continue
            other_pt = (p_other["distance"], p_other["co2_kg"])
            if is_dominated(cand_pt, other_pt):
                dominated = True
                break

        if not dominated:
            p_cand_copy = dict(p_cand)
            p_cand_copy["is_pareto_optimal"] = True
            pareto_set.append(p_cand_copy)

    # Sort Pareto points by distance ascending
    pareto_set.sort(key=lambda x: x["distance"])
    return pareto_set


def compute_hypervolume_2d(
    pareto_points: List[Dict[str, Any]],
    ref_point: Tuple[float, float] = (500.0, 150.0)
) -> float:
    """
    Computes 2D hypervolume indicator relative to a reference point (max_dist, max_co2).
    Higher hypervolume implies better Pareto coverage and convergence.
    """
    if not pareto_points:
        return 0.0

    pts = sorted([(p["distance"], p["co2_kg"]) for p in pareto_points], key=lambda x: x[0])

    # Filter points within reference bounds
    valid_pts = [p for p in pts if p[0] <= ref_point[0] and p[1] <= ref_point[1]]
    if not valid_pts:
        return 0.0

    hv = 0.0
    curr_y = ref_point[1]

    for x, y in valid_pts:
        if y < curr_y:
            hv += (ref_point[0] - x) * (curr_y - y)
            curr_y = y

    return float(round(hv, 2))
