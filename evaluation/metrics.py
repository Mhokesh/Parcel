"""
evaluation/metrics.py
Comparative Metrics, Optimality Gap Analysis, and LaTeX/Markdown Table Generator.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


def compute_optimality_gap(solution_cost: float, best_known_cost: float) -> float:
    """
    Computes percentage optimality gap:
    Gap (%) = ((Cost - BestKnown) / BestKnown) * 100
    """
    if best_known_cost <= 0:
        return 0.0
    return float(round(((solution_cost - best_known_cost) / best_known_cost) * 100.0, 2))


def generate_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compiles solver results into a structured comparative DataFrame.
    """
    rows = []
    # Identify best (minimum) distance among all algorithms as reference
    valid_dists = [r["metrics"]["total_distance"] for r in results if "metrics" in r and "total_distance" in r["metrics"]]
    best_dist = min(valid_dists) if valid_dists else 1.0

    for r in results:
        m = r.get("metrics", {})
        dist = m.get("total_distance", 0.0)
        gap = compute_optimality_gap(dist, best_dist)
        t_sec = r.get("solve_time_sec", 0.0)

        rows.append({
            "Algorithm": r.get("algorithm", "Unknown"),
            "Routes": m.get("num_routes", 0),
            "Distance (km)": dist,
            "Fuel Consumed": f"{m.get('total_fuel_consumed', 0.0)} {m.get('fuel_unit', 'L')}",
            "CO2 (kg)": m.get("total_co2_kg", 0.0),
            "Fuel Cost (INR)": f"Rs. {m.get('total_fuel_cost_inr', 0.0):,.2f}",
            "TW Violations": m.get("time_window_violations", 0),
            "Gap vs Best (%)": f"{gap:>+5.2f}%" if gap > 0 else "0.00% (Best)",
            "Solve Time": f"{t_sec * 1000:.1f} ms" if t_sec < 1.0 else f"{t_sec:.2f} s"
        })

    df = pd.DataFrame(rows)
    return df


def export_markdown_report(df: pd.DataFrame, title: str = "Benchmark Evaluation Results") -> str:
    """
    Converts results DataFrame to a formatted markdown report.
    """
    md = f"### {title}\n\n"
    md += df.to_markdown(index=False)
    md += "\n\n*Evaluation completed across distance, load-dependent emissions (CMEM), and time window constraints.*"
    return md
