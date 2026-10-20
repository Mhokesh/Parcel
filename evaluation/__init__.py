"""Comparison metrics and Pareto-front utilities for routing experiments."""
from evaluation.metrics import compute_optimality_gap, generate_comparison_table, export_markdown_report
from evaluation.pareto import is_dominated, extract_pareto_front, compute_hypervolume_2d

__all__ = [
    "compute_optimality_gap", "generate_comparison_table", "export_markdown_report",
    "is_dominated", "extract_pareto_front", "compute_hypervolume_2d",
]
