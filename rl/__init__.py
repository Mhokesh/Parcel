# rl package
from rl.rewards import evaluate_solution_metrics, compute_leg_fuel_and_emissions, FuelAndEmissionConfig, compute_scalarized_objective

def __getattr__(name):
    if name in ("compute_batch_costs", "parse_action_sequence_to_routes"):
        from rl import environment
        return getattr(environment, name)
    if name == "RLTrainer":
        from rl.trainer import RLTrainer
        return RLTrainer
    raise AttributeError(name)

__all__ = [
    "evaluate_solution_metrics",
    "compute_leg_fuel_and_emissions",
    "FuelAndEmissionConfig",
    "compute_scalarized_objective",
    "compute_batch_costs",
    "parse_action_sequence_to_routes",
    "RLTrainer"
]
