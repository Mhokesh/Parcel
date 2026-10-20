"""Classical routing baselines."""
def __getattr__(name):
    if name == "NearestNeighborSolver":
        from baselines.nearest_neighbor import NearestNeighborSolver
        return NearestNeighborSolver
    if name == "ClarkeWrightSolver":
        from baselines.clarke_wright import ClarkeWrightSolver
        return ClarkeWrightSolver
    if name == "ORToolsSolver":
        from baselines.ortools_solver import ORToolsSolver
        return ORToolsSolver
    raise AttributeError(name)
