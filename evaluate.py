"""Reproducible homogeneous research comparisons; operational fleet planning lives in web/planner.py."""
import argparse
import json
import platform
import sys
from pathlib import Path
sys.pycache_prefix = str(Path(__file__).resolve().parent / "runtime/pycache")
from datetime import datetime, timezone
from baselines.nearest_neighbor import NearestNeighborSolver
from baselines.clarke_wright import ClarkeWrightSolver
from baselines.ortools_solver import ORToolsSolver
from data.cvrp_generator import CVRPInstanceGenerator
from data.benchmark_parser import load_cvrplib_benchmark
from real_world.coimbatore_data import get_coimbatore_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["all", "coimbatore", "synthetic", "a32", "a45"], default="coimbatore")
    parser.add_argument("--algorithm", choices=["all", "ortools", "clarke_wright", "nearest_neighbor", "neural"], default="all")
    parser.add_argument("--fuel_type", choices=["diesel", "petrol", "ev"], default="diesel")
    parser.add_argument("--checkpoint", default="saved_checkpoints/mo_cvrp_model.pt")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    datasets = ["coimbatore", "synthetic", "a32"] if args.dataset == "all" else [args.dataset]
    results = []
    for dataset in datasets:
        if dataset == "coimbatore":
            instance = get_coimbatore_instance()
        elif dataset == "synthetic":
            instance = CVRPInstanceGenerator(seed=args.seed, vehicle_capacity=40).generate_instance(20)
        else:
            instance = load_cvrplib_benchmark("A-n32-k5" if dataset == "a32" else "A-n45-k6")
        for name in (["ortools", "clarke_wright", "nearest_neighbor", "neural"] if args.algorithm == "all" else [args.algorithm]):
            try:
                solver = {"ortools": ORToolsSolver(time_limit_seconds=2), "clarke_wright": ClarkeWrightSolver(), "nearest_neighbor": NearestNeighborSolver()}.get(name)
                if name == "neural":
                    from models.policy_solver import NeuralPolicySolver
                    solver = NeuralPolicySolver(args.checkpoint)
                result = solver.solve(instance, fuel_type=args.fuel_type)
                if result.get("error"):
                    raise RuntimeError(result["error"])
                visited = [n for route in result["routes"] for n in route if n]
                if sorted(visited) != list(range(1, instance["num_nodes"])):
                    raise RuntimeError("Incomplete customer coverage")
                if any(sum(instance["demands"][n] for n in r if n) > instance["capacity"] for r in result["routes"]):
                    raise RuntimeError("Capacity violation")
                row = dict(dataset=instance["name"], algorithm=name, status="feasible", capacity=instance["capacity"], result=result)
                print(f"{instance['name']} | {name}: {result['metrics']['total_distance']} distance units, {result['solve_time_sec']*1000:.1f} ms")
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                row = dict(dataset=instance["name"], algorithm=name, status="unavailable", error=str(exc))
                print(f"{instance['name']} | {name}: unavailable ({exc})")
            results.append(row)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(), platform=platform.platform(), seed=args.seed, fuel_type=args.fuel_type,
                  scope="Single run per homogeneous instance; search/inference time excludes road-service calls. No certified optimality claim. Synthetic/benchmark units are not field-measured kilometres.", results=results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=lambda v: v.item() if hasattr(v, "item") else str(v)), encoding="utf-8")
        print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
