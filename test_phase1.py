"""
test_phase1.py
Verification script for Phase 1: Generators, CVRPLIB loader, Clarke-Wright, Nearest Neighbor, and OR-Tools.
"""

from data.cvrp_generator import CVRPInstanceGenerator
from data.benchmark_parser import load_cvrplib_benchmark
from baselines.clarke_wright import ClarkeWrightSolver
from baselines.nearest_neighbor import NearestNeighborSolver
from baselines.ortools_solver import ORToolsSolver


def run_phase1_verification():
    print("=" * 60)
    print("PHASE 1 VERIFICATION: BASES & SOLVERS")
    print("=" * 60)

    # 1. Test Synthetic Instance Generator
    gen = CVRPInstanceGenerator(seed=42, vehicle_capacity=40)
    instance = gen.generate_instance(num_customers=15, distribution="clustered", include_time_windows=True)
    print(f"Generated instance '{instance['name']}': {instance['num_customers']} customers, capacity {instance['capacity']}")

    # 2. Test Solvers on Synthetic Instance
    cw = ClarkeWrightSolver()
    nn = NearestNeighborSolver()
    ortools = ORToolsSolver(time_limit_seconds=2)

    res_nn = nn.solve(instance, fuel_type="diesel")
    res_cw = cw.solve(instance, fuel_type="diesel")
    res_ortools = ortools.solve(instance, fuel_type="diesel")

    print("\n--- Synthetic Benchmark Comparison (Diesel Fleet) ---")
    for res in [res_nn, res_cw, res_ortools]:
        algo = res["algorithm"]
        dist = res["metrics"]["total_distance"]
        fuel = res["metrics"]["total_fuel_consumed"]
        co2 = res["metrics"]["total_co2_kg"]
        cost = res["metrics"]["total_fuel_cost_inr"]
        t = res["solve_time_sec"]
        print(f"[{algo:<26}] Dist: {dist:>6.1f} km | Fuel: {fuel:>5.2f} L | CO2: {co2:>5.2f} kg | Cost: Rs.{cost:>7.2f} | Time: {t*1000:>6.2f} ms")

    # 3. Test CVRPLIB Benchmark (Augerat A-n32-k5)
    print("\n--- Standard CVRPLIB Benchmark: A-n32-k5 ---")
    cvrplib_inst = load_cvrplib_benchmark("A-n32-k5")
    cvrp_nn = nn.solve(cvrplib_inst, fuel_type="petrol")
    cvrp_cw = cw.solve(cvrplib_inst, fuel_type="petrol")
    cvrp_ortools = ortools.solve(cvrplib_inst, fuel_type="petrol")

    for res in [cvrp_nn, cvrp_cw, cvrp_ortools]:
        algo = res["algorithm"]
        dist = res["metrics"]["total_distance"]
        fuel = res["metrics"]["total_fuel_consumed"]
        co2 = res["metrics"]["total_co2_kg"]
        cost = res["metrics"]["total_fuel_cost_inr"]
        t = res["solve_time_sec"]
        print(f"[{algo:<26}] Dist: {dist:>6.1f} | Fuel: {fuel:>5.2f} L | CO2: {co2:>5.2f} kg | Cost: Rs.{cost:>7.2f} | Time: {t*1000:>6.2f} ms")

    # 4. EV Fleet Test
    print("\n--- Electric Vehicle (EV) Fleet Test ---")
    ev_ortools = ortools.solve(instance, fuel_type="ev")
    print(f"EV Energy Consumed: {ev_ortools['metrics']['total_fuel_consumed']} kWh | CO2: {ev_ortools['metrics']['total_co2_kg']} kg | Cost: Rs.{ev_ortools['metrics']['total_fuel_cost_inr']}")

    print("\n[SUCCESS] Phase 1 verification complete.")


if __name__ == "__main__":
    run_phase1_verification()
