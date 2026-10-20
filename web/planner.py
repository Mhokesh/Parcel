"""Finite heterogeneous fleet planning, with auditable estimated metrics.

Distances are geographic estimates unless a configured OSRM table is available.
Time windows are soft; physical vehicle capacity and unique service are hard.
"""
import csv
import io
import math
import os
import time
from pathlib import Path
import numpy as np
import requests

from web.schemas import Hub, CustomInstanceData, SolveRequest
from rl.rewards import compute_leg_fuel_and_emissions

ROOT = Path(__file__).resolve().parents[1]
COLORS = ["#527f70", "#b87550", "#8c7db0", "#5488aa", "#b59b45", "#bd778e"]


def parse_csv(content, name="Imported orders"):
    reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")))
    if not {"id", "name", "lat", "lon"}.issubset(reader.fieldnames or []):
        raise ValueError("CSV needs id, name, lat, lon, and weight_kg (or demand) columns")
    hubs = []
    for row_num, row in enumerate(reader, 2):
        if len(hubs) >= 201:
            raise ValueError("Maximum 200 customers per dispatch")
        try:
            depot = int(row["id"]) == 0
            hubs.append(Hub(
                id=int(row["id"]), name=row["name"], lat=float(row["lat"]), lon=float(row["lon"]),
                demand=float(row.get("weight_kg", row.get("demand", "0"))),
                tw=(float(row.get("ready_min") or 0), float(row.get("due_min") or 480)),
                service=0 if depot else float(row.get("service_min") or 5),
                type="depot" if depot else "customer",
                **{k: row.get(k) or "" for k in ("customer_name", "address", "phone", "product", "size")},
            ))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"CSV row {row_num}: {exc}") from exc
    return CustomInstanceData(name=name, hubs=hubs)


def sample_data():
    return parse_csv((ROOT / "data/sample_50_orders.csv").read_text(encoding="utf-8"), "Coimbatore · 50 orders")


def load_problem(req):
    if req.dataset_type == "custom":
        data = req.custom_data
        hubs = [h.model_dump() for h in data.hubs]
        name = data.name
    elif req.dataset_type == "coimbatore":
        data = sample_data()
        hubs = [h.model_dump() for h in data.hubs[:req.num_customers + 1]]
        name = f"Coimbatore · {len(hubs)-1} orders"
    else:
        if req.dataset_type == "synthetic":
            from data.cvrp_generator import CVRPInstanceGenerator
            instance = CVRPInstanceGenerator(seed=42, grid_size=12, vehicle_capacity=req.vehicle_capacity).generate_instance(req.num_customers)
        else:
            from data.benchmark_parser import load_cvrplib_benchmark
            instance = load_cvrplib_benchmark("A-n32-k5" if req.dataset_type == "cvrplib_a32" else "A-n45-k6")
        name = instance["name"]
        # Explicitly illustrative map projection for non-geographic datasets.
        hubs = [Hub(id=i, name="Depot" if i == 0 else f"Order {i}",
                    lat=11.0168 + float(y)/111, lon=76.9558 + float(x)/109,
                    demand=float(instance["demands"][i]), service=0 if i == 0 else 5,
                    type="depot" if i == 0 else "customer").model_dump()
                for i, (x, y) in enumerate(instance["coords"] - instance["coords"][0])]
    # Stable external IDs are kept separately from solver array positions.
    for i, h in enumerate(hubs):
        h["node_index"] = i
    gps = np.array([[h["lat"], h["lon"]] for h in hubs])
    lat, lon = np.radians(gps).T
    dlat, dlon = lat[:, None] - lat, lon[:, None] - lon
    a = np.sin(dlat/2)**2 + np.cos(lat[:, None])*np.cos(lat)*np.sin(dlon/2)**2
    matrix = 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) * 1.28
    distance_source = "Geographic estimate (1.28 × great-circle distance)"
    osrm = os.getenv("OSRM_BASE_URL", "").rstrip("/")
    if osrm:
        coords = ";".join(f"{h['lon']},{h['lat']}" for h in hubs)
        try:
            response = requests.get(f"{osrm}/table/v1/driving/{coords}", params={"annotations": "distance"}, timeout=5)
            response.raise_for_status()
            candidate = np.array(response.json()["distances"], dtype=float) / 1000
            if candidate.shape == matrix.shape and np.isfinite(candidate).all() and (candidate >= 0).all():
                matrix = candidate
                distance_source = "OSRM road distance"
        except (requests.RequestException, ValueError, KeyError, TypeError):
            pass
    return hubs, matrix, name, distance_source


def vehicles_for(fleet):
    vehicles = []
    for kind, count, capacity, label in (
        ("bike", fleet.num_bikes, 15, "Bike"),
        ("petrol", fleet.num_petrol_trucks, 60, "Petrol van"),
        ("diesel", fleet.num_diesel_trucks, 80, "Diesel van"),
    ):
        for number in range(1, count + 1):
            vehicles.append(dict(vehicle_id=len(vehicles)+1, vehicle_type=kind, capacity_kg=capacity,
                                 vehicle_name=f"{label} {number}"))
    return vehicles


def leg_metrics(distance, payload, kind, fleet):
    result = compute_leg_fuel_and_emissions(distance, payload, fuel_type=kind)
    price = fleet.diesel_price_inr if kind == "diesel" else fleet.petrol_price_inr
    result["fuel_cost_inr"] = result["fuel_consumed"] * price
    return result


def route_metrics(route, vehicle, hubs, matrix, fleet, start_time=0, retained_load=0):
    payload = sum(hubs[n]["demand"] for n in route[1:] if n != 0) + retained_load
    distance = fuel = cost = co2 = late = 0.
    now = start_time
    schedule = []
    for u, v in zip(route, route[1:]):
        km = float(matrix[u, v])
        m = leg_metrics(km, payload, vehicle["vehicle_type"], fleet)
        distance += km
        fuel += m["fuel_consumed"]
        cost += m["fuel_cost_inr"]
        co2 += m["co2_emissions_kg"]
        now += km * 2  # 30 km/h; minutes from shift start
        if v:
            hub = hubs[v]
            arrival = max(now, hub["tw"][0])
            lateness = max(0, arrival - hub["tw"][1])
            late += lateness
            schedule.append(dict(**hub, customer_id=hub["id"], stop_number=len(schedule)+1,
                                 demand_kg=hub["demand"], arrival_time_min=round(arrival, 1),
                                 late_minutes=round(lateness, 1), status="pending", attempts=0))
            now = arrival + hub["service"]
            payload -= hub["demand"]
    return dict(distance_km=distance, fuel_liters=fuel, fuel_cost_inr=cost, co2_kg=co2,
                late_minutes=late, finish_time_min=now, schedule=schedule)


def score(route, vehicle, hubs, matrix, req):
    m = route_metrics(route, vehicle, hubs, matrix, req.fleet_config)
    return req.weight_dist*m["distance_km"]/100 + req.weight_fuel*m["fuel_liters"]/10 + req.weight_tw*m["late_minutes"]/60


def insertion_plan(hubs, matrix, vehicles, req, ordering=None):
    routes = [[0, 0] for _ in vehicles]
    loads = [0.] * len(vehicles)
    # Largest first avoids filling trucks with light parcels before heavy ones.
    pending = ordering or sorted(range(1, len(hubs)), key=lambda n: (-hubs[n]["demand"], hubs[n]["tw"][1], n))
    for node in pending:
        best = None
        for vi, vehicle in enumerate(vehicles):
            if loads[vi] + hubs[node]["demand"] > vehicle["capacity_kg"] + 1e-6:
                continue
            route = routes[vi]
            original = score(route, vehicle, hubs, matrix, req)
            positions = [len(route)-1] if req.algorithm == "nearest_neighbor" else range(1, len(route))
            for pos in positions:
                new_route = route[:pos] + [node] + route[pos:]
                candidate = (score(new_route, vehicle, hubs, matrix, req)-original, vi, pos)
                if best is None or candidate < best:
                    best = candidate
        if best is not None:
            _, vi, pos = best
            routes[vi].insert(pos, node)
            loads[vi] += hubs[node]["demand"]
    return routes


def ortools_plan(hubs, matrix, vehicles, req):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    manager = pywrapcp.RoutingIndexManager(len(hubs), len(vehicles), 0)
    routing = pywrapcp.RoutingModel(manager)
    callbacks = []
    for vi, vehicle in enumerate(vehicles):
        base = leg_metrics(1, vehicle["capacity_kg"] / 2, vehicle["vehicle_type"], req.fleet_config)["fuel_consumed"]
        def arc(a, b, rate=base):
            km = matrix[manager.IndexToNode(a), manager.IndexToNode(b)]
            return max(1, int(100000 * (req.weight_dist*km/100 + req.weight_fuel*km*rate/10)))
        callbacks.append(arc)
        routing.SetArcCostEvaluatorOfVehicle(routing.RegisterTransitCallback(arc), vi)
    demand_cb = routing.RegisterUnaryTransitCallback(lambda idx: round(hubs[manager.IndexToNode(idx)]["demand"]*1000))
    routing.AddDimensionWithVehicleCapacity(demand_cb, 0, [v["capacity_kg"]*1000 for v in vehicles], True, "Capacity")
    def travel(a, b):
        u, v = manager.IndexToNode(a), manager.IndexToNode(b)
        return math.ceil(matrix[u, v]*2 + hubs[u]["service"])
    callbacks.append(travel)
    routing.AddDimension(routing.RegisterTransitCallback(travel), 1440, 100000, True, "Time")
    times = routing.GetDimensionOrDie("Time")
    for node in range(1, len(hubs)):
        idx = manager.NodeToIndex(node)
        times.CumulVar(idx).SetMin(math.ceil(hubs[node]["tw"][0]))
        times.SetCumulVarSoftUpperBound(idx, int(hubs[node]["tw"][1]), round(req.weight_tw*100000/60))
        routing.AddDisjunction([idx], 100000000)  # Explicitly report unassigned orders.
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 2
    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("No routing solution found in the search budget")
    routes = []
    for vi in range(len(vehicles)):
        idx, route = routing.Start(vi), []
        while not routing.IsEnd(idx):
            route.append(manager.IndexToNode(idx))
            idx = solution.Value(routing.NextVar(idx))
        routes.append(route + [0])
    return routes


def plan(req, hubs, matrix):
    start = time.perf_counter()
    vehicles = vehicles_for(req.fleet_config)
    algorithm = req.algorithm
    warning = None
    if algorithm == "ortools":
        try:
            routes = ortools_plan(hubs, matrix, vehicles, req)
        except ImportError:
            routes = insertion_plan(hubs, matrix, vehicles, req)
            algorithm = "capacity_insertion"
            warning = "OR-Tools unavailable; used capacity-constrained insertion"
    elif algorithm in ("neural", "clarke_wright", "nearest_neighbor"):
        ordering = None
        if algorithm == "neural":
            from models.policy_solver import NeuralPolicySolver
            capacity = max(v["capacity_kg"] for v in vehicles)
            eligible = [0] + [i for i in range(1, len(hubs)) if hubs[i]["demand"] <= capacity]
            xy = np.array([[hubs[i]["lon"]*109, hubs[i]["lat"]*111] for i in eligible])
            instance = dict(coords=xy, demands=np.array([hubs[i]["demand"] for i in eligible]),
                            capacity=capacity, distance_matrix=matrix[np.ix_(eligible, eligible)])
            seed = NeuralPolicySolver(os.getenv("NEURAL_CHECKPOINT_PATH", str(ROOT / "saved_checkpoints/mo_cvrp_model.pt"))).solve(
                instance, preference_weights=(req.weight_dist, req.weight_fuel, req.weight_tw))
            ordering = [eligible[n] for route in seed["routes"] for n in route if n]
        elif algorithm == "clarke_wright":
            from baselines.clarke_wright import ClarkeWrightSolver
            capacity = max(v["capacity_kg"] for v in vehicles)
            eligible = [0] + [i for i in range(1, len(hubs)) if hubs[i]["demand"] <= capacity]
            seed = ClarkeWrightSolver().solve(dict(coords=np.zeros((len(eligible), 2)), demands=np.array([hubs[i]["demand"] for i in eligible]),
                capacity=capacity, distance_matrix=matrix[np.ix_(eligible, eligible)]))
            ordering = [eligible[n] for route in seed["routes"] for n in route if n]
        else:
            pending, ordering, current = set(range(1, len(hubs))), [], 0
            while pending:
                node = min(pending, key=lambda n: (matrix[current, n], n))
                ordering.append(node)
                pending.remove(node)
                current = node
        routes = insertion_plan(hubs, matrix, vehicles, req, ordering)
        if algorithm != "nearest_neighbor":
            warning = "Seed routes adapted by capacity-constrained insertion for the selected mixed fleet"
    else:
        raise ValueError("Unsupported planning algorithm")
    manifests, assigned = [], set()
    for route, vehicle in zip(routes, vehicles):
        if len(route) <= 2:
            continue
        customers = route[1:-1]
        if route[0] != 0 or route[-1] != 0 or any(n in assigned or n == 0 for n in customers) or len(set(customers)) != len(customers):
            raise RuntimeError("Solver produced an invalid or duplicate assignment")
        payload = sum(hubs[n]["demand"] for n in customers)
        if payload > vehicle["capacity_kg"] + 1e-6:
            raise RuntimeError("Solver produced a capacity violation")
        assigned.update(customers)
        metrics = route_metrics(route, vehicle, hubs, matrix, req.fleet_config)
        manifests.append(dict(**vehicle, **metrics, total_payload_kg=payload, remaining_payload_kg=payload,
                              num_stops=len(customers), route_stops=[hubs[n]["id"] for n in route],
                              current_customer_id=None, current_node_index=0, elapsed_minutes=0,
                              pending_ids=[hubs[n]["id"] for n in customers], color=COLORS[(vehicle["vehicle_id"]-1)%len(COLORS)]))
    unassigned = [dict(customer_id=h["id"], name=h["name"], demand_kg=h["demand"],
                       reason="Parcel exceeds every vehicle capacity" if h["demand"] > max(v["capacity_kg"] for v in vehicles)
                       else "Insufficient fleet capacity or search budget; add vehicles and plan again")
                  for i, h in enumerate(hubs) if i and i not in assigned]
    return dict(algorithm=algorithm, solve_time_sec=round(time.perf_counter()-start, 5), warning=warning,
                vehicle_manifests=manifests, unassigned_orders=unassigned,
                status="partial" if unassigned else "planned")


def decorate_dispatch(data):
    manifests = data["vehicle_manifests"]
    data["fleet_summary"] = dict(
        total_vehicles_used=len(manifests), total_distance_km=round(sum(v["distance_km"] for v in manifests), 2),
        total_fuel_liters=round(sum(v["fuel_liters"] for v in manifests), 3),
        total_cost_inr=round(sum(v["fuel_cost_inr"] for v in manifests), 2),
        total_co2_kg=round(sum(v["co2_kg"] for v in manifests), 2),
        assigned_orders=sum(v["num_stops"] for v in manifests),
        delivered_orders=sum(s["status"] == "delivered" for v in manifests for s in v["schedule"]),
        failed_orders=sum(s["status"] == "failed" for v in manifests for s in v["schedule"]),
        unassigned_orders=len(data["unassigned_orders"]))
    hub_map = {h["id"]: h for h in data["hubs"]}
    data["polylines"] = [dict(route_id=v["vehicle_id"], color=v["color"], stops=v["route_stops"],
                              polyline=[[hub_map[n]["lat"], hub_map[n]["lon"]] for n in v["route_stops"]],
                              source="Stop connections · illustrative") for v in manifests]
    return data
