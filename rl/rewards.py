"""
rl/rewards.py
Multi-Objective Reward & Cost Evaluation for MO-CVRP and MO-CVRPTW.

Supported Objectives:
1. Total Travel Distance (Euclidean or Road km)
2. Fuel Consumption & CO2 Emissions (Load-dependent CMEM model for Diesel, Petrol, and EV)
3. Soft Time Window Penalty (Early wait time and Late delivery penalties)
4. Overall Fuel Cost (INR ₹)
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class FuelAndEmissionConfig:
    """
    Fleet fuel, energy, and emissions parameters based on the
    Comprehensive Modal Emission Model (CMEM) / heavy-duty, light-duty, and 2-wheelers.
    """
    VEHICLE_CURB_WEIGHT = 1500.0  # Empty vehicle mass in kg (e.g. delivery van)

    # 2-Wheeler (Electric & Petrol Delivery Scooters)
    BIKE_CAPACITY = 15.0          # Max 15 kg parcel payload
    BIKE_CURB_WEIGHT = 110.0      # Scooter weight in kg
    BIKE_COST_PER_KM_INR = 1.80   # Very low operating cost (₹1.80/km)
    BIKE_CO2_PER_KM = 0.045       # 0.045 kg CO2/km

    # EV Van
    EV_VAN_CAPACITY = 40.0        # Max 40 kg payload
    EV_BASE_RATE_KWH = 0.18       # ~18 kWh/100km empty
    EV_LOAD_COEFF_KWH = 0.000090  # Additional kWh/km per kg
    EV_CO2_PER_KWH = 0.71         # kg CO2 per kWh
    EV_PRICE_INR_PER_KWH = 8.00   # INR ₹ per kWh

    # Diesel Van / Truck
    DIESEL_CAPACITY = 80.0        # Max 80 kg payload
    DIESEL_BASE_RATE = 0.085      # ~8.5 L/100km empty
    DIESEL_LOAD_COEFF = 0.000045  # Additional L/km per kg of cargo
    DIESEL_CO2_PER_LITER = 2.68   # kg CO2 per liter of diesel
    DIESEL_PRICE_INR = 92.50      # INR ₹ per liter

    # Petrol Van
    PETROL_CAPACITY = 60.0
    PETROL_BASE_RATE = 0.095      # ~9.5 L/100km empty
    PETROL_LOAD_COEFF = 0.000050  # Additional L/km per kg
    PETROL_CO2_PER_LITER = 2.31   # kg CO2 per liter of petrol
    PETROL_PRICE_INR = 102.80     # INR ₹ per liter

    # Time window penalty coefficients
    LATE_PENALTY_PER_TIME_UNIT = 2.0   # Penalty per minute late
    EARLY_WAIT_PENALTY_PER_UNIT = 0.5  # Idle vehicle cost per unit early wait


def compute_leg_fuel_and_emissions(
    distance: float,
    current_payload: float,
    fuel_type: str = "diesel",
    curb_weight: float = FuelAndEmissionConfig.VEHICLE_CURB_WEIGHT
) -> Dict[str, float]:
    """
    Computes fuel/energy consumed, CO2 emissions, and fuel cost for a single travel leg
    given the instantaneous payload on that leg.

    Physics: As the vehicle delivers cargo along the route, payload decreases,
    making subsequent legs lighter and more fuel efficient!
    """
    fuel_type = fuel_type.lower()

    if fuel_type == "bike":
        fuel_consumed = distance * (1 / 45.0 + 0.00005 * current_payload)
        co2_emissions = fuel_consumed * FuelAndEmissionConfig.PETROL_CO2_PER_LITER
        fuel_cost_inr = fuel_consumed * FuelAndEmissionConfig.PETROL_PRICE_INR
        unit = "Liters"

    elif fuel_type == "diesel":
        fuel_consumed = distance * (
            FuelAndEmissionConfig.DIESEL_BASE_RATE +
            FuelAndEmissionConfig.DIESEL_LOAD_COEFF * current_payload
        )
        co2_emissions = fuel_consumed * FuelAndEmissionConfig.DIESEL_CO2_PER_LITER
        fuel_cost_inr = fuel_consumed * FuelAndEmissionConfig.DIESEL_PRICE_INR
        unit = "Liters"

    elif fuel_type == "petrol":
        fuel_consumed = distance * (
            FuelAndEmissionConfig.PETROL_BASE_RATE +
            FuelAndEmissionConfig.PETROL_LOAD_COEFF * current_payload
        )
        co2_emissions = fuel_consumed * FuelAndEmissionConfig.PETROL_CO2_PER_LITER
        fuel_cost_inr = fuel_consumed * FuelAndEmissionConfig.PETROL_PRICE_INR
        unit = "Liters"

    elif fuel_type == "ev":
        fuel_consumed = distance * (
            FuelAndEmissionConfig.EV_BASE_RATE_KWH +
            FuelAndEmissionConfig.EV_LOAD_COEFF_KWH * current_payload
        )
        co2_emissions = fuel_consumed * FuelAndEmissionConfig.EV_CO2_PER_KWH
        fuel_cost_inr = fuel_consumed * FuelAndEmissionConfig.EV_PRICE_INR_PER_KWH
        unit = "kWh"

    else:
        # Default mixed fallback
        fuel_consumed = distance * 0.08
        co2_emissions = distance * 0.15
        fuel_cost_inr = distance * 4.5
        unit = "Mixed Units"

    return {
        "fuel_type": fuel_type,
        "fuel_consumed": fuel_consumed,
        "unit": unit,
        "co2_emissions_kg": co2_emissions,
        "fuel_cost_inr": fuel_cost_inr
    }


def evaluate_solution_metrics(
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    demands: np.ndarray,
    time_windows: Optional[np.ndarray] = None,
    service_times: Optional[np.ndarray] = None,
    speed: float = 30.0, # km/h or units/time
    fuel_type: str = "diesel"
) -> Dict[str, Any]:
    """
    Evaluates complete multi-objective metrics for a set of vehicle routes.

    routes: List of routes, each route is a list of node indices starting at 0 (depot)
            e.g., [[0, 3, 5, 2, 0], [0, 1, 4, 0]]
    """
    total_distance = 0.0
    total_fuel = 0.0
    total_co2 = 0.0
    total_fuel_cost = 0.0
    total_tw_late_penalty = 0.0
    total_tw_early_penalty = 0.0
    total_time_violations = 0
    route_details = []

    has_tw = time_windows is not None and len(time_windows) > 0

    for r_idx, route in enumerate(routes):
        if len(route) <= 2:
            continue # empty or depot-only route

        # Calculate initial payload for this route (sum of demands of all visited customers)
        route_customers = [node for node in route if node != 0]
        initial_payload = float(sum(demands[c] for c in route_customers))
        current_payload = initial_payload

        route_dist = 0.0
        route_fuel = 0.0
        route_co2 = 0.0
        route_cost = 0.0
        route_late_pen = 0.0
        route_early_pen = 0.0

        current_time = 0.0

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            leg_dist = float(distance_matrix[u][v])
            route_dist += leg_dist

            # Calculate fuel & emissions for this leg with current payload
            leg_metrics = compute_leg_fuel_and_emissions(leg_dist, current_payload, fuel_type=fuel_type)
            route_fuel += leg_metrics["fuel_consumed"]
            route_co2 += leg_metrics["co2_emissions_kg"]
            route_cost += leg_metrics["fuel_cost_inr"]

            # Drop off demand at node v
            if v != 0:
                current_payload = max(0.0, current_payload - float(demands[v]))

            # Time window tracking
            if has_tw:
                travel_time = leg_dist / speed * 60 if speed > 0 else 0
                arrival_time = current_time + travel_time

                if v != 0:
                    ready_time = time_windows[v][0]
                    due_time = time_windows[v][1]
                    s_time = service_times[v] if service_times is not None else 0.0

                    if arrival_time < ready_time:
                        # Early arrival -> wait until ready_time
                        wait_time = ready_time - arrival_time
                        route_early_pen += wait_time * FuelAndEmissionConfig.EARLY_WAIT_PENALTY_PER_UNIT
                        current_time = ready_time + s_time
                    elif arrival_time > due_time:
                        # Late arrival -> penalty
                        late_time = arrival_time - due_time
                        route_late_pen += late_time * FuelAndEmissionConfig.LATE_PENALTY_PER_TIME_UNIT
                        total_time_violations += 1
                        current_time = arrival_time + s_time
                    else:
                        current_time = arrival_time + s_time
                else:
                    current_time = arrival_time

        total_distance += route_dist
        total_fuel += route_fuel
        total_co2 += route_co2
        total_fuel_cost += route_cost
        total_tw_late_penalty += route_late_pen
        total_tw_early_penalty += route_early_pen

        route_details.append({
            "route_id": r_idx + 1,
            "stops": route,
            "distance": round(route_dist, 2),
            "fuel_consumed": round(route_fuel, 3),
            "co2_kg": round(route_co2, 3),
            "cost_inr": round(route_cost, 2),
            "late_penalty": round(route_late_pen, 2),
            "early_wait_penalty": round(route_early_pen, 2)
        })

    tw_total_penalty = total_tw_late_penalty + total_tw_early_penalty

    return {
        "num_routes": len(route_details),
        "total_distance": round(total_distance, 2),
        "total_fuel_consumed": round(total_fuel, 3),
        "fuel_unit": "kWh" if fuel_type == "ev" else "Liters",
        "total_co2_kg": round(total_co2, 3),
        "total_fuel_cost_inr": round(total_fuel_cost, 2),
        "total_tw_penalty": round(tw_total_penalty, 2),
        "late_penalty": round(total_tw_late_penalty, 2),
        "early_wait_penalty": round(total_tw_early_penalty, 2),
        "time_window_violations": total_time_violations,
        "route_details": route_details,
        "fuel_type": fuel_type
    }


def compute_scalarized_objective(
    metrics: Dict[str, Any],
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    norm_dist: float = 100.0,
    norm_co2: float = 25.0,
    norm_tw: float = 50.0
) -> float:
    """
    Computes weighted scalarized objective value (minimization):
    J = w1 * (dist / norm_dist) + w2 * (co2 / norm_co2) + w3 * (tw_pen / norm_tw)
    """
    w1, w2, w3 = weights
    w_sum = w1 + w2 + w3
    if w_sum > 0:
        w1, w2, w3 = w1 / w_sum, w2 / w_sum, w3 / w_sum

    d_norm = metrics["total_distance"] / max(1e-5, norm_dist)
    c_norm = metrics["total_co2_kg"] / max(1e-5, norm_co2)
    tw_norm = metrics["total_tw_penalty"] / max(1e-5, norm_tw)

    return float(w1 * d_norm + w2 * c_norm + w3 * tw_norm)
