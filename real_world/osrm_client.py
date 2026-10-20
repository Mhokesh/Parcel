"""
real_world/osrm_client.py
OpenStreetMap / OSRM API Client & Offline Fallback for Road Geometries & Routing.
"""

import requests
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def get_route_geometry(
    origin_gps: Tuple[float, float],
    dest_gps: Tuple[float, float],
    use_online_osrm: bool = False
) -> List[Tuple[float, float]]:
    """
    Returns a sequence of (lat, lon) coordinates representing the path between two points.
    If online OSRM is enabled and accessible, fetches real road geometry.
    Otherwise, generates interpolated points along the geodesic.
    """
    lat1, lon1 = origin_gps
    lat2, lon2 = dest_gps

    if use_online_osrm:
        try:
            # Public OSRM demo server
            url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
            resp = requests.get(url, timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                coords = data["routes"][0]["geometry"]["coordinates"]
                # Convert [lon, lat] -> [lat, lon]
                return [[c[1], c[0]] for c in coords]
        except Exception:
            pass # fallback to interpolation

    # High-quality Bezier-like interpolated road path fallback
    steps = 10
    lats = np.linspace(lat1, lat2, steps)
    lons = np.linspace(lon1, lon2, steps)

    # Slight organic road curve
    mid_lat = (lat1 + lat2) / 2.0 + (lon2 - lon1) * 0.05
    mid_lon = (lon1 + lon2) / 2.0 - (lat2 - lat1) * 0.05

    # Quadratic curve
    t = np.linspace(0, 1, steps)
    interp_lats = (1 - t)**2 * lat1 + 2 * (1 - t) * t * mid_lat + t**2 * lat2
    interp_lons = (1 - t)**2 * lon1 + 2 * (1 - t) * t * mid_lon + t**2 * lon2

    return [[float(la), float(lo)] for la, lo in zip(interp_lats, interp_lons)]


def build_full_route_polylines(
    routes: List[List[int]],
    hubs: List[Dict[str, Any]],
    use_online_osrm: bool = False
) -> List[Dict[str, Any]]:
    """
    Builds full Leaflet polyline paths and stop information for all routes in a solution.
    """
    route_polylines = []
    colors = ["#4361ee", "#2ec4b6", "#e71d36", "#ff9f1c", "#7209b7", "#3a86ff", "#06d6a0"]

    for r_idx, route in enumerate(routes):
        path_coords = []
        stops_info = []

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            u_gps = (hubs[u]["lat"], hubs[u]["lon"])
            v_gps = (hubs[v]["lat"], hubs[v]["lon"])

            leg_coords = get_route_geometry(u_gps, v_gps, use_online_osrm=use_online_osrm)
            path_coords.extend(leg_coords)

            stops_info.append({
                "node_id": u,
                "name": hubs[u]["name"],
                "lat": hubs[u]["lat"],
                "lon": hubs[u]["lon"],
                "demand": hubs[u]["demand"],
                "tw": hubs[u]["tw"]
            })

        # Append last depot stop
        last_node = route[-1]
        stops_info.append({
            "node_id": last_node,
            "name": hubs[last_node]["name"],
            "lat": hubs[last_node]["lat"],
            "lon": hubs[last_node]["lon"],
            "demand": hubs[last_node]["demand"],
            "tw": hubs[last_node]["tw"]
        })

        route_polylines.append({
            "route_id": r_idx + 1,
            "color": colors[r_idx % len(colors)],
            "stops": route,
            "stops_info": stops_info,
            "polyline": path_coords
        })

    return route_polylines
