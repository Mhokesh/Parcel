"""Persistent dispatch API. Blocking planners run in FastAPI's thread pool."""
import logging
import os
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from web.planner import decorate_dispatch, leg_metrics, load_problem, parse_csv, plan, sample_data
from web.schemas import CSVImport, DeliveryEvent, SolveRequest
from web.store import Conflict, DispatchStore

log = logging.getLogger("dispatch")
ROOT = Path(__file__).resolve().parents[1]
API_KEY = os.getenv("DISPATCH_API_KEY", "")
if os.getenv("APP_ENV") == "production" and len(API_KEY) < 24:
    raise RuntimeError("Production requires DISPATCH_API_KEY with at least 24 characters")
app = FastAPI(title="Parcel · Dispatch API", version="2.0.0", docs_url=None if os.getenv("APP_ENV") == "production" else "/docs")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8081,http://localhost:19006").split(","), allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-API-Key"])
app.mount("/static", StaticFiles(directory=ROOT / "web/static"), name="static")
store = DispatchStore()
solve_slots = threading.BoundedSemaphore(2)


def authorize(request: Request):
    if API_KEY and not secrets.compare_digest(request.headers.get("X-API-Key", ""), API_KEY):
        raise HTTPException(401, "Enter the server access key in Settings")


@app.middleware("http")
async def request_limits(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    if request.method == "POST":
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 600000:
                return JSONResponse(status_code=413, content={"detail": "Request exceeds 600 KB"})
        request._body = bytes(body)
    response = await call_next(request)
    response.headers.update({"X-Request-ID": request_id, "X-Content-Type-Options": "nosniff", "Referrer-Policy": "strict-origin-when-cross-origin", "X-Frame-Options": "DENY"})
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    log.info("request id=%s method=%s path=%s status=%s ms=%.1f", request_id, request.method, request.url.path, response.status_code, (time.perf_counter()-started)*1000)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    return JSONResponse(status_code=422, content={"detail": "; ".join(f"{'.'.join(map(str,e['loc']))}: {e['msg']}" for e in exc.errors())})


@app.get("/")
def index():
    return FileResponse(ROOT / "web/static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "2.0.0", "authentication_required": bool(API_KEY)}


@app.get("/api/map-config")
def map_config():
    return {"tile_url": os.getenv("MAP_TILE_URL", "https://tile.openstreetmap.org/{z}/{x}/{y}.png"),
            "attribution": os.getenv("MAP_ATTRIBUTION", '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors')}


@app.get("/api/sample-50-orders", dependencies=[Depends(authorize)])
def sample():
    data = sample_data()
    return {"total_orders": len(data.hubs)-1, "hub_name": data.hubs[0].name, "orders": [h.model_dump() for h in data.hubs]}


@app.post("/api/import-csv", dependencies=[Depends(authorize)])
def import_csv(req: CSVImport):
    try:
        return parse_csv(req.content, req.name).model_dump()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/sample.csv", dependencies=[Depends(authorize)])
def sample_csv():
    return FileResponse(ROOT / "data/sample_50_orders.csv", media_type="text/csv", filename="coimbatore-orders.csv")


@app.get("/api/fuel-prices/live", dependencies=[Depends(authorize)])
def fuel_prices():
    return dict(city="Coimbatore, Tamil Nadu", petrol_price_inr=102.8, diesel_price_inr=92.5, currency="INR", last_updated=None, status="Manual reference rates", source="User-editable planning assumptions; not a live feed")


def public(data):
    return {k: v for k, v in data.items() if k not in ("distance_matrix", "request")}


@app.post("/api/solve", dependencies=[Depends(authorize)])
def solve(req: SolveRequest):
    if not solve_slots.acquire(blocking=False):
        raise HTTPException(429, "The planner is busy. Try again shortly.")
    try:
        hubs, matrix, name, source = load_problem(req)
        comparisons, primary = {}, None
        for algorithm in (["ortools", "clarke_wright", "nearest_neighbor", "neural"] if req.algorithm == "all" else [req.algorithm]):
            try:
                result = plan(req.model_copy(update={"algorithm": algorithm}), hubs, matrix)
                comparisons[algorithm] = dict(algorithm=result["algorithm"], solve_time_sec=result["solve_time_sec"], distance_km=round(sum(v["distance_km"] for v in result["vehicle_manifests"]), 2), unassigned=len(result["unassigned_orders"]), warning=result["warning"])
                if primary is None:
                    primary = result
            except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
                if req.algorithm != "all":
                    raise HTTPException(503, str(exc)) from exc
                comparisons[algorithm] = {"error": str(exc)}
        if primary is None:
            raise HTTPException(503, "No solver is available")
        data = decorate_dispatch(dict(**primary, dispatch_id=uuid.uuid4().hex, version=1, created_at=datetime.now(timezone.utc).isoformat(), dataset_name=name, num_customers=len(hubs)-1, hubs=hubs, distance_source=source, geometry_source="Stop connections · illustrative", fuel_source="User-entered rates · load-dependent estimates", results=comparisons, request=req.model_dump(), distance_matrix=matrix.tolist()))
        store.create(data)
        return public(data)
    finally:
        solve_slots.release()


@app.get("/api/dispatches", dependencies=[Depends(authorize)])
def recent():
    return [{"dispatch_id": d["dispatch_id"], "dataset_name": d["dataset_name"], "created_at": d["created_at"], "fleet_summary": d["fleet_summary"]} for d in store.recent()]


@app.get("/api/dispatches/{dispatch_id}", dependencies=[Depends(authorize)])
def get_dispatch(dispatch_id: str):
    data = store.get(dispatch_id)
    if not data:
        raise HTTPException(404, "Dispatch not found")
    return public(data)


def mutate_delivery(data, event):
    vehicle = next((v for v in data["vehicle_manifests"] if v["vehicle_id"] == event.vehicle_id), None)
    if vehicle is None:
        raise ValueError("Vehicle does not belong to this dispatch")
    if not vehicle["pending_ids"] or vehicle["pending_ids"][0] != event.customer_id:
        raise ValueError("Only the next pending stop can be updated")
    stop = next(s for s in vehicle["schedule"] if s["customer_id"] == event.customer_id)
    if event.action == "retry" and stop["attempts"] >= 1:
        raise ValueError("This stop has already been retried. Mark delivered or return the parcel as failed.")
    hub_by_id = {h["id"]: h for h in data["hubs"]}
    current, target = vehicle["current_node_index"], stop["node_index"]
    km = data["distance_matrix"][current][target]
    fleet = SolveRequest.model_validate(data["request"]).fleet_config
    actual = leg_metrics(km, vehicle["remaining_payload_kg"], vehicle["vehicle_type"], fleet)
    vehicle["travelled_distance_km"] = vehicle.get("travelled_distance_km", 0) + km
    vehicle["estimated_fuel_used_liters"] = vehicle.get("estimated_fuel_used_liters", 0) + actual["fuel_consumed"]
    vehicle["elapsed_minutes"] = max(vehicle["elapsed_minutes"] + km*2, stop["tw"][0]) + stop["service"]
    vehicle["current_node_index"], vehicle["current_customer_id"] = target, event.customer_id
    stop["attempts"] += 1
    stop["updated_at"] = datetime.now(timezone.utc).isoformat()
    vehicle["pending_ids"].pop(0)
    if event.action == "delivered":
        stop["status"] = "delivered"
        vehicle["remaining_payload_kg"] = max(0, vehicle["remaining_payload_kg"]-stop["demand_kg"])
    elif event.action == "failed":
        stop["status"] = "failed"
    else:
        stop["status"] = "retry"
        pending, ordered, at = list(vehicle["pending_ids"]), [], target
        while pending:
            chosen = min(pending, key=lambda n: (data["distance_matrix"][at][hub_by_id[n]["node_index"]], n))
            ordered.append(chosen)
            pending.remove(chosen)
            at = hub_by_id[chosen]["node_index"]
        vehicle["pending_ids"] = ordered + [event.customer_id]
    vehicle["remaining_route_stops"] = [event.customer_id] + vehicle["pending_ids"] + [0]
    now, at = vehicle["elapsed_minutes"], target
    for cid in vehicle["pending_ids"]:
        next_stop = next(s for s in vehicle["schedule"] if s["customer_id"] == cid)
        now = max(now + data["distance_matrix"][at][next_stop["node_index"]]*2, next_stop["tw"][0])
        next_stop["arrival_time_min"] = round(now, 1)
        next_stop["late_minutes"] = round(max(0, now-next_stop["tw"][1]), 1)
        now += next_stop["service"]
        at = next_stop["node_index"]
    data["updated_at"] = stop["updated_at"]
    data["status"] = "completed" if all(not v["pending_ids"] for v in data["vehicle_manifests"]) else "in_progress"
    decorate_dispatch(data)
    for poly, v in zip(data["polylines"], data["vehicle_manifests"]):
        if "remaining_route_stops" in v:
            poly["polyline"] = [[hub_by_id[n]["lat"], hub_by_id[n]["lon"]] for n in v["remaining_route_stops"]]


@app.post("/api/dispatches/{dispatch_id}/events", dependencies=[Depends(authorize)])
def delivery(dispatch_id: str, event: DeliveryEvent):
    try:
        return public(store.apply_event(dispatch_id, event, mutate_delivery))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Conflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
