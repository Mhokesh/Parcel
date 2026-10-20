import copy
import uuid
import pytest
from fastapi.testclient import TestClient
from web import app as api_module
from web.schemas import SolveRequest
from web.planner import load_problem, plan
from web.store import DispatchStore


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "store", DispatchStore(tmp_path / "dispatch.db"))
    monkeypatch.setattr(api_module, "API_KEY", "")
    return TestClient(api_module.app)


def custom_request(weights=(4, 7, 12), ids=(101, 205, 309)):
    hubs = [dict(id=0, name="Depot", lat=11.0168, lon=76.9558, demand=0, service=0)]
    hubs += [dict(id=cid, name=f"Order {cid}", lat=11.02+i*.01, lon=76.96+i*.01, demand=w, tw=[0, 480], service=5) for i, (cid,w) in enumerate(zip(ids,weights))]
    return dict(dataset_type="custom", custom_data=dict(name="Test", hubs=hubs), algorithm="nearest_neighbor", fleet_config=dict(num_bikes=0, num_petrol_trucks=0, num_diesel_trucks=1))


def create(client, payload=None):
    response = client.post("/api/solve", json=payload or custom_request())
    assert response.status_code == 200, response.text
    return response.json()


def event(data, action="delivered", cid=None, event_id=None):
    v = data["vehicle_manifests"][0]
    return dict(event_id=event_id or uuid.uuid4().hex, expected_version=data["version"], vehicle_id=v["vehicle_id"], customer_id=cid or v["pending_ids"][0], action=action)


def test_default_sample_is_fifty_real_rows(client):
    sample = client.get("/api/sample-50-orders").json()
    assert sample["total_orders"] == 50
    data = create(client, dict(algorithm="ortools"))
    assert data["fleet_summary"]["assigned_orders"] == 50
    assigned = [s["customer_id"] for v in data["vehicle_manifests"] for s in v["schedule"]]
    assert len(assigned) == len(set(assigned)) == 50
    assert all(v["total_payload_kg"] <= v["capacity_kg"] for v in data["vehicle_manifests"])


@pytest.mark.parametrize("algorithm", ["ortools", "nearest_neighbor", "clarke_wright"])
def test_insufficient_fleet_never_invents_vehicles(client, algorithm):
    payload = custom_request((20, 20, 100))
    payload["algorithm"] = algorithm
    payload["fleet_config"] = dict(num_bikes=1, num_petrol_trucks=0, num_diesel_trucks=0)
    data = create(client, payload)
    assert not data["vehicle_manifests"]
    assert len(data["unassigned_orders"]) == 3
    assert data["status"] == "partial"


def test_partition_and_price_isolation(client):
    p = custom_request((10, 10, 10))
    p["fleet_config"] = dict(num_bikes=1, num_petrol_trucks=0, num_diesel_trucks=0, petrol_price_inr=100)
    low = create(client, p)
    assert len(low["vehicle_manifests"]) == 1
    assert low["fleet_summary"]["assigned_orders"] == 1
    assert len(low["unassigned_orders"]) == 2
    p["fleet_config"]["petrol_price_inr"] = 200
    high = create(client, p)
    assert high["fleet_summary"]["total_cost_inr"] == pytest.approx(low["fleet_summary"]["total_cost_inr"]*2, abs=.02)
    assert client.get(f"/api/dispatches/{low['dispatch_id']}").json()["fleet_summary"] == low["fleet_summary"]


def test_delivery_is_persistent_idempotent_and_conflict_safe(client):
    data = create(client)
    path = f"/api/dispatches/{data['dispatch_id']}/events"
    action = event(data)
    response = client.post(path, json=action)
    assert response.status_code == 200
    updated = response.json()
    assert updated["version"] == 2
    assert updated["fleet_summary"]["delivered_orders"] == 1
    assert client.post(path, json=action).json() == updated
    assert client.get(f"/api/dispatches/{data['dispatch_id']}").json() == updated
    stale = event(data)
    assert client.post(path, json=stale).status_code == 409
    action["action"] = "failed"
    assert client.post(path, json=action).status_code == 409


def test_retry_starts_at_current_location_and_preserves_cargo(client):
    data = create(client)
    v = data["vehicle_manifests"][0]
    failed = v["pending_ids"][0]
    path = f"/api/dispatches/{data['dispatch_id']}/events"
    updated = client.post(path, json=event(data, "retry")).json()
    after = updated["vehicle_manifests"][0]
    assert after["remaining_payload_kg"] == v["total_payload_kg"]
    assert after["pending_ids"][-1] == failed
    assert after["remaining_route_stops"][0] == failed
    assert after["current_customer_id"] == failed
    assert len(after["pending_ids"]) == len(set(after["pending_ids"]))
    while len(after["pending_ids"]) > 1:
        updated = client.post(path, json=event(updated)).json()
        after = updated["vehicle_manifests"][0]
    assert client.post(path, json=event(updated, "retry")).status_code == 422
    final = client.post(path, json=event(updated, "failed")).json()
    assert final["fleet_summary"]["failed_orders"] == 1
    assert final["vehicle_manifests"][0]["remaining_payload_kg"] > 0


def test_out_of_order_and_unknown_dispatch(client):
    data = create(client)
    assert client.post(f"/api/dispatches/{data['dispatch_id']}/events", json=event(data, cid=309)).status_code == 422
    assert client.get("/api/dispatches/unknown").status_code == 404


@pytest.mark.parametrize("patch", [dict(algorithm="invented"), dict(weight_dist=0, weight_fuel=0, weight_tw=0), dict(num_customers=999), dict(weight_fuel=float('inf'))])
def test_invalid_input(client, patch):
    p = custom_request(); p.update(patch)
    if patch.get("weight_fuel") == float('inf'):
        assert client.post('/api/solve', content='{"weight_fuel":Infinity}', headers={"Content-Type":"application/json"}).status_code == 422
    else:
        assert client.post("/api/solve", json=p).status_code == 422


def test_csv_validation_and_non_contiguous_ids(client):
    content = 'id,name,lat,lon,weight_kg\n0,Depot,11,77,0\n50,"A, B",11.01,77.01,4\n'
    result = client.post("/api/import-csv", json=dict(content=content, name="CSV"))
    assert result.status_code == 200
    p = custom_request(); p["custom_data"] = result.json()
    data = create(client, p)
    assert data["vehicle_manifests"][0]["pending_ids"] == [50]
    assert client.post("/api/import-csv", json=dict(content=content.replace('11.01','911'))).status_code == 422
    assert client.post("/api/import-csv", json=dict(content=content.replace('50,','0,'))).status_code == 422


def test_auth_and_payload_limit(client, monkeypatch):
    monkeypatch.setattr(api_module, "API_KEY", "secret")
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/dispatches").status_code == 401
    assert client.get("/api/dispatches", headers={"X-API-Key":"secret"}).status_code == 200
    assert client.post('/api/import-csv', content='x'*600001).status_code == 413


def test_fuel_endpoint_does_not_claim_live(client):
    data = client.get('/api/fuel-prices/live').json()
    assert data['last_updated'] is None
    assert 'Manual' in data['status']


def test_drive_time_is_minutes():
    import numpy as np
    from rl.rewards import evaluate_solution_metrics
    m = evaluate_solution_metrics([[0,1,0]], np.array([[0,30],[30,0]]), np.array([0,1]), np.array([[0,480],[0,30]]), np.array([0,0]), speed=30)
    assert m['time_window_violations'] == 1
    assert m['late_penalty'] == 60
