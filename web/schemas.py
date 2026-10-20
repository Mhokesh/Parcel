"""Validated contracts shared by the dispatcher and mobile client."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Hub(StrictModel):
    id: int = Field(ge=0)
    name: str = Field(min_length=1, max_length=160)
    lat: float = Field(ge=-85, le=85)
    lon: float = Field(ge=-180, le=180)
    demand: float = Field(default=0, ge=0, le=10000)
    tw: tuple[float, float] = (0, 480)
    service: float = Field(default=5, ge=0, le=120)
    type: str = Field(default="customer", max_length=40)
    customer_name: str = Field(default="", max_length=160)
    address: str = Field(default="", max_length=500)
    phone: str = Field(default="", max_length=40)
    product: str = Field(default="Parcel", max_length=200)
    size: str = Field(default="", max_length=40)

    @model_validator(mode="after")
    def valid_window(self):
        if not 0 <= self.tw[0] <= self.tw[1] <= 1440:
            raise ValueError("Time windows must be ordered minutes from shift start, within 24 hours")
        return self


class FleetConfig(StrictModel):
    num_bikes: int = Field(default=2, ge=0, le=30)
    num_petrol_trucks: int = Field(default=1, ge=0, le=30)
    num_diesel_trucks: int = Field(default=8, ge=0, le=30)
    # Accepted for older clients; new clients send separate petrol/diesel counts.
    num_mini_trucks: int | None = Field(default=None, ge=0, le=30)
    truck_fuel_type: Literal["diesel", "petrol"] = "diesel"
    petrol_price_inr: float = Field(default=102.8, gt=0, le=500)
    diesel_price_inr: float = Field(default=92.5, gt=0, le=500)

    @model_validator(mode="after")
    def available_fleet(self):
        if self.num_mini_trucks is not None:
            self.num_petrol_trucks = self.num_mini_trucks if self.truck_fuel_type == "petrol" else 0
            self.num_diesel_trucks = self.num_mini_trucks if self.truck_fuel_type == "diesel" else 0
        if not 1 <= self.num_bikes + self.num_petrol_trucks + self.num_diesel_trucks <= 40:
            raise ValueError("Choose between 1 and 40 vehicles")
        return self


class CustomInstanceData(StrictModel):
    name: str = Field(default="Imported orders", min_length=1, max_length=160)
    capacity: int = Field(default=80, ge=1, le=10000)
    hubs: list[Hub] = Field(min_length=2, max_length=201)

    @model_validator(mode="after")
    def valid_depot(self):
        ids = [h.id for h in self.hubs]
        if len(ids) != len(set(ids)) or ids[0] != 0 or self.hubs[0].demand != 0:
            raise ValueError("Use unique order IDs and a first row depot with id 0 and demand 0")
        if any(h.demand <= 0 for h in self.hubs[1:]):
            raise ValueError("Customer weights must be positive")
        return self


class SolveRequest(StrictModel):
    dataset_type: Literal["coimbatore", "synthetic", "custom", "cvrplib_a32", "cvrplib_a45"] = "coimbatore"
    num_customers: int = Field(default=50, ge=1, le=200)
    vehicle_capacity: int = Field(default=80, ge=1, le=10000)
    fuel_type: Literal["mixed", "diesel", "petrol", "bike"] = "mixed"
    weight_dist: float = Field(default=.5, ge=0, le=1)
    weight_fuel: float = Field(default=.3, ge=0, le=1)
    weight_tw: float = Field(default=.2, ge=0, le=1)
    algorithm: Literal["all", "ortools", "neural", "clarke_wright", "nearest_neighbor"] = "ortools"
    fleet_config: FleetConfig = Field(default_factory=FleetConfig)
    custom_data: CustomInstanceData | None = None

    @model_validator(mode="after")
    def valid_problem(self):
        if self.weight_dist + self.weight_fuel + self.weight_tw <= 0:
            raise ValueError("At least one objective weight must be positive")
        if self.dataset_type == "custom" and self.custom_data is None:
            raise ValueError("Custom orders are required")
        return self


class DeliveryEvent(StrictModel):
    event_id: str = Field(min_length=8, max_length=100)
    expected_version: int = Field(ge=1)
    vehicle_id: int = Field(ge=1)
    customer_id: int = Field(ge=1)
    action: Literal["delivered", "retry", "failed"]


class CSVImport(StrictModel):
    content: str = Field(min_length=1, max_length=500000)
    name: str = Field(default="Imported orders", min_length=1, max_length=160)
