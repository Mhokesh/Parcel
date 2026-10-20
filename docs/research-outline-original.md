> **Historical research outline.** This file preserves the original project proposal. Its benchmark numbers, live-price claims, full-CMEM description, and production-readiness implications are unverified and do not describe the current release. Use the repository [README](../README.md) and [implementation review](implementation-review.md) for the implemented behaviour and limitations.

# 🚚 Multi-Objective Deep Reinforcement Learning for Heterogeneous Last-Mile Logistics (MO-CVRP)

An end-to-end Operations Research (OR) and Deep Reinforcement Learning (DRL) framework for solving the **Multi-Objective Capacitated Vehicle Routing Problem (MO-CVRP)** across urban road networks. Featuring sub-second neural route optimization, physics-based load-dependent fuel consumption modeling (CMEM), Indian heterogeneous fleet allotment (2-Wheelers & 4-Wheeler Mini-Trucks), automated live regional fuel tickers, and interactive Google-Maps-style turn-by-turn satellite GPS navigation.

---

## 📑 Table of Contents
1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [Operations Research (OR) Foundations & Formulations](#2-operations-research-or-foundations--formulations)
3. [Deep Reinforcement Learning (DRL) Architecture](#3-deep-reinforcement-learning-drl-architecture)
4. [Physics-Based Fuel Consumption & Emissions Model (CMEM)](#4-physics-based-fuel-consumption--emissions-model-cmem)
5. [Heterogeneous Fleet Logistics (Indian Urban Context)](#5-heterogeneous-fleet-logistics-indian-urban-context)
6. [Real-World Urban Network: Coimbatore Case Study](#6-real-world-urban-network-coimbatore-case-study)
7. [Classical Operations Research Solvers (Baselines)](#7-classical-operations-research-solvers-baselines)
8. [Interactive Web Platform & GPS Navigation HUD](#8-interactive-web-platform--gps-navigation-hud)
9. [Dynamic On-Arrival Re-Routing & Resilience](#9-dynamic-on-arrival-re-routing--resilience)
10. [Benchmark Results & Comparative Analysis](#10-benchmark-results--comparative-analysis)
11. [Project Directory Architecture](#11-project-directory-architecture)
12. [Installation & Quickstart Guide](#12-installation--quickstart-guide)

---

## 1. Executive Summary & Problem Statement

Last-mile delivery represents over **40% to 53% of total supply chain logistics costs**. In rapidly growing urban centers like **Coimbatore (Tamil Nadu, India)**, logistics managers face severe operational challenges:
- **Traffic Congestion & Road Hierarchies**: Dense arterial roads (Avinashi Road, Trichy Road, Sathy Road) vs narrow residential cross-streets.
- **Heterogeneous Fleet Management**: Mixing lightweight two-wheelers (motorcycles/scooters) for compact parcels ($\le 15\text{ kg}$) with four-wheeler commercial mini-trucks (Tata Ace, Mahindra Bolero, Maruti Super Carry) for bulky/heavy freight ($\le 80\text{ kg}$).
- **Volatile Fuel Costs & Carbon Emissions**: Constantly shifting petrol and diesel pump prices, requiring multi-objective trade-offs between travel distance, fuel consumption in liters, and operational costs in Indian Rupees (₹).
- **Dynamic Delivery Uncertainties**: Customer unavailabilities, door-locked scenarios, and cancellations that demand real-time route rescheduling in milliseconds rather than hours.

Traditional Exact Mixed-Integer Programming (MIP) methods struggle to scale on dynamic real-time operations ($NP$-hard complexity), taking minutes or hours per instance. This project bridges **Modern Operations Research** and **Deep Reinforcement Learning (DRL)**, delivering mathematically near-optimal routes in **~15 milliseconds**.

```
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│  Morning Orders (CSV)  │ ───► │  Neural Policy Solver  │ ───► │ Turn-by-Turn GPS Nav   │
│  50 Real Urban Parcels │      │  Attention Model (RL)  │      │ Live Driver Manifests  │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

---

## 2. Operations Research (OR) Foundations & Formulations

The **Capacitated Vehicle Routing Problem (CVRP)** is an $NP$-hard combinatorial optimization problem defined on a complete directed graph $G = (V, E)$:
- $V = \{0, 1, 2, \dots, n\}$ is the set of vertices, where vertex $0$ denotes the **Central Logistics Depot** (Gandhipuram Depot), and vertices $i \in \{1, \dots, n\}$ represent customer delivery locations.
- $E = \{(i, j) : i, j \in V, i \ne j\}$ is the edge set representing road segments between locations.
- Each customer node $i \in V \setminus \{0\}$ is associated with a cargo demand $q_i \ge 0$.
- A fleet of $K$ vehicles is available at the depot, where each vehicle $k \in \{1, \dots, K\}$ has a maximum load capacity $Q_k$.

### Mathematical Formulation

Let the binary decision variable $x_{ijk} \in \{0, 1\}$ be $1$ if vehicle $k$ travels directly from node $i$ to node $j$, and $0$ otherwise. Let $u_{ik} \ge 0$ denote the cumulative cargo delivered by vehicle $k$ upon visiting node $i$.

$$\min \quad Z = w_{\text{dist}} \sum_{k=1}^K \sum_{i \in V} \sum_{j \in V} c_{ij} x_{ijk} + w_{\text{fuel}} \sum_{k=1}^K \sum_{i \in V} \sum_{j \in V} f_{ij}(u_{ik}) x_{ijk}$$

Subject to:

1. **Every customer is visited exactly once**:
   $$\sum_{k=1}^K \sum_{j \in V, j \ne i} x_{ijk} = 1 \quad \forall i \in V \setminus \{0\}$$

2. **Flow conservation at depot and customer nodes**:
   $$\sum_{j \in V, j \ne p} x_{jpk} - \sum_{j \in V, j \ne p} x_{pjk} = 0 \quad \forall p \in V, \forall k \in \{1, \dots, K\}$$

3. **Fleet departure and return constraint**:
   $$\sum_{j \in V \setminus \{0\}} x_{0jk} \le 1 \quad \forall k \in \{1, \dots, K\}$$

4. **Vehicle capacity and subtour elimination (Miller-Tucker-Zemlin constraints)**:
   $$u_{ik} - u_{jk} + Q_k x_{ijk} \le Q_k - q_j \quad \forall i, j \in V \setminus \{0\}, i \ne j, \forall k$$
   $$q_i \le u_{ik} \le Q_k \quad \forall i \in V \setminus \{0\}, \forall k$$

---

## 3. Deep Reinforcement Learning (DRL) Architecture

```
                    ┌──────────────────────────────────────────────────────────┐
                    │       MULTI-HEAD TRANSFORMER ENCODER (128-DIM)           │
                    │   Input Node Features: (x, y, demand, norm_dist)         │
                    └─────────────────────────────┬────────────────────────────┘
                                                  │ Node Embeddings (h_i)
                                                  ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                          DYNAMIC AUTOREGRESSIVE ATTENTION DECODER                             │
│ Context: [Depot Embedding, Current Vehicle Position, Remaining Capacity, Preference Vector]    │
│ Dynamic Infeasibility Mask: Masks visited nodes AND nodes where demand q_j > remaining Q      │
└───────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                                │ Softmax Tour Probabilities P(π|s)
                                                ▼
                                    Selected Stop Sequence
```

### Architecture Specifications:
1. **Preference-Conditioned Transformer Encoder**:
   - Maps each delivery hub coordinate $(x_i, y_i)$, cargo demand $q_i$, and normalized distance to a $128$-dimensional dense latent representation using Multi-Head Self-Attention ($3$ layers, $8$ attention heads).
   - Conditions dynamically on preference weights $\vec{w} = (w_{\text{dist}}, w_{\text{fuel}})$.
2. **Contextual Decoder with Dynamic Masking**:
   - Maintains real-time state: current vehicle location $v_t$, remaining vehicle capacity $Q_{\text{rem}}$, and visited node bitmask.
   - **Dynamic Infeasibility Mask**: At every decoding step $t$, mathematically enforces feasibility by setting logit $\mu_j = -\infty$ for any node $j$ if:
     - Customer $j$ has already been visited.
     - Customer demand $q_j > Q_{\text{rem}}$ (capacity violation).
   - When no valid customer can be served, the policy forces a return to the Central Depot, resets capacity $Q$, and dispatches the next vehicle.
3. **Training via REINFORCE with Rollout Baseline**:
   - Uses Policy Gradient with an exponential moving baseline $b(s)$ updated periodically via greedy rollout evaluations:
     $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim P_\theta} \left[ (R(\tau) - b(s)) \nabla_\theta \log P_\theta(\tau) \right]$$

---

## 4. Physics-Based Fuel Consumption & Emissions Model (CMEM)

Unlike basic CVRP formulations that assume fuel is strictly linear with distance, this project incorporates the **Comprehensive Modal Emissions Model (CMEM)**, where vehicle energy dissipation depends on instantaneous speed, rolling resistance, aerodynamic drag, vehicle curb weight, and **current cargo payload mass**:

```
                       ┌───────────────────────────────┐
                       │  Engine Friction Base Power   │
                       └──────────────┬────────────────┘
                                      │
┌────────────────────────┐            │            ┌────────────────────────┐
│  Aerodynamic Drag Rate │ ───────────┼─────────── │   Cargo Payload Mass   │
│   ~ 0.5 * Cd * A * v^3 │            │            │   ~ m_cargo * g * Cr   │
└────────────────────────┘            ▼            └────────────────────────┘
                       ┌───────────────────────────────┐
                       │ Total Fuel Rate (Liters / km) │
                       └───────────────────────────────┘
```

The fuel consumption $F_{ij}$ over a road segment of length $d_{ij}$ carrying cargo payload $m_{\text{load}}$ is computed as:

$$F_{ij} = \left( \frac{\xi}{k \cdot \psi} \left( k N_e V_d + \frac{1}{2} C_d \rho A v^3 + (M_{\text{curb}} + m_{\text{load}}) g (\sin \theta + C_r \cos \theta) v \right) \right) \cdot \frac{d_{ij}}{v}$$

Where:
- $M_{\text{curb}}$ = Vehicle unladen curb weight ($kg$)
- $m_{\text{load}}$ = Instantaneous cargo payload ($kg$), decreasing as deliveries are completed
- $C_d, A$ = Aerodynamic drag coefficient and frontal surface area
- $C_r$ = Rolling resistance coefficient of Indian tarmac roads
- $v$ = Operating urban speed ($30-45\text{ km/h}$)

---

## 5. Heterogeneous Fleet Logistics (Indian Urban Context)

To match actual urban delivery operations in Indian cities, the system assigns orders to two primary vehicle categories:

| Vehicle Class | Example Models | Max Payload ($Q_k$) | Engine & Fuel Type | Efficiency & Pricing | Typical Order Assignment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2-Wheeler (Delivery Bike / Scooter)** | Honda Activa, TVS Jupiter, Hero Splendor | **$15\text{ kg}$** | Petrol ($4\text{-Stroke}$) | ~₹1.80 / km (~$45\text{ km/L}$) | Small parcels, medicines, documents, mobile accessories |
| **4-Wheeler Mini-Truck (Petrol)** | Maruti Suzuki Super Carry | **$60\text{ kg}$** | Petrol | ₹102.80 / L (~$14\text{ km/L}$) | Medium packages, consumer electronics, kitchen appliances |
| **4-Wheeler Mini-Truck (Diesel)** | Tata Ace Gold, Mahindra Bolero Maxx | **$80\text{ kg}$** | High-Torque Diesel (CRDe) | ₹92.50 / L (~$12\text{ km/L}$) | Heavy freight, bulk cartons, washing machines, furniture |

### Automated Live Fuel Price Engine
The backend includes an automated real-time scraper/feed for Coimbatore / Tamil Nadu retail pump prices:
- **Live Petrol**: `₹102.80 / Litre`
- **Live Diesel**: `₹92.50 / Litre`
- Auto-calculates exact fuel expenditure in ₹ for each individual driver manifest.

---

## 6. Real-World Urban Network: Coimbatore Case Study

The system is calibrated on the urban road infrastructure of **Coimbatore, Tamil Nadu, India**, with the central distribution warehouse situated at the **Gandhipuram Central Logistics Depot** ($11.0168^\circ\text{N}, 76.9558^\circ\text{E}$).

```
                             [Thudiyalur 31-33]
                                     │
                     [Saravanampatti Hubs 9-12] (IT Corridor)
                                     │
   [Vadavalli 28-30] ── [Saibaba Colony 43-45] ── [Ganapathy 24-27]
           │                         │
           │              ★ [Gandhipuram Central Depot]
           │                         │
    [RS Puram 1-4] ─────── [Town Hall 50] ─────── [Peelamedu Tech Park 5-8]
           │                         │                         │
    [Ukkadam 17-20] ─── [Ramanathapuram 40-42] ─── [Singanallur 13-16]
           │                         │                         │
   [Kovaipudur 37-39]      [Kurichi SIDCO 46-47]      [Ondipudur 34-36]
```

### Pre-loaded 50-Order Realistic Dataset (`data/sample_50_orders.csv`):
- **50 Customer Orders** distributed across 15 major urban commercial and residential zones:
  - **RS Puram & Saibaba Colony**: High-density residential retail.
  - **Peelamedu & Saravanampatti**: IT SEZ tech parks, apartments, and educational institutes.
  - **Singanallur & Kurichi SIDCO**: Medium and heavy industrial hardware orders.
  - **Ukkadam & Town Hall**: Commercial wholesale and apparel deliveries.
  - **Vadavalli, Thudiyalur, Kovaipudur, Ondipudur, Aerodrome**: Suburban and mixed residential.

---

## 7. Classical Operations Research Solvers (Baselines)

To rigorously validate the Deep RL model, the platform includes exact and heuristic OR baselines in `baselines/`:

1. **Google OR-Tools (Constraint Programming & Guided Local Search)**:
   - Formulates the CVRP with vehicle routing dimension callbacks.
   - Solves with `PARALLEL_CHEAPEST_INSERTION` initial heuristic followed by `GUIDED_LOCAL_SEARCH` metaheuristic.
2. **Clarke-Wright Savings Algorithm (1964)**:
   - Computes savings $S_{ij} = c_{0i} + c_{0j} - c_{ij}$ for all customer pairs $(i, j)$ and merges routes in parallel subject to vehicle capacity constraints.
3. **Nearest Neighbor Greedy Heuristic**:
   - Constructs routes by iteratively extending the tour to the closest unvisited feasible customer node.

---

## 8. Interactive Web Platform & GPS Navigation HUD

The interactive web dashboard provides a complete operations control center:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  ⚡ MO-CVRP DISPATCHER                     ⛽ Petrol: ₹102.80/L  │  🚚 Diesel: ₹92.50/L  │
├───────────────────────────────┬────────────────────────────────────────────────────────┤
│  FLEET & ALGORITHM SETUP      │  🗺️ REAL-TIME ROAD NETWORK & GPS SATELLITE HUD          │
│                               │                                                        │
│  Dataset: [Coimbatore (50)]   │  ┌──────────────────────────────────────────────────┐  │
│  🛵 2-Wheelers: [ 2 ]         │  │ 🧭 GPS HUD: Navigating to Peelamedu Tech Park    │  │
│  🚚 Mini-Trucks: [ 2 ]        │  │ Speed: 30 km/h • Cargo: 28 kg • Est: 09:45 AM    │  │
│  Fuel: [ Diesel ▼ ]           │  └──────────────────────────────────────────────────┘  │
│                               │  [🛰️ Satellite Layer]  [🗺️ Street Layer]              │
│  Preference Sliders:          │                                                        │
│  • Distance:  [ 0.50 ]        │  🛵 Bike #1: 16.4 km | 0.41 L | ₹42.15                 │
│  • Fuel Eco:  [ 0.30 ]        │  🚚 Truck #1: 42.8 km | 3.75 L | ₹346.88               │
│                               │  🚚 Truck #2: 38.2 km | 3.32 L | ₹307.10               │
│  [🚀 Optimize Routes (15ms)]  │                                                        │
├───────────────────────────────┴────────────────────────────────────────────────────────┤
│  📋 INDIVIDUAL DRIVER MANIFEST CARDS & STOP-BY-STOP DELIVERY SCHEDULES                 │
│  • Stop #1: RS Puram (8 kg - Medium) ➔ Delivered 08:45 AM                             │
│  • Stop #2: Peelamedu Tech Park (14 kg - Large) ➔ Est. Arrival 09:15 AM               │
│  • Stop #3: Saravanampatti Zone (42 kg - Heavy) ➔ Est. Arrival 09:50 AM               │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Features:
- **Turn-by-Turn GPS Navigation Simulation**: Watch animated delivery vehicles (🛵/🚚) follow road geometry with live driving HUD, speed controls ($1\times, 2\times, 4\times$), and real-time cargo payload reduction.
- **Satellite & Street View Layer Toggle**: Instant high-resolution satellite imagery (Esri World Imagery) and clean street maps (CartoDB Voyager).
- **Vehicle Route Isolation**: One-click filtering to isolate specific vehicle tours and inspect their driving path and stops.
- **Driver Manifest Cards**: Stop sequence, customer names, package sizes (**Small**, **Medium**, **Large/Heavy**), estimated arrival times, Liters consumed, and total Rupees spent.

---

## 9. Dynamic On-Arrival Re-Routing & Resilience

When a driver arrives at a delivery location:
1. An **On-Arrival Delivery Action Modal** displays:
   - Customer Name, Phone Number, Full Address, and Package Item.
2. Two interactive options:
   - **`[✅ Mark Delivered]`**: Stamps the delivery timestamp (e.g., `09:48 AM`), deducts package weight from cargo mass, recalculates fuel economy, and advances navigation.
   - **`[❌ Customer Unavailable / Failed]`**: Triggers dynamic fallback $\rightarrow$ The **Deep RL neural policy re-optimizes remaining stops in ~12 milliseconds**, rescheduling a retry at the end of the shift with minimal deadhead kilometers and zero wasted fuel.

---

## 10. Benchmark Results & Comparative Analysis

Evaluation across 100 CVRP test instances (20–50 customers) comparing solution quality and computational latency:

| Optimization Algorithm | Average Tour Distance | Total Fuel Consumed | Fleet Cost (₹ INR) | Average Solve Latency | Optimality Gap |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **🧠 Deep RL Attention Policy (Ours)** | **152.4 km** | **11.67 L** | **₹1,067.63** | **~14.8 ms** | **+1.2% (Real-Time)** |
| **🎯 Google OR-Tools (Exact / GLS)** | **150.6 km** | **11.45 L** | **₹1,048.20** | **~3,250.0 ms** | **0.0% (Optimal Baseline)** |
| **📊 Clarke-Wright Savings (1964)** | **168.2 km** | **13.10 L** | **₹1,198.40** | **~38.5 ms** | **+11.7%** |
| **🚶 Nearest Neighbor Greedy** | **194.5 km** | **15.82 L** | **₹1,447.50** | **~8.2 ms** | **+29.1%** |

> **Key Takeaway**: The Deep RL Neural Policy achieves within **~1.2% of Google OR-Tools' solution quality** while solving **over 200× faster**, enabling dynamic real-time re-routing directly on driver mobile devices.

---

## 11. Project Directory Architecture

```
orproject/
├── data/
│   ├── sample_50_orders.csv       # 50 realistic Coimbatore customer orders
│   ├── cvrp_generator.py          # Synthetic uniform/clustered CVRP generator
│   └── benchmark_parser.py        # CVRPLIB Augerat Set A/B/P benchmark parser
├── models/
│   ├── attention_model.py         # Multi-Head Attention Encoder-Decoder Policy
│   └── policy_solver.py           # Sub-second neural inference wrapper
├── rl/
│   ├── environment.py             # Vectorized Gym-like MO-CVRP environment
│   ├── rewards.py                 # CMEM load-dependent physics & ₹ fuel pricing
│   ├── baseline_policy.py         # Rollout baseline evaluator
│   └── trainer.py                 # REINFORCE training loop with baseline updates
├── baselines/
│   ├── ortools_solver.py          # Google OR-Tools GLS solver
│   ├── clarke_wright.py           # Parallel Clarke-Wright Savings heuristic
│   └── nearest_neighbor.py        # Greedy nearest neighbor heuristic
├── real_world/
│   ├── coimbatore_data.py         # 15 delivery zones GPS & road coordinates
│   └── osrm_client.py             # Road routing & polyline network generator
├── evaluation/
│   ├── metrics.py                 # Distance, emissions, fuel, cost calculations
│   └── pareto.py                  # Pareto frontier & hypervolume indicator
├── web/
│   ├── app.py                     # FastAPI REST server & re-optimization API
│   └── static/
│       ├── index.html             # Glassmorphic Operations Dashboard
│       ├── css/styles.css         # Dark luxury theme & GPS HUD styling
│       └── js/app.js              # Leaflet Satellite map & turn-by-turn navigator
├── mobile-app/                    # iOS React Native Expo Go Mobile Application
│   ├── App.js                     # Mobile dispatcher & on-arrival delivery modal
│   ├── package.json               # Expo SDK 54 dependencies
│   └── app.json                   # iOS configuration & bundle settings
├── saved_checkpoints/             # Trained PyTorch neural network weights
│   ├── mo_cvrp_model.pt           # Multi-Objective baseline model
│   └── mo_cvrp_petrol.pt          # Petrol-tuned fleet weights
├── train.py                       # CLI DRL training script
├── evaluate.py                    # Algorithm benchmarking & battle runner
├── run_dashboard.py               # Web dashboard server launcher
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## 12. Installation & Quickstart Guide

### Prerequisites
- **Python 3.10+** (Tested on Python 3.12)
- **Node.js 18+** (Optional, for Expo Go mobile app)

### Step 1: Clone Repository & Install Python Dependencies
```bash
git clone https://github.com/your-username/orproject.git
cd orproject

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Launch the Interactive Web Dashboard
```bash
python run_dashboard.py --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to:
1. Select the **Coimbatore (50 Orders)** dataset or upload a custom CSV.
2. Configure today's available **2-Wheelers** and **4-Wheeler Mini-Trucks (Diesel / Petrol)**.
3. Click **"🚀 Generate Optimal Routes"** to view neural dispatch routes in ~15 ms.
4. Click on any vehicle and press **"▶ Start GPS Navigation"** to watch turn-by-turn vehicle tracking on satellite maps.

### Step 3: Run the Benchmark Suite (Algorithm Battles)
```bash
python evaluate.py --dataset coimbatore --algorithm all --fuel_type diesel
```

### Step 4: Train Custom Deep RL Models
```bash
python train.py --epochs 20 --batches 30 --batch_size 64 --num_customers 20 --capacity 30 --fuel_type diesel --save_path saved_checkpoints/custom_cvrp.pt
```

---

## 📜 Academic References & Citations
1. **Kool, W., van Hoof, H., & Welling, M. (2019)**. *Attention, Learn to Solve Routing Problems!* International Conference on Learning Representations (ICLR).
2. **Barth, M., et al. (2000)**. *Comprehensive Modal Emissions Model (CMEM)*. Transportation Research Record.
3. **Clarke, G., & Wright, J. W. (1964)**. *Scheduling of Vehicles from a Central Depot to a Number of Delivery Points*. Operations Research, 12(4), 568-581.
4. **Perron, L., & Furnon, V.** *Google OR-Tools*. Google Developers.

---

<div align="center">
  <sub>Developed for Smart Urban Logistics & Green Operations Research • Built with PyTorch, FastAPI, Leaflet, and Expo.</sub>
</div>
