# Implementation review · 20 October 2026

This review distinguishes the operational dispatch product from the routing research code. The original README is preserved as a historical outline, not verified implementation evidence.

## Directory findings and changes

| Area | Original issue | Current implementation |
| --- | --- | --- |
| FastAPI | Unbounded fleet assignment after homogeneous solve; blocking work inside async handlers; mutable global fuel prices | Finite mixed-fleet planner, sync handlers in thread pool, bounded solves, request-local prices |
| Validation | Arbitrary strings and untyped customer dictionaries; custom import referenced missing NumPy import | Typed constrained schemas, ID/depot/window checks, bounded CSV parsing |
| Orders | Default Coimbatore solver used 15 hubs despite a separate 50-row CSV | Operational planner loads the actual 50 customer rows and preserves addresses/products |
| Fuel | Fixed prices described as live; bike cost unrelated to litres; minutes/hours mismatch | Manual prices labelled honestly, cost derived from litres, minute-based timing |
| Delivery workflow | Reordered list claimed neural optimization in a fabricated 12 ms; no saved delivery state | Transactional events, current-position retry, retained failed payload, updated remaining ETAs, one retry then explicit failure |
| Persistence | No operational database | SQLite dispatches/events, history, version conflict detection and deduplication |
| Web | Dense dark dashboard with simulated navigation | Responsive warm-white dashboard, contrasting serif/sans typography, sage/peach accents, search/import, route isolation and saved progress |
| Mobile | Hardcoded loopback URL, sample manifests silently retained after API failure, placeholder upload | Native Expo SDK 57 app matched to the App Store client, LAN discovery/settings, actual API data, visible errors, file picker, native maps and Apple Maps handoff |
| Attention model | Oversize demands could generate incomplete tours; missing weights silently allowed random model use | Demand/coverage validation; required trained weights; offline checkpoint detection |
| Training | Fuel computed after parcel drop; normalized mass scaled by arbitrary 1000; unsupported time objective sampled | Fuel charged before drop, real route load, capacity conversion, two-objective preference training |
| Baselines | Nearest neighbour could loop indefinitely on oversize parcel; savings allowed overweight singleton | Explicit invalid demand rejection in research solvers; operational adapter reports oversize parcels separately |
| Benchmark parser | Floating distances compared with rounded EUC_2D reference | Round EUC_2D distances before research comparisons; unverified embedded second dataset flagged separately |
| Dependencies | Unbounded Python requirements; Expo 54; offline dependency/bytecode placeholders | Locked updated backend, separate training extras, SDK-aligned mobile packages, clean install and local cache |
| Operations | No deployment definition or substantive assertions | Nonroot Docker image, health endpoint, environment configuration, integration and browser interaction tests |

## Scope and remaining release work

The application supports single-depot, single-departure planning with soft time windows. It does not implement an authenticated driver identity system, background GPS, offline event synchronization, route push notifications, proof-of-delivery photos, customer notifications, multi-depot routing, reload shifts or depot handover events. These require explicit product design and acceptance criteria.

Shared access-key protection is available, including mandatory configuration in production mode. A public multi-user release still needs per-user roles, credential lifecycle, perimeter rate limits and a supported TLS deployment. Customer information stays on the API; mobile local storage holds connection settings and a dispatch ID, while iOS Keychain holds the access key.

The new planner's fuel objective uses a mean-payload arc approximation during OR-Tools search, followed by actual route-payload evaluation. Fuel physics remain a linear surrogate. The neural model is a research seed with finite-fleet adaptation; heterogeneous fleet feasibility comes from the adapter, not the learned model itself.

The original three checkpoint files are offline OneDrive placeholders in this workspace. They were preserved. They must be hydrated and evaluated, or replaced by retrained weights selected through NEURAL_CHECKPOINT_PATH. A smoke-trained checkpoint is not evidence of routing quality.

The offline geometry utility in `real_world/osrm_client.py` remains a legacy research utility. The operational clients deliberately use labelled stop connections and external maps for actual directions.

## Verification

- API integration tests cover all 50 orders, unique coverage, hard capacities, fleet scarcity, per-dispatch price isolation, CSV validation, nonconsecutive IDs, persistent delivery state, event replay, concurrency conflicts, retry semantics, authentication and payload limits.
- Browser automation verifies planning, route selection, delivery state/history, CSV import, HTML escaping and a 390 px layout. Screenshots are in artifacts/.
- Expo dependency compatibility checks and iOS bundle export validate packaging. A bundle export is not physical-device acceptance.
- A small corrected training/evaluation smoke run completes successfully, without changing original checkpoints.
- Compatible npm audit fixes removed a critical shell-quote advisory. Remaining Expo/Metro transitive advisories have no compatible automatic fix in the installed SDK tree; do not apply the proposed forced downgrade to Expo 44. Track upstream resolutions before a public release.

No original benchmark claim has been reproduced. Use generated benchmark reports and record hardware, data, capacities, versions, seeds and timing scope before publishing performance claims.
