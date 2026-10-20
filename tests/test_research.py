import pytest
import numpy as np
from baselines.clarke_wright import ClarkeWrightSolver
from baselines.nearest_neighbor import NearestNeighborSolver
from data.benchmark_parser import load_cvrplib_benchmark


@pytest.mark.parametrize("solver", [ClarkeWrightSolver(), NearestNeighborSolver()])
def test_oversize_research_order_fails_instead_of_hanging(solver):
    with pytest.raises(ValueError):
        solver.solve(dict(coords=np.zeros((2, 2)), demands=np.array([0, 100]), capacity=15, distance_matrix=np.zeros((2, 2))))


def test_cvrplib_uses_integer_euc2d_distances():
    instance = load_cvrplib_benchmark("A-n32-k5")
    assert np.array_equal(instance['distance_matrix'], np.round(instance['distance_matrix']))


def test_neural_requires_real_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from models.policy_solver import NeuralPolicySolver
    with pytest.raises(FileNotFoundError):
        NeuralPolicySolver(str(tmp_path / 'missing.pt'))


def test_training_fuel_carries_package_until_arrival():
    torch = pytest.importorskip("torch")
    from rl.environment import compute_batch_costs
    coords = torch.tensor([[[0., 0.], [1., 0.]]])
    demands = torch.tensor([[0., .5]])
    result = compute_batch_costs(coords, demands, torch.tensor([[1, 0]]), torch.tensor([[.5, .5, 0.]]), vehicle_capacity=80)
    expected = (0.085 + 0.000045*40) + 0.085
    assert result['total_fuel'].item() == pytest.approx(expected)


def test_neural_checkpoint_roundtrip_covers_customers(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    from models.attention_model import MultiObjectiveAttentionModel
    from models.policy_solver import NeuralPolicySolver
    from data.cvrp_generator import CVRPInstanceGenerator
    path = tmp_path / 'test-only.pt'
    torch.save({'model_state_dict': MultiObjectiveAttentionModel().state_dict()}, path)
    solver = NeuralPolicySolver(str(path), device='cpu')
    instance = CVRPInstanceGenerator(seed=42, vehicle_capacity=30).generate_instance(5)
    result = solver.solve(instance)
    assert sorted(n for r in result['routes'] for n in r if n) == [1,2,3,4,5]
    assert all(sum(instance['demands'][n] for n in route) <= 30 for route in result['routes'])
