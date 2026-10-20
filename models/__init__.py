"""Neural models are optional and loaded only when requested."""
def __getattr__(name):
    if name == "NeuralPolicySolver":
        from models.policy_solver import NeuralPolicySolver
        return NeuralPolicySolver
    if name == "MultiObjectiveAttentionModel":
        from models.attention_model import MultiObjectiveAttentionModel
        return MultiObjectiveAttentionModel
    raise AttributeError(name)
