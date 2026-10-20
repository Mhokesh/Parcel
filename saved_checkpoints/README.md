# Optional neural checkpoints

Legacy `.pt` files from the original local research workspace are not distributed with this repository. They predate the corrected training objective and were unavailable as local files during publication.

The dispatcher defaults to OR-Tools and runs without these weights or PyTorch. To use the neural research engine:

1. Install `requirements-training.txt` in your Python environment.
2. Train a model with `train.py`, saving it under this directory.
3. Set `NEURAL_CHECKPOINT_PATH` to the resulting file before starting the API.
4. Evaluate the model using `evaluate.py` before relying on its routing quality.

Example:

```bash
python train.py --epochs 20 --batches 30 --batch_size 64 --num_customers 20 --capacity 30 --fuel_type diesel --save_path saved_checkpoints/custom_cvrp.pt
```

Checkpoint binaries are ignored by Git. Keep any OneDrive-hosted checkpoints available locally before inference. The existence of a checkpoint does not establish model quality or an optimality guarantee.
