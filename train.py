"""
train.py
CLI Training Script for Multi-Objective Deep Reinforcement Learning (MO-CVRP).

Usage:
  python train.py --epochs 20 --batches 30 --batch_size 64 --num_customers 20 --capacity 30
"""

import os
import sys
from pathlib import Path
sys.pycache_prefix = str(Path(__file__).resolve().parent / "runtime/pycache")
import argparse
import time
from rl.trainer import RLTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Objective Attention Policy for CVRP")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batches", type=int, default=25, help="Batches per epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_customers", type=int, default=20, help="Number of customers per instance")
    parser.add_argument("--capacity", type=int, default=30, help="Vehicle capacity")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="saved_checkpoints/mo_cvrp_model.pt", help="Checkpoint save path")
    parser.add_argument("--fuel_type", type=str, default="diesel", choices=["diesel", "petrol", "ev"], help="Fleet fuel type")
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-OBJECTIVE DEEP REINFORCEMENT LEARNING TRAINING FOR CVRP")
    print(f"Epochs: {args.epochs} | Batches/Epoch: {args.batches} | Batch Size: {args.batch_size}")
    print(f"Instance Size: {args.num_customers} customers | Capacity: {args.capacity} | Fleet: {args.fuel_type.upper()}")
    print("=" * 70)

    trainer = RLTrainer(
        num_customers=args.num_customers,
        vehicle_capacity=args.capacity,
        learning_rate=args.lr
    )

    print(f"Using compute device: {trainer.device}")
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_res = trainer.train_epoch(
            num_batches=args.batches,
            batch_size=args.batch_size,
            fuel_type=args.fuel_type
        )
        eval_res = trainer.evaluate_greedy(
            batch_size=args.batch_size,
            fuel_type=args.fuel_type
        )
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] ({elapsed:.1f}s) | "
              f"Loss: {train_res['loss']:>7.4f} | "
              f"Avg Dist: {train_res['avg_distance']:>6.2f} | "
              f"Avg Fuel: {train_res['avg_fuel']:>5.2f} | "
              f"Eval Dist: {eval_res['eval_distance']:>6.2f} | "
              f"Eval Cost: {eval_res['eval_scalarized_cost']:>6.3f}")

    total_time = time.time() - total_start
    print(f"\nTraining completed in {total_time:.1f} seconds.")
    trainer.save_checkpoint(args.save_path)


if __name__ == "__main__":
    main()
