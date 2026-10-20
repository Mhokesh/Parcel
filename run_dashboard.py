"""
run_dashboard.py
Launch script for the interactive MO-CVRPTW Web Dashboard.

Usage:
  python run_dashboard.py --port 8000
"""

import argparse
import sys
from pathlib import Path
sys.pycache_prefix = str(Path(__file__).resolve().parent / "runtime/pycache")
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Launch MO-CVRPTW Web Application")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    args = parser.parse_args()

    print("=" * 70)
    print("STARTING MULTI-OBJECTIVE CVRP DEEP RL WEB DASHBOARD")
    print(f"URL: http://{args.host}:{args.port}")
    print("=" * 70)

    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
