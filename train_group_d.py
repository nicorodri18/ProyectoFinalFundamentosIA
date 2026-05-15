import argparse
import importlib.util
from pathlib import Path


def load_group_d_agent():
    root = Path(__file__).resolve().parent
    policy_path = root / "groups" / "Group D" / "policy.py"
    spec = importlib.util.spec_from_file_location("group_d_policy", policy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load policy from {policy_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MonteCarloQAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Group D Q-values.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of self-play games used for training.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.15,
        help="Exploration rate for epsilon-greedy training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=911,
        help="Random seed for reproducible training.",
    )
    args = parser.parse_args()

    agent_class = load_group_d_agent()
    summary = agent_class.train_q_values(
        episodes=args.episodes,
        epsilon=args.epsilon,
        seed=args.seed,
    )

    print(f"Episodes: {summary['episodes']}")
    print(f"Stored q-values: {summary['stored_q_values']}")
    print(f"Red wins: {summary['red_wins']}")
    print(f"Yellow wins: {summary['yellow_wins']}")
    print(f"Draws: {summary['draws']}")


if __name__ == "__main__":
    main()
