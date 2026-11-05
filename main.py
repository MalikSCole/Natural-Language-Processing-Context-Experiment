"""Prepare inputs; this stage does not run translation or produce scores."""

import argparse
from pathlib import Path

from experiments.run_all import prepare_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/samples/synthetic.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/prepared_inputs.csv"))
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    count = prepare_inputs(args.dataset, args.output, args.limit)
    print(f"Prepared {count} inputs in {args.output}. No model inference performed.")


if __name__ == "__main__":
    main()
