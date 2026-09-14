"""Command-line entry point for the reproducible small-set validation."""

import argparse
from src.experiment import run_small_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-files", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    report = run_small_experiment(args.n_files, args.seed, args.output_dir)
    print(
        f"Validated {report['n_files']} stars; grid error={report['grid_relative_max_error']:.3g}"
    )
    for name, metrics in report["models"].items():
        print(
            f"{name}: ARI={metrics['ari_all']:.3f}, coverage={metrics['coverage']:.1%}, clusters={metrics['n_clusters']}"
        )


if __name__ == "__main__":
    main()
