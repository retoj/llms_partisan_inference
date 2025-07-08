import argparse
from pathlib import Path
import pandas as pd

from helpers import post_process_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--results_path")
    parser.add_argument("-o", "--results_post_processed_dir")
    args = parser.parse_args()

    results = pd.read_csv(args.results_path)

    results = results.apply(post_process_output, axis=1)

    results_post_processed_success_path = Path(args.results_post_processed_dir) / "post_processed_success.csv"
    results_post_processed_failure_path = Path(args.results_post_processed_dir) / "post_processed_failure.csv"

    results_success = results[~results["Failure"]]
    results_failure = results[results["Failure"]]

    results_success.to_csv(results_post_processed_success_path, index=False)
    results_failure.to_csv(results_post_processed_failure_path, index=False)



if __name__ == "__main__":
    main()