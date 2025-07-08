import argparse
from pathlib import Path
import pandas as pd

from helpers import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--prompts_path")
    parser.add_argument("-o", "--results_path")
    parser.add_argument("-p", "--model_path")
    parser.add_argument("-n", "--model_name")
    args = parser.parse_args()

    prompts = pd.read_csv(args.prompts_path)

    if args.model_path == "OPEN_AI":
        run_model_open_ai(prompts, args.results_path, args.model_name)
    else:
        run_model_llama_cpp(prompts, args.results_path, args.model_name, args.model_path)

if __name__ == "__main__":
    main()