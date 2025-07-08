# llms_partisan_inference
Repository for the ACL 2025 Paper

## Overview

This repository contains code and resources for the ACL 2025 paper on LLMs and partisan inference. It is designed to run the experiments described in the paper using both open-weight (GGUF) models and OpenAI models, and to post-process the results.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/llms_partisan_inference.git
    cd llms_partisan_inference
    ```

2. (Recommended) Create and activate a virtual environment and install dependencies: llama-cpp-python, openai, pandas

## Usage

### Running Experiments

#### For Open-Weight (GGUF) Models

1. Download your GGUF model and place it on disk.
2. Run an experiment with:
    ```sh
    python run.py \
      --prompts_path path/to/prompts.json \
      --results_path path/to/results.csv \
      --model_path path/to/model.gguf \
      --model_name your_model_name
    ```
   - `--prompts_path`: Path to the prompts file (JSON).
   - `--results_path`: Path where the results CSV will be saved.
   - `--model_path`: Path to the GGUF model file.
   - `--model_name`: Name for the model (for logging/results).

#### For OpenAI Models

1. Ensure you have your OpenAI API key set up.
2. Run an experiment with:
    ```sh
    python run.py \
      --prompts_path path/to/prompts.json \
      --results_path path/to/results.csv \
      --model_path OPEN_AI \
      --model_name openai-model-name
    ```
   - `--model_name` should match the model name as described in OpenAI’s documentation (e.g., `gpt-4-turbo`).

### Post-Processing Results

After running an experiment, post-process the results with:
```sh
python post_process.py \
  --results_path path/to/results.csv \
  --results_post_processed_dir path/to/output_dir
```
This will generate:
- `post_processed_success.csv`: Samples that succeeded in post-processing.
- `post_processed_failure.csv`: Samples that failed post-processing.

## Project Structure

- `run.py` - Script to run experiments for a single LLM.
- `post_process.py` - Script to post-process experiment results.
- `helpers.py` - Helper functions and utilities.
- `data/` - Data files (prompts).

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this codebase, please cite our paper:

```
@inproceedings{your2025paper,
  title={LLMs and Partisan Inference},
  author={Your Name and Coauthors},
  booktitle={ACL 2025},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact [reto.gubelmann@uzh.ch](mailto:reto.gubelmann@uzh.ch).