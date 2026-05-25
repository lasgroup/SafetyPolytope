import argparse
import os
import subprocess
from typing import List

from safety_polytope.data.beaver_data import (
    get_categories,
    merge_hidden_states,
)


def get_model_name(model_path: str) -> str:
    """Extract standardized model name from model path."""
    if "Ministral-8B" in model_path:
        return "ministral-8b"
    elif "llama-2-7b" in model_path:
        return "llama2-7b"
    elif "Qwen2-1.5B" in model_path:
        return "qwen2-1.5b"
    elif "Llama-3.1-8B" in model_path:
        return "llama3-8b"
    elif "Qwen3-8B" in model_path:
        return "qwen3-8b"
    else:
        raise NotImplementedError(
            f"Please manually configure a model name for {model_path}."
        )


def escape_category(text: str) -> str:
    """Escape commas in text if present."""
    if "," in text:
        return text.replace(",", "\\,")
    return text


def run_polytope_training(
    data_path: str,
    model_name: str,
    categories: List[str],
    mode: str,
    dataset_name: str,
    model_path: str,
):
    """Run polytope training for each balanced dataset. If one wishes to train
    on the full dataset, one can first merge all the category hidden state
    files into a single file, and then run polytope training on that file."""

    if mode == "slurm":
        # Collect all valid hidden states paths
        valid_paths = []
        for i, category in enumerate(categories):
            hidden_states_path = os.path.join(
                data_path,
                dataset_name,
                model_name,
                f"balanced_hidden_states_{category}.pth",
            )
            if not os.path.exists(hidden_states_path):
                print(f"Warning: File not found - {hidden_states_path}")
                continue
            # Escape commas in the path
            valid_paths.append(escape_category(hidden_states_path))

        if not valid_paths:
            print(
                "No valid hidden states files found. Skipping polytope training."
            )
            return

        # Join all escaped paths with commas for parallel execution
        all_paths = ",".join(valid_paths)
        cmd = [
            "python",
            "src/safety_polytope/polytope/learn_polytope.py",
            f"dataset.hidden_states_path={all_paths}",
            f"model_path={model_path}",
            "exp_ident=polytope_training",
            "--multirun",
        ]

        print("\nStarting parallel training for all categories using SLURM")
        print(f"Command: {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)

    else:
        # Sequential execution for local mode
        for i, category in enumerate(categories):
            hidden_states_path = os.path.join(
                data_path,
                dataset_name,
                model_name,
                f"balanced_hidden_states_{category}.pth",
            )

            if not os.path.exists(hidden_states_path):
                print(f"Warning: File not found - {hidden_states_path}")
                continue

            cmd = [
                "python",
                "src/safety_polytope/polytope/learn_polytope.py",
                f"dataset.hidden_states_path={hidden_states_path}",
                f"model_path={model_path}",
                f"exp_ident=polytope_training_{i}",
            ]

            print(f"\nStarting training for category: {category}")
            print(f"Command: {' '.join(cmd)}\n")
            # Wait for the process to complete before starting the next one
            subprocess.run(cmd, check=True)


def save_hidden_states(args, categories: List[str], data_path: str):
    """Save hidden states for categories either sequentially or using slurm."""
    layer_number = args.layer_number
    
    base_cmd = [
        "python",
        "src/safety_polytope/data/save_hs.py",
        f"dataset={args.dataset}",
        f"layer_number={layer_number}",
        f"model_path={args.model_path}",
        "exp_ident=save_beaver_states",
        f"save_hs_root_dir={data_path}",
        "hydra.job.chdir=False"
    ]

    # Add total_datapoints parameter if using reduced data
    if args.reduced_data:
        base_cmd.append("total_datapoints=100")

    if args.mode == "slurm":
        # For slurm, run all categories in parallel using multirun
        # Escape commas in each category name and join them
        escaped_categories = [escape_category(cat) for cat in categories]
        categories_str = ",".join(escaped_categories)
        cmd = base_cmd + [f"dataset.category={categories_str}", "--multirun"]
        print("\nGenerating hidden states for all categories using SLURM")
        print(f"Command: {' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)
    else:
        # For local mode, run categories sequentially
        for category in categories:
            escaped_category = escape_category(category)
            cmd = base_cmd + [f"dataset.category={escaped_category}"]
            print(f"\nGenerating hidden states for category: {category}")
            print(f"Command: {' '.join(cmd)}\n")
            # Wait for the process to complete before starting the next one
            subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run BeaverTails dataset training pipeline"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model, e.g., Qwen/Qwen2-1.5B-Instruct",
    )
    parser.add_argument(
        "--layer_number",
        type=int,
        required=True,
        help="Layer number to extract hidden states from",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="beaver_tails",
        help="Dataset name",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=os.getcwd(),
        help="Base path for saving data",
    )
    parser.add_argument(
        "--skip_hs_generation",
        action="store_true",
        help="Skip hidden states generation and only run polytope training",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "slurm"],
        default="local",
        help="Run mode: local (sequential) or slurm (parallel)",
    )
    parser.add_argument(
        "--reduced_data",
        action="store_true",
        help="Run with reduced dataset size (100 datapoints)",
    )
    args = parser.parse_args()

    # Get all categories
    categories = get_categories()

    # Get model name
    model_name = get_model_name(args.model_path)

    data_path = os.path.join(args.base_path, "hs_data")

    if not args.skip_hs_generation:
        print("Starting hidden states generation...")
        save_hidden_states(args, categories, data_path)

        # After generating hidden states for all categories, combine safe and
        # unsafe data to create a (balanced) classification dataset.
        print("Merging hidden states for all categories...")
        merge_hidden_states(
            base_path=data_path,
            dataset_name=args.dataset,
            model_file_name=model_name,
        )

    # Run polytope training
    print("Starting polytope training...")
    # run_polytope_training(
    #     data_path,
    #     model_name,
    #     categories[:-1],  # Exclude the last category (safety)
    #     args.mode,
    #     args.dataset,
    #     args.model_path,
    # )


if __name__ == "__main__":
    main()
