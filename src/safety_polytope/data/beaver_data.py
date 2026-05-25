import os
from typing import List

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


def get_beaver_dataset(
    dataset_name, split="30k_train", category=None, hint_category=True
):
    train_dataset = load_dataset(dataset_name, split=split)
    test_split = "30k_test" if split == "30k_train" else "330k_test"
    test_dataset = load_dataset(dataset_name, split=test_split)

    train_dataset = reformat_beaver_dataset(train_dataset, hint_category)
    test_dataset = reformat_beaver_dataset(test_dataset, hint_category)

    if category is not None:
        train_dataset = filter_by_category(train_dataset, category)
        test_dataset = filter_by_category(test_dataset, category)

    return train_dataset, test_dataset


def reformat_beaver_dataset(dataset, hint_category=True):
    """Reformat the BeaverTail dataset to be compatible with the SafetyDataset
    class."""
    prompt = dataset["prompt"]
    response = dataset["response"]
    is_safe = dataset["is_safe"]
    category_str = []

    for category_dict in dataset["category"]:
        if hint_category:
            cat_str = [k for k, v in category_dict.items() if v]
            if len(cat_str) == 0:
                cat_str = ["safety,_ethics,_and_legality"]
        else:
            cat_str = ["safety,_ethics,_and_legality"]
        category_str.append(cat_str[0])

    return {
        "prompt": prompt,
        "response": response,
        "is_safe": is_safe,
        "category": category_str,
    }


def filter_by_category(dataset, category):
    """Filter the dataset to include only entries that match the specified
    category."""
    filtered_prompt = []
    filtered_response = []
    filtered_is_safe = []
    filtered_category = []

    for i in range(len(dataset["prompt"])):
        if (
            category
            in dataset["category"][i]
            # or dataset["category"][i] == "safety,_ethics,_and_legality"
        ):
            filtered_prompt.append(dataset["prompt"][i])
            filtered_response.append(dataset["response"][i])
            filtered_is_safe.append(dataset["is_safe"][i])
            filtered_category.append(dataset["category"][i])

    return {
        "prompt": filtered_prompt,
        "response": filtered_response,
        "is_safe": filtered_is_safe,
        "category": filtered_category,
    }


def merge_hidden_states_data(hs_files, save_name, balance_data=True):
    # Make sure all hs_files exists
    for hs_file in hs_files:
        assert os.path.exists(hs_file), f"{hs_file} does not exist"

    new_data = {
        "train": {},
        "test": {},
    }

    # Infer category from the first file's name
    first_file = os.path.basename(hs_files[0])
    # Extract category from "hidden_states_<category>.pth"
    data_category = first_file.replace("hidden_states_", "").replace(
        ".pth", ""
    )

    # Load the first file to get the reference size for balancing
    first_file_data = torch.load(hs_files[0], weights_only=False)
    reference_sizes = {
        split: len(next(iter(first_file_data[split].values())))
        for split in ["train", "test"]
    }

    if balance_data:
        print(
            f"Reference sizes for category {data_category}: {reference_sizes}"
        )

    # Process each hidden states file with progress tracking
    for i, hs_file in enumerate(
        tqdm(hs_files, desc="Merging hidden states files", unit="file")
    ):
        data = torch.load(hs_file, weights_only=False)
        for split in ["train", "test"]:
            split_data = data[split]
            reference_size = reference_sizes[split]
            indices = None

            for key, value in split_data.items():
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)

                if balance_data and i > 0:
                    if indices is None:
                        indices = torch.randperm(len(value))[:reference_size]

                    if isinstance(value, torch.Tensor):
                        value = value[indices]
                    elif isinstance(value, list):
                        value = [value[j] for j in indices]

                if key not in new_data[split]:
                    new_data[split][key] = value
                else:
                    if isinstance(value, torch.Tensor):
                        new_data[split][key] = torch.cat(
                            [new_data[split][key], value], dim=0
                        )
                    elif isinstance(value, list):
                        new_data[split][key].extend(value)

    final_filepath = os.path.join(os.path.dirname(hs_files[0]), save_name)
    torch.save({**new_data, "category": data_category}, final_filepath)


def get_categories() -> List[str]:
    """Define list of all categories."""
    return [
        "animal_abuse",
        "child_abuse",
        "controversial_topics,politics",
        "discrimination,stereotype,injustice",
        "drug_abuse,weapons,banned_substance",
        "financial_crime,property_crime,theft",
        "hate_speech,offensive_language",
        "misinformation_regarding_ethics,laws_and_safety",
        "non_violent_unethical_behavior",
        "privacy_violation",
        "self_harm",
        "sexually_explicit,adult_content",
        "terrorism,organized_crime",
        "violence,aiding_and_abetting,incitement",
        "safety,_ethics,_and_legality",
    ]


def run_merge_safe_and_unsafe_hs(
    base_path, exp_path, hs_filename, safe_file_path
):
    file_paths = [f"{i}/{hs_filename}" for i in range(0, 14)]

    all_full_paths = [
        os.path.join(base_path, exp_path, file) for file in file_paths
    ]
    all_full_paths.append(safe_file_path)

    for file in file_paths:
        full_path = os.path.join(base_path, exp_path, file)
        merge_hidden_states_data(
            [full_path, safe_file_path],
            save_name="balanced_new_hidden_states.pth",
            balance_data=True,
        )

    merge_hidden_states_data(
        all_full_paths, save_name="all_hidden_states.pth", balance_data=False
    )


def merge_hidden_states(base_path, dataset_name, model_file_name):
    """Merge hidden states from all categories for a specific model.

    Args:
        base_path: Root directory containing the hidden states files
        dataset_name: Name of the dataset
        model_file_name: Name of the model directory/file (e.g., 'qwen2-1.5b')
    """
    # Get all categories
    categories = get_categories()

    # Construct paths for each category's hidden states file
    hs_files = []
    safe_file = None

    for category in categories:
        hs_path = os.path.join(
            base_path,
            dataset_name,
            model_file_name,
            f"hidden_states_{category}.pth",
        )
        if os.path.exists(hs_path):
            if category == "safety,_ethics,_and_legality":
                safe_file = hs_path
            else:
                hs_files.append(hs_path)
        else:
            print(f"Warning: File not found - {hs_path}")

    if not hs_files:
        raise FileNotFoundError("No hidden states files found")

    # Merge all files
    if hs_files:
        hs_files.append(safe_file)
        merge_hidden_states_data(
            hs_files=hs_files,
            save_name="all_hidden_states.pth",
            balance_data=False,
        )


def balance_merged_hidden_states(input_file, output_file, reference_file):
    print(f"Loading data from {input_file}...")
    data = torch.load(input_file)

    print(f"Loading reference data from {reference_file}...")
    reference_data = torch.load(reference_file, weights_only=False)

    reference_sizes = {
        split: len(next(iter(reference_data[split].values())))
        for split in ["train", "test"]
    }

    balanced_data = {"train": {}, "test": {}}
    category = data["category"]

    for split in ["train", "test"]:
        print(f"Balancing {split} split...")
        split_data = data[split]
        reference_size = reference_sizes[split]
        indices = None

        # Balance each key
        for key, tensor in tqdm(
            split_data.items(), desc=f"Balancing {split} keys"
        ):
            if tensor.size(0) >= 2 * reference_size:
                # Take the first reference_size slices
                first_part = tensor[:reference_size]

                # Randomly sample reference_size from the rest
                remaining = tensor[reference_size:]
                if indices is None:
                    indices = torch.randperm(len(remaining))[:reference_size]
                second_part = remaining[indices]

                # Concatenate the two parts
                balanced_data[split][key] = torch.cat(
                    [first_part, second_part], dim=0
                )
            else:
                # If we don't have enough data, use all available data
                balanced_data[split][key] = tensor

    # Save the balanced data
    print(f"Saving balanced data to {output_file}...")
    torch.save({**balanced_data, "category": category}, output_file)
    print("Balancing complete!")


def balance_all_merged_files(
    base_path,
    exp_path,
    existing_hs_file,
    reference_hs_file,
    balanced_hs_filename,
):

    all_input_files = [
        os.path.join(base_path, exp_path, str(i), existing_hs_file)
        for i in range(14)
    ]
    all_output_files = [
        os.path.join(base_path, exp_path, str(i), balanced_hs_filename)
        for i in range(14)
    ]
    all_reference_files = [
        os.path.join(base_path, exp_path, str(i), reference_hs_file)
        for i in range(14)
    ]

    for input_file, output_file, reference_file in zip(
        all_input_files, all_output_files, all_reference_files
    ):
        print(f"Processing {input_file}...")
        balance_merged_hidden_states(input_file, output_file, reference_file)
