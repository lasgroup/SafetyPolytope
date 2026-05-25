import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_representations(rep_save_dir):
    # Walk through the directory and load all representations
    directions = {}
    signs = {}
    for root, _, files in os.walk(rep_save_dir):
        for file in files:
            if file.endswith("directions.pth"):
                rep = torch.load(os.path.join(root, file))
                category = file[: file.index("_directions.pth")]
                directions[category] = rep
            elif file.endswith("signs.pth"):
                sign = torch.load(os.path.join(root, file))
                category = file[: file.index("_signs.pth")]
                signs[category] = sign
    return directions, signs


def load_model_and_tokenizer(model_path, **kwargs):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        **kwargs,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        # use_fast=False, padding_side="left", padding="max_length"
    )
    # FIXED: Handle cases where both pad_token and unk_token are None
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Last resort: add a new pad token
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from sklearn.utils import check_random_state

    check_random_state(seed)
