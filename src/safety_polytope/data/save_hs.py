import logging
import os

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from safety_polytope.common.load_util import load_model_and_tokenizer
from safety_polytope.data.safety_data import (
    format_prompt_response_plain,
    get_dataset,
    get_safety_dataloader,
)
from safety_polytope.polytope.lm_constraints import PolytopeConstraint

log = logging.getLogger("polytope")


def get_hidden_states_dict(cfg, data, model, tokenizer):
    dataloader = get_safety_dataloader(
        cfg.dataset.name_or_path,
        data,
        format_fn=format_prompt_response_plain,
        model="",
        batch_size=cfg.batch_size,
        shuffle=False,
        dataset_type=cfg.dataset.dataset_type,
    )

    hidden_states_list = []
    labels_list = []
    input_texts_list = []

    for batch in tqdm(dataloader, desc="Processing Batches"):
        text, label = batch
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(model.device)

        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True)
            layer_hs = output.hidden_states[cfg.layer_number]
            last_token_hs = layer_hs[:, -1, :]

        hidden_states_list.append(last_token_hs.cpu().numpy())
        labels_list.append(label.cpu().numpy())
        input_texts_list.extend(text)

        if len(hidden_states_list) * cfg.batch_size >= cfg.total_datapoints:
            break

    result = {
        "hidden_states": np.concatenate(hidden_states_list, axis=0),
        "labels": np.concatenate(labels_list, axis=0),
        "input_texts": input_texts_list,
    }
    return result


@hydra.main(
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="interpret_config",
    version_base="1.1",
)
def main(cfg: DictConfig):
    train_data, test_data = get_dataset(cfg.dataset.name_or_path, cfg.dataset)

    model, tokenizer = load_model_and_tokenizer(cfg.model_path)
    safety_model = PolytopeConstraint(model, tokenizer)

    safety_model.to(model.device)

    if "Ministral-8B" in cfg.model_path:
        model_name = "ministral-8b"
    elif "llama-2-7b" in cfg.model_path:
        model_name = "llama2-7b"
    elif "Qwen2-1.5B" in cfg.model_path:
        model_name = "qwen2-1.5b"
    elif "Llama-3.1-8B" in cfg.model_path:
        model_name = "llama3-8b"
    elif "Qwen3-8B" in cfg.model_path:
        model_name = "qwen3-8b"
    else:
        raise NotImplementedError(
            f"Please mannually configure a model name for {cfg.model_path}."
        )
    save_dir = f"{cfg.save_hs_root_dir}/{cfg.dataset.short_name}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    category = cfg.dataset.category or "all"
    pth_save_path = os.path.join(
        save_dir,
        f"hidden_states_{category}.pth",
    )

    train_dict = get_hidden_states_dict(cfg, train_data, model, tokenizer)
    test_dict = get_hidden_states_dict(cfg, test_data, model, tokenizer)

    torch.save({"train": train_dict, "test": test_dict}, pth_save_path)
    log.info(f"Hidden states, labels, and inputs saved to {pth_save_path}")


if __name__ == "__main__":
    main()
