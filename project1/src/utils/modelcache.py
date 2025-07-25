from copy import copy
import json
import os
import torch
from os.path import dirname


def save_results(model, train_cfg, history, folder_name):
  # Make directory inside results
  root_path = dirname(dirname(dirname(dirname(os.path.realpath(__file__)))))
  folder_path = os.path.join(root_path, "results", "proj1", folder_name)
  os.makedirs(folder_path, exist_ok=True)
  # Write train_cfg in a json file
  with open(os.path.join(folder_path, "train_cfg.json"), "w") as f:
    trcfg_cpy = copy(train_cfg)
    trcfg_cpy.transforms = [str(tr) for tr in trcfg_cpy.transforms]
    json.dump(vars(trcfg_cpy), f, indent=2)
  # Write results (history) in another json
  with open(os.path.join(folder_path, "history.json"), "w") as f:
    json.dump(history, f, indent=2)
  # Save model weights
  torch.save(model.state_dict(), os.path.join(folder_path, "weights.pth"))