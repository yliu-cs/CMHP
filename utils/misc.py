import os
import random
import warnings

import torch
import numpy as np
import transformers
import torch.nn as nn


def ignore_warnings():
    warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity_error()


def seed_everything(
    seed: int
):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_model_size(
    model: nn.Module
):
    param_size, param_num = 0, 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_num += param.nelement()
    buffer_size, buffer_num = 0, 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_num += buffer.nelement()
    total_size = (param_size + buffer_size) / 1024 ** 2
    total_num = (param_num + buffer_num) / 10 ** 6
    return f"Model Size={total_size:.2f}MB & Parameter number={total_num:.2f}M"