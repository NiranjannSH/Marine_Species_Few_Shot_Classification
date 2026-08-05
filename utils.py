import os
import sys
import time
import random
import logging
import copy
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from thop import profile
from config import CHECKPOINT_DIR

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
LOG_FILE_PATH = os.path.join(CHECKPOINT_DIR, f"Proposed_parallel_execution_log_{timestamp}.txt")
TABLE_FILE_PATH = os.path.join(CHECKPOINT_DIR, f"Proposed_final_paper_table_{timestamp}.txt")

def log_to_file(message):
    current_ts = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{current_ts}] {message}"
    with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")
    print(formatted_msg)
    sys.stdout.flush()

def save_table_to_file(table_string):
    with open(TABLE_FILE_PATH, "w", encoding="utf-8") as f:
        f.write(table_string)

def set_seed(seed_val):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def freeze_batchnorm_stats(module):
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()

def mean_confidence_interval(data, confidence=0.95):
    values = np.asarray(data, dtype=np.float64)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    ci = scipy.stats.sem(values) * scipy.stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1)
    return mean, float(ci)

def compute_model_macs(model, input_tensor):
    was_training = model.training
    model.eval()
    
    thop_logger = logging.getLogger("thop")
    original_level = thop_logger.getEffectiveLevel()
    thop_logger.setLevel(logging.ERROR)
    
    with torch.no_grad():
        macs, _ = profile(copy.deepcopy(model), inputs=(input_tensor,), verbose=False)
        
    thop_logger.setLevel(original_level)
    if was_training:
        model.train()
        
    return macs