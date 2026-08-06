import os
import torch

# Set matmul precision to prevent TF32 rounding variances
torch.set_float32_matmul_precision('highest')

SEED = 42

# Dynamic Portable Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Datasets Configuration
DATASETS_CONFIG = [
    {
        "name": "Croatian",
        "root": os.path.join(DATA_DIR, "croatian"),
        "train_way": 3, "val_way": 2, "test_way": 3,
        "train_classes_cnt": 7, "val_classes_cnt": 2
    },
    {
        "name": "Fish4Knowledge",
        "root": os.path.join(DATA_DIR, "fish4knowledge"),
        "train_way": 5, "val_way": 2, "test_way": 5,
        "train_classes_cnt": 16, "val_classes_cnt": 2
    },
    {
        "name": "Aqua20",
        "root": os.path.join(DATA_DIR, "aqua20"),
        "train_way": 5, "val_way": 2, "test_way": 5,
        "train_classes_cnt": 12, "val_classes_cnt": 2
    }
]

# Image Parameters
IMAGE_SIZE = 84
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Stage 1: SupCon Pre-training
S1_BATCH_SIZE = 128
S1_EPOCHS = 100
S1_LR = 3e-4
S1_WEIGHT_DECAY = 1e-4
S1_TEMPERATURE = 0.04

# Stage 2: Test-Time Adaptation
K_SHOTS = [1, 5]
N_QUERY = 5
ADAPT_STEPS = 25
ADAPT_LR = 0.1
ADAPT_BACKBONE = True
FINAL_TEST_EPISODES = 600
SAVE_STAGE_2_MODELS = False
