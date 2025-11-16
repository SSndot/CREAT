# ============================================================================
# Dataset Configuration
# ============================================================================
DATASET = "ml-1m"
DATASET_PATH = "./dataset/clean_dataset"

# ============================================================================
# Model Configuration
# ============================================================================
TARGET_MODEL = "bert"
MODEL_PATH = "./Rec_models/trained_model"

# Device
CUDA = True
DEVICE = "cuda:0" if CUDA else "cpu"

# ============================================================================
# Embedding and Hidden Dimensions
# ============================================================================
SEQ_EMB_SIZE = 64  # Sequence embedding size
ITEM_EMB_SIZE = 64  # Item embedding size
GEN_HIDDEN_SIZE = 128  # Generator model hidden size
POS_EMB_SIZE = 64  # Position embedding size

# ============================================================================
# BERT Model Configuration
# ============================================================================
BERT_MAX_LEN = 50  # Maximum sequence length for BERT

# ============================================================================
# Attack Configuration
# ============================================================================
MAX_ATTACK_NUM = 5  # Maximum number of items to attack per sequence
ATTACK_RATIO = 0.01  # Ratio of samples to attack
TARGET_ITEM = 1  # Target item ID to inject into sequences

# ============================================================================
# Training Configuration
# ============================================================================
BATCH_SIZE = 64  # Batch size for training
ATTACK_TRAIN_EPOCHS = 10  # Number of epochs for attack training
ADV_TRAIN_EPOCHS = 10  # Number of epochs for adversarial training
TRAIN_STEP = 1  # Number of training steps per batch

# ============================================================================
# Generator Model Configuration
# ============================================================================
# Generator optimizer
LEARNING_RATE = 1e-6  # Learning rate for Adam optimizer
CLIP_EPSILON = 0.2  # Epsilon for PPO clipping
GROUP_NUM = 16  # Number of groups for data processing
DROPOUT_RATE = 0.1  # Dropout rate in generator model
NUM_LAYERS = 2  # Number of GRU layers

# ============================================================================
# Reward Configuration
# ============================================================================
# Optional: Add reward scaling factors here if needed
# PATTERN_REWARD_WEIGHT = 1.0
# DPP_REWARD_WEIGHT = 1.0
# OT_REWARD_WEIGHT = 1.0

# ============================================================================
# OT (Optimal Transport) Configuration
# ============================================================================
OT_MAX_ITER = 20  # Maximum iterations for optimal transport

# ============================================================================
# Lambda Calculation Configuration (for adversarial training)
# ============================================================================
K_PERCENTILE = 0.5  # Percentile (0-1) for constraint threshold calculation. e.g., 0.5 means 50th percentile

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_LEVEL = "INFO"
