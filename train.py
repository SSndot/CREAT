import torch
import logging
import os
from torch.utils.data import TensorDataset, DataLoader
from Attack import Generator, GeneratorModel
from Rec_models.utils import *
from utils import *
from config import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up training environment and device."""
    device = DEVICE if CUDA else "cpu"
    logger.info(f"Using device: {device}")
    if CUDA and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return device


def load_dataset():
    logger.info(f"Loading dataset: {DATASET}")
    dataset_path = os.path.join(DATASET_PATH, f"{DATASET}.csv")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    user_seq, max_item, long_sequence, _ = get_user_seqs_long(dataset_path)
    logger.info(f"Dataset loaded: {len(user_seq)} sequences, max_item={max_item}")
    
    return user_seq, max_item


def prepare_model(device):
    logger.info(f"Initializing models: {TARGET_MODEL}")
    
    # Set up args for target model
    args.num_items = 0  # Will be updated after loading dataset
    set_template(args, DATASET, TARGET_MODEL)
    
    # Load target recommendation model
    model_path = os.path.join(MODEL_PATH, f"{TARGET_MODEL}_{DATASET}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    rec_model_helper = create_rec_model(TARGET_MODEL, model_path, args, device)
    logger.info(f"Recommendation model loaded from {model_path}")
    
    # Initialize generator model
    gen_model = GeneratorModel(
        seq_emb_size=args.seq_emb_size,
        item_emb_size=ITEM_EMB_SIZE,
        hidden_size=GEN_HIDDEN_SIZE,
        max_seq_len=args.bert_max_len,
        pos_emb_size=POS_EMB_SIZE,
        dropout_rate=DROPOUT_RATE,
        num_layers=NUM_LAYERS,
        device=device
    )
    gen_model = gen_model.to(device)
    logger.info(f"Generator model initialized on {device}")
    
    return rec_model_helper, gen_model, args


def prepare_training_data(user_seq):
    logger.info(f"Preparing training data with attack_ratio={ATTACK_RATIO}")
    
    # Pad sequences
    user_seq_padded = padding(user_seq, args.bert_max_len)
    train_samples = torch.Tensor(user_seq_padded).type(torch.long)
    
    # Split into attack and remain samples
    attack_num = int(ATTACK_RATIO * len(train_samples))
    random_indices = torch.randperm(train_samples.shape[0])
    attack_indices = random_indices[:attack_num]
    attack_samples = train_samples[attack_indices]
    
    logger.info(f"Attack samples: {len(attack_samples)}, Remain samples: {len(train_samples) - len(attack_samples)}")
    
    # Create dataloader
    dataset = TensorDataset(attack_samples)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    return dataloader


def train_attack_phase(gen, dataloader, device):
    logger.info(f"Starting attack training phase ({ATTACK_TRAIN_EPOCHS} epochs)")
    
    for epoch in range(ATTACK_TRAIN_EPOCHS):
        for batch_idx, batch in enumerate(dataloader):
            inp = batch[0].to(device)
            gen.get_group(inp)
            
            for _ in range(TRAIN_STEP):
                gen.old_train()
        
        logger.info(f"Attack Training - Epoch {epoch + 1}/{ATTACK_TRAIN_EPOCHS} completed")


def train_adversarial_phase(gen, dataloader, device):
    logger.info(f"Starting adversarial training phase ({ADV_TRAIN_EPOCHS} epochs)")
    
    for epoch in range(ADV_TRAIN_EPOCHS):
        for batch_idx, batch in enumerate(dataloader):
            inp = batch[0].to(device)
            gen.get_group(inp, is_adv=True)
            
            for _ in range(TRAIN_STEP):
                gen.train(is_adv=True)
        
        logger.info(f"Adversarial Training - Epoch {epoch + 1}/{ADV_TRAIN_EPOCHS} completed")


def main():
    try:
        logger.info("="*80)
        logger.info("CREAT Training Started")
        logger.info("="*80)
        logger.info(f"Configuration: Dataset={DATASET}, Model={TARGET_MODEL}, "
                   f"Epochs={ATTACK_TRAIN_EPOCHS}+{ADV_TRAIN_EPOCHS}, Batch={BATCH_SIZE}")
        
        # Setup
        device = setup_environment()
        
        # Load data
        user_seq, max_item = load_dataset()
        
        # Prepare models
        rec_model_helper, gen_model, args_obj = prepare_model(device)
        args.num_items = max_item
        
        # Create generator
        gen = Generator(
            model=gen_model,
            model_helper=rec_model_helper,
            max_attack_num=MAX_ATTACK_NUM,
            target_item=TARGET_ITEM,
            max_seq_len=args.bert_max_len,
            device=device
        )
        logger.info(f"Generator initialized with target_item={TARGET_ITEM}")
        
        # Prepare training data
        dataloader = prepare_training_data(user_seq)
        
        # Training phases
        train_attack_phase(gen, dataloader, device)
        train_adversarial_phase(gen, dataloader, device)
        
        logger.info("="*80)
        logger.info("Training completed successfully!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
