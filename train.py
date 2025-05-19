import torch
from math import ceil
import sys
import os
from torch.utils.data import TensorDataset, DataLoader
from generator_mask import Generator, GeneratorModel
from Rec_models.utils import *
from utils import *
import logging

CUDA = True

def main():
    dataset = "ml-1m"
    target_model = "bert"
    if CUDA: device = "cuda:0"
    else: device = "cpu"

    target_item = get_attack_item(dataset)
    dataset_path = os.path.join("./dataset", "clean_dataset", f"{dataset}.csv")

    user_seq, max_item, long_sequence, _ = get_user_seqs_long(dataset_path)
    args.num_items = max_item
    set_template(args, dataset, target_model)
    user_seq = padding(user_seq, args.bert_max_len)
    train_samples = torch.Tensor(user_seq).type(torch.long)

    model_path = "./Rec_models/trained_model/" + target_model + "_" + dataset + ".pth"
    rec_model_helper = create_rec_model(target_model, model_path, args, device)

    gen_model = GeneratorModel(args.seq_emb_size, ITEM_EMB_SIZE, GEN_HIDDEN_SIZE, max_seq_len=args.bert_max_len, device=device)
    gen_model = gen_model.to(device)

    gen = Generator(gen_model, rec_model_helper, max_attack_num=5, target_item=target_item, max_seq_len=args.bert_max_len, device=device)

    attack_ratio = 0.01
    attack_num = int(attack_ratio * len(train_samples))
    random_indices = torch.randperm(train_samples.shape[0])
    attack_indices, remain_indices = random_indices[:attack_num], random_indices[attack_num:]
    attack_samples, remain_samples = train_samples[attack_indices], train_samples[remain_indices]

    dataset = TensorDataset(attack_samples)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Attack Training
    for epoch in range(ATTACK_TRAIN_EPOCHS):
        for batch in dataloader:
            inp = batch[0]
            gen.get_group(inp)

            for j in range(TRAIN_STEP):
                gen.old_train()

    # Adversarial Training
    for epoch in range(ADV_TRAIN_EPOCHS):
        for batch in dataloader:
            inp = batch[0]
            gen.get_group(inp, is_adv=True)

            for j in range(TRAIN_STEP):
                gen.train(is_adv=True)

if __name__ == '__main__':
    main()
