import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import csv
from collections import Counter
from Rec_models.utils import *

def get_grpo_rewards(tensor):
    """ tensor: [group_num x time_step x batch_size x 1] """
    return (tensor - torch.mean(tensor, dim=0, keepdim=True)) / (torch.std(tensor, dim=0, keepdim=True) + 1e-7)

def t_padding(seq, max_seq_length):
    pad_len = max_seq_length - len(seq)
    if pad_len > 0:
        return F.pad(seq, (0, pad_len), "constant", 0)
    else:
        return seq[:max_seq_length]


def padding(user_seq, max_seq_length):
    user_seqs = []
    for s in user_seq:
        pad_len = max_seq_length - len(s)
        s = s + [0] * pad_len
        s = s[:max_seq_length]
        user_seqs.append(s)
    return user_seqs


def unpadding(fake_seq, start_id):
    #tensor to list
    fake_seqs = []
    id = start_id
    for fake_data in fake_seq:
        seq = []
        seq.append(id)
        for i, f in enumerate(fake_data):
            if f != 0:
                seq.append(f)
            else:
                # clip when there are two consecutive "0"
                if i == (len(fake_data) - 1) or fake_data[i+1] == 0:
                    break
                else:
                    continue
        fake_seqs.append(seq)
        id += 1
    return fake_seqs


def read_file(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lis = []
    for line in lines:
        l = line.strip().split(' ')
        l = [int(s) for s in l]
        lis.append(l)
    return lis


def get_user_seqs_long(data_file):
    user_seq = []
    long_sequence = []
    test = []
    item_set = set()
    with open(data_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            seq = eval(row[1])
            if len(seq) < 3: continue
            items = seq[:-1]
            test_item = seq[-1]
            long_sequence.extend(items)
            user_seq.append(items)
            test.append(test_item)
            item_set = item_set | set(items)
        max_item = max(item_set)

    return user_seq, max_item, long_sequence, test

def get_lengths(seq):
    seq = torch.Tensor(seq)
    if seq.dim() == 1:
        length = torch.count_nonzero(seq).type(torch.LongTensor)
        return max(1, length)
    else:
        lengths = torch.count_nonzero(seq, dim=1).type(torch.LongTensor)
        lengths[lengths == 0] = 1
        return lengths

def separate_targets_from_datas(datas):
    lengths = get_lengths(datas)
    target_index = torch.Tensor([train_seq_len - 1 for train_seq_len in lengths]).unsqueeze(-1).type(
        torch.LongTensor).to(datas.device)
    targets = datas.gather(1, target_index).squeeze(-1)
    for idx, length in enumerate(lengths):
        datas[idx, length - 1] = 0
    return targets

def create_rec_model(target_model, model_path, args, device):
    model_dict = {
        "narm": (NARM, NARMHelper),
        "bert": (BERT, BERTHelper)
    }

    model_class, helper_class = model_dict.get(target_model)

    if model_class is None:
        raise ValueError(f"Unsupported target model: {target_model}")

    rec_model = model_class(args)
    rec_model.load_state_dict(torch.load(model_path, map_location=device).get('model_state_dict'))
    rec_model.eval()
    rec_model_helper = helper_class(rec_model, args, device)

    return rec_model_helper