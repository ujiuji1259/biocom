import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize

from data import load_concept_vocab, SentDataset, my_collate_fn_for_sent
from model import EntityBERT, FaissIndexer


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description='Precompute the embeddings')

    parser.add_argument('--concept_map', required=True, help='File path of concept mapping')
    parser.add_argument('--input_dir', required=True, help='Input directory path')
    parser.add_argument('--learning_rate', type=int, help='learning rate')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--model_path', type=int, help='model save path')

    args = parser.parse_args()

    return args


def train(model, dataset, batch_size, ctoi, lr):
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate_fn_for_sent, shuffle=True)

    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    loss_func = losses.MultiSimilarityLoss(alpha=2, beta=50)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    all_loss = 0
    steps = 0

    for ent_idxs, tokens, tag in tqdm(dataloader):
        steps += 1
        optimizer.zero_grad()
        tag = [tt for t in tag for tt in t if tt != []]
        labels = torch.tensor([ctoi[t[0]] for t in tag])

        inputs = pad_sequence([torch.LongTensor(token)
                                for token in tokens], padding_value=0).t()

        embeddings = model.get_entity_reps(inputs, ent_idxs, device, shard_bsz=16)

        hard_pairs = miner(embeddings, labels)
        hard_pairs = filter_pairs(hard_pairs)
        loss = loss_func(embeddings, labels, hard_pairs)

        all_loss += loss.item()
        loss.backward()
        optimizer.step()

    return all_loss / steps


if __name__ == '__main__':
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]', '[/ENT]']})
    bert = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = EntityBERT(bert).to(device)

    with open(args.concept_map, 'r') as f:
        concept_map = json.load(f)
    itoc = list(concept_map.keys())
    ctoi = {c:i for i, c in enumerate(itoc)}

    base_valid_tags = set()
    for k, v in concept_map.items():
        base_valid_tags.add(k)
        [base_valid_tags.add(vv) for vv in v]

    train_dataset = SentDataset(args.input_dir, tokenizer, from_jsonl=True, concept_map=concept_map)

    train_dataset.__sample__(minimum=50, only=False)
    loss_ = train(model, train_dataset, args.batch_size, ctoi, args.learning_rate)
    torch.save(model.state_dict(), args.model_path)

