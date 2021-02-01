import json
import os
import pdb
from collections import Counter
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
import mlflow
from torchsummary import summary

from data import EntityDataset, load_concept_vocab, my_collate_fn, SentTrainDataset, SentDataset, my_collate_fn_for_sent, SentEntityDataset, SampleSentDataset
from model import EntityBERT, load_bert, FaissIndexer, filter_pairs


device = "cuda" if torch.cuda.is_available() else "cpu"


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

        mlflow.log_metric("training loss", loss.item(), step=steps)

        all_loss += loss.item()
        loss.backward()
        optimizer.step()

    return all_loss / steps


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]', '[/ENT]']})
    bert = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = EntityBERT(bert).to(device)

    with open("dataset/concept_map.json", 'r') as f:
        concept_map = json.load(f)
    itoc = list(concept_map.keys())
    ctoi = {c:i for i, c in enumerate(itoc)}

    base_valid_tags = set()
    for k, v in concept_map.items():
        base_valid_tags.add(k)
        [base_valid_tags.add(vv) for vv in v]

    with open('dataset/disease_down_half.tsv', 'r') as f:
        disease_list = [json.loads(l)[0] for l in f.read().split('\n') if l != '']

    train_dataset = SampleSentDataset('/data1/ujiie/pubmed/pubmed_CTD_wo_stopwords', tokenizer, from_jsonl=True, concept_map=concept_map, disease_list=disease_list)

    datasets = {
            "ncbi": ["dataset/dev_NCBID.jsonl", "dataset/test_NCBID.jsonl", "pred_NCBID.json", 4],
            "bc5cdr": ["dataset/dev_BC5CDR.jsonl", "dataset/test_BC5CDR.jsonl", "pred_BC5CDR.json", 0],
            "medmentions": ["dataset/dev_medmentions.jsonl", "dataset/test_medmentions.jsonl", "pred_medmentions.json", 1]
            }

    train_dataset.__sample__(minimum=50, only=False)
    loss_ = train(model, train_dataset, 16, ctoi, 1e-5)



