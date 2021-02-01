import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize

from data import SentDataset, my_collate_fn_for_sent
from model import EntityBERT


def parse_args():
    parser = argparse.ArgumentParser(description='Precompute the embeddings')

    parser.add_argument('--dictionary_path', required=True, help='Dictionary path')
    parser.add_argument('--concept_map', required=True, help='File path of concept mapping')
    parser.add_argument('--input_dir', required=True, help='Input directory path')
    parser.add_argument('--output_dir', required=True, help='Output directory. Note that this directory needs at least 35G memory space')

    args = parser.parse_args()

    return args


def get_all_embed(model, dataset):
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=my_collate_fn_for_sent)

    result_tensor = []
    result_tags = []
    result_sents = []
    all_sent = set()
    for ent_idxs, tokens, tags in dataloader:
        sents = [" ".join([str(t) for t in token]) for token in tokens]
        new_tokens = []
        new_tags = []
        new_ent_idxs = []
        for sent, token, tag, ent_idx in zip(sents, tokens, tags, ent_idxs):
            if sent not in all_sent:
                all_sent.add(sent)
                new_tokens.append(token)
                new_tags.append(tag)
                new_ent_idxs.append(ent_idx)
        tokens = new_tokens
        tags = new_tags
        ent_idxs = new_ent_idxs

        inputs = pad_sequence([torch.LongTensor(token)
                                for token in tokens], padding_value=0).t().to(device)

        embeddings = model.get_entity_reps(inputs, ent_idxs, device).half()

        result_tensor.extend([t.cpu().tolist() for t in embeddings])

        ori_tokens = [tokenizer.convert_ids_to_tokens(t) for t in tokens]
        tags = [{"label": t, "sent": " ".join(o_t)} for tag, o_t in zip(tags, ori_tokens) for t in tag if t != []]
        result_tags.extend(tags)

    return result_tensor, result_tags


def save_all_embed(model, dataset, base_dir):
    result_tensor = []
    result_tags = []
    result_sents = []
    save_cnt = 0
    total = 0
    for ent_idxs, tokens, tags in tqdm(dataset):
        inputs = pad_sequence([torch.LongTensor(token)
                                for token in tokens], padding_value=0).t().to(device)

        embeddings = model.get_entity_reps(inputs, ent_idxs, device)

        result_tensor.extend([t.cpu().tolist() for t in embeddings])

        ori_tokens = [tokenizer.convert_ids_to_tokens(t) for t in tokens]
        tags = [','.join(t) + "\t" + " ".join(o_t) for tag, o_t in zip(tags, ori_tokens) for t in tag if t != []]
        result_tags.extend(tags)

        if len(result_tags) > 500000:
            result_tensor = np.array(result_tensor, dtype=np.float32)
            result_tensor = normalize(torch.tensor(result_tensor, requires_grad=False), p=2, dim=-1).numpy()

            np.save(str(base_dir / ("embedding" + str(save_cnt))), result_tensor)
            with open(base_dir / ("label" + str(save_cnt) + ".tsv"), "w") as f:
                f.write("\n".join(result_tags))

            save_cnt += 1
            total += len(result_tags)
            result_tensor = []
            result_tags = []

    if len(result_tags) > 0:
        result_tensor = np.array(result_tensor, dtype=np.float32)
        result_tensor = normalize(torch.tensor(result_tensor, requires_grad=False), p=2, dim=-1).numpy()

        np.save(str(base_dir / ("embedding" + str(save_cnt))), result_tensor)
        with open(base_dir / ("label" + str(save_cnt) + ".tsv"), "w") as f:
            f.write("\n".join(result_tags))


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]', '[/ENT]']})
    bert = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model = EntityBERT(bert).to(device)

    with open(args.concept_map, 'r') as f:
        concept_map = json.load(f)

    with open(args.dictioanry_path, 'r') as f:
        disease_list = [json.loads(l)[0] for l in f.read().split('\n') if l != '']

    train_dataset = SampleSentDataset(args.input_dir, tokenizer, from_jsonl=True, concept_map=concept_map, disease_list=disease_list)
    ite_data = train_dataset.__sample_all__(only=False)
    save_all_embed(model, ite_data, args.output_dir)

