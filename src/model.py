import re
from pathlib import Path
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import DataLoader, Subset
from more_itertools import chunked


BIOBERT_PATH = "/home/is/ujiie/biobert_v1.1_pubmed"

device = "cuda:2" if torch.cuda.is_available() else "cpu"

def filter_pairs(pairs):
    a1, p, a2, n = pairs
    pos_anc = a1.unsqueeze(1)
    neg_anc = a2.unsqueeze(0)
    mask = torch.sum((pos_anc == neg_anc).byte(), dim=0) > 0
    a2 = a2[mask]
    n = n[mask]

    return a1, p, a2, n

def load_bert(path):
    tokenizer = BertTokenizer(path + "/vocab.txt")
    config = BertConfig.from_json_file(path + "/bert_config.json")
    model = BertModel.from_pretrained(path + "/biobert_model.ckpt.index", config=config, from_tf=True)

    return model, tokenizer


class EntityBERT(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def get_entity_reps(self, x, idxs, device, shard_bsz=None):
        if shard_bsz is not None:
            wreps = []
            T = x.size(1)
            xsplits = torch.split(x, shard_bsz, dim=0)
            bidxs = chunked(idxs, shard_bsz)
            bidxs = [torch.tensor([T*i + t for i, j in enumerate(bb) for t in j]).long().to(device) for bb in bidxs]
            for i, xsplit in enumerate(xsplits):
                xsplit = xsplit.to(device)
                mask = (xsplit != 0).long()
                bertrep, _ = self.bert(xsplit, attention_mask=mask)
                bertrep = bertrep.view(-1, bertrep.size(2))
                splitrep = torch.index_select(bertrep, 0, bidxs[i])
                wreps.append(splitrep)

            wreps = torch.cat(wreps, 0)
        else:
            T = x.size(1)
            mask = (x != 0).long()
            bidxs = torch.tensor([T*i + t for i, j in enumerate(idxs) for t in j]).long().to(device)

            bertrep, _ = self.bert(x, attention_mask=mask)
            bertrep = bertrep.view(-1, bertrep.size(2))
            wreps = torch.index_select(bertrep, 0, bidxs)

        return wreps

    def get_CLS_reps(self, x, shard_bsz=None):
        if shard_bsz is not None:
            wreps = []
            T = x.size(1)
            xsplits = torch.split(x, shard_bsz, dim=0)
            for i, xsplit in enumerate(xsplits):
                mask = (xsplit != 0).long()
                bertrep, _ = self.bert(xsplit, attention_mask=mask)
                bertrep = bertrep[:, 0, :]
                wreps.append(bertrep)

            wreps = torch.cat(wreps, 0)
        else:
            T = x.size(1)
            mask = (x != 0).long()
            bertrep, _ = self.bert(x, attention_mask=mask)
            wreps = bertrep[:, 0, :]

        return wreps


class CoherentEntityBERT(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def get_entity_reps(self, x, idxs, device, shard_bsz=None):
        if shard_bsz is not None:
            wreps = []
            T = x.size(1)
            xsplits = torch.split(x, shard_bsz, dim=0)
            bidxs = chunked(idxs, shard_bsz)
            bidxs = [torch.tensor([T*i + t for i, j in enumerate(bb) for t in j]).long().to(device) for bb in bidxs]
            for i, xsplit in enumerate(xsplits):
                mask = (xsplit != 0).long()
                bertrep, _ = self.bert(xsplit, attention_mask=mask)
                bertrep = bertrep.view(-1, bertrep.size(2))
                splitrep = torch.index_select(bertrep, 0, bidxs[i])
                wreps.append(splitrep)

            wreps = torch.cat(wreps, 0)
        else:
            T = x.size(1)
            mask = (x != 0).long()
            bidxs = torch.tensor([T*i + t for i, j in enumerate(idxs) for t in j]).long().to(device)

            bertrep, _ = self.bert(x, attention_mask=mask)
            bertrep = bertrep.view(-1, bertrep.size(2))
            wreps = torch.index_select(bertrep, 0, bidxs)

        return wreps

    def get_coherent_reps(self, x, idxs, is_train=True):
        seq_size, dim = x.size()
        C = []
        cor_idxs = []
        cnt = 0
        cor_cnt = 0
        for idx in idxs:
            cor_idxs.extend([cor_cnt for i in range(len(idx))])
            cor_cnt += 1

            tmp = [0] * cnt + [1/len(idx) for i in range(len(idx))]
            tmp += [0] * (seq_size - len(tmp))

            if is_train:
                C.append(tmp)
            else:
                tmp = [tmp] * len(idx)
                C.extend(tmp)

            cnt += len(idx)

        C = torch.tensor(C).type_as(x)

        return torch.mm(C, x), cor_idxs


class FaissIndexer(object):
    def __init__(self, index=None, emb_dim=None, labels=None, is_gpu=False):
        import faiss as faiss_module

        self.faiss_module = faiss_module
        self.index = index
        self.emb_dim = emb_dim
        self.labels = labels
        self.is_gpu = is_gpu

    def train_index(self, vectors, labels):
        self.emb_dim = len(vectors[0])
        self.cpu_index = self.faiss_module.IndexFlatIP(self.emb_dim)
        if self.is_gpu:
            res = self.faiss_module.StandardGpuResources()
            flat_config = self.faiss_module.GpuIndexFlatConfig()
            flat_config.device = 1
            self.index = self.faiss_module.GpuIndexFlatIP(res, self.emb_dim, flat_config) 
            #self.index = self.faiss_module.index_cpu_to_all_gpus(self.faiss_module.StandardGpuResources(), 0, self.index)
            #self.index = self.faiss_module.index_cpu_to_all_gpus(self.cpu_index)
        else:
            self.index = self.cpu_index

        self.labels = labels
        self.index.add(np.array(vectors, dtype=np.float32))

    def search_nn(self, query_batch, k):
        query_batch = torch.split(query_batch, 64)
        batch_tags = []
        for batch in query_batch:
            D, I = self.index.search(batch.cpu().numpy(), k)
            batch_tag = [[self.labels[l] for l in ll] for ll in I]
            batch_tags.extend(batch_tag)

        return batch_tags

    def search_nn_with_score(self, query_batch, k):
        query_batch = torch.split(query_batch, 64)
        batch_tags = []
        batch_scores = np.zeros((1, k))
        for batch in query_batch:
            D, I = self.index.search(batch.cpu().numpy(), k)
            batch_tag = [[self.labels[l] for l in ll] for ll in I]
            batch_tags.extend(batch_tag)
            batch_scores = np.vstack([batch_scores, D])

        return batch_tags, batch_scores[1:, :]

if __name__ == '__main__':
    itoc, itod, d_num = load_concept_vocab('/home/is/ujiie/NCBID/concept_vocab.csv')
    #itoc, itod, d_num = load_concept_vocab('/home/is/ujiie/NCBID/concept_vocab_plus_train.csv')
    trans = SpanNormTransformer(itoc)
    bert, tokenizer = load_bert(BIOBERT_PATH)

    data = model.get_sparse_scores(diseases, torch.tensor([[[0,0]] for i in range(len(diseases))]))
    score, id = torch.max(data, dim=-1)
    preds = id.reshape(-1)
    cnt = 0
    total = 0
    for code, pred in zip(codes, preds):
        pred = trans.itot[pred.item()]
        if pred == code:
            cnt += 1
        total += 1

    print(cnt / total, cnt, total)

