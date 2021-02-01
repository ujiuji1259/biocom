import json
import random
import csv
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
from torch.utils.data import Dataset

def load_concept_vocab(path):
    itoc = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            concept = row[1].replace('MESH:', '').replace('OMIM:', '')
            itoc.append(concept)

    return itoc


class SentTrainDataset(object):
    def __init__(self, fn, tokenizer, only=False, concept_set=None):
        self.tokenizer = tokenizer
        self.concept_set = concept_set
        self._read(fn, only)

    def _read(self, fn, only):
        self.data = []
        with open(fn, 'r') as f:
            for line in f:
                line = line.rstrip()
                if not line:
                    continue

                line = json.loads(line)
                if only:
                    for entity in line['entities']:
                        self.data.append({"tokens": line['tokens'].copy(), "entities": [entity.copy()]})
                else:
                    self.data.append(line)

    def __preprocess__(self, sent):
        ent_idxs, wpcs, tags = [], ['[CLS]'], []
        curr_start = 1

        tokens = sent['tokens']
        ents = sorted(sent['entities'], key=lambda s: s['span'][0])
        ent_cnt = 0
        for idx, word in enumerate(tokens):
            if len(wpcs) + 2 > 510:
                break

            if ent_cnt < len(ents) and idx == ents[ent_cnt]['span'][1]:
                wpcs.append('[/ENT]')
                ent_cnt += 1
            if ent_cnt < len(ents) and idx == ents[ent_cnt]['span'][0]:
                concepts = ents[ent_cnt]['descriptor'].replace('MESH:', '').replace('OMIM:', '').split('|')
                if self.concept_set is None or any(c in self.concept_set for c in concepts):
                    ent_idxs.append(len(wpcs))
                    tags.append(ents[ent_cnt]['descriptor'].replace('MESH:', '').replace('OMIM:', '').split('|'))
                wpcs.append('[ENT]')
            wpcs.append(word)

        if ent_cnt < len(ents) and idx + 1 == ents[ent_cnt]['span'][1]:
            wpcs.append('[/ENT]')

        wpcs.append('[SEP]')
        wpcs = self.tokenizer.convert_tokens_to_ids(wpcs)
        
        return ent_idxs, wpcs, tags

    def __getitem__(self, idx):
        ent_idxs, tokens, tags = self.__preprocess__(self.data[idx])

        return [ent_idxs], [tokens], [tags]

    def __len__(self):
        return len(self.data)


class SentDataset(Dataset):
    def __init__(self, fn, tokenizer, from_jsonl=False, concept_map=None):
        self.tokenizer = tokenizer
        self.base_dir = Path(fn)
        self.files = list(self.base_dir.glob("*.jsonl"))
        self.__load_summary__()
        self.concept_map = concept_map

    def __preprocess__(self, sent):
        ent_idxs, wpcs, tags = [], ['[CLS]'], []
        curr_start = 1

        tokens = sent['tokens']
        ents = sorted(sent['entities'], key=lambda s: s['span'][0])
        ent_cnt = 0
        for idx, word in enumerate(tokens):
            if len(wpcs) + 2 > 510:
                break

            if ent_cnt < len(ents) and idx == ents[ent_cnt]['span'][1]:
                wpcs.append('[/ENT]')
                ent_cnt += 1
            if ent_cnt < len(ents) and idx == ents[ent_cnt]['span'][0]:
                ent_idxs.append(len(wpcs))
                wpcs.append('[ENT]')
                if self.concept_map is not None:
                    tags.append(self.concept_map[ents[ent_cnt]['descriptor']])
                else:
                    tags.append(ents[ent_cnt]['descriptor'])
            wpcs.append(word)

        if ent_cnt < len(ents) and idx + 1 == ents[ent_cnt]['span'][1]:
            wpcs.append('[/ENT]')

        wpcs.append('[SEP]')
        wpcs = self.tokenizer.convert_tokens_to_ids(wpcs)
        
        return ent_idxs, wpcs, tags

    def __load_summary__(self):
        summary = self.base_dir / "summary"
        self.code2sent = {}
        if not summary.exists():
            summary.mkdir()

            for fn in tqdm(self.files):
                code = fn.stem
                self.code2sent[code] = defaultdict(list)
                with open(str(fn), 'r') as f:
                    for idx, line in enumerate(f):
                        line = line.rstrip()
                        if not line:
                            continue

                        line = json.loads(line)
                        entities = line['entities']
                        for entity in entities:
                            if entity['descriptor'][0] == code:
                                self.code2sent[code][entity['string']].append(idx)

                code_file = summary / fn.name
                with open(str(code_file), 'w') as g:
                    json.dump(self.code2sent[code], g, ensure_ascii=False)
        else:
            for fn in self.files:
                code = fn.stem
                code_file = summary / fn.name
                with open(str(code_file), 'r') as f:
                    l = json.load(f)
                    self.code2sent[str(code)] = l


    def __load__(self, path, N, minimum, fast, only, load_all=False):
        sents = []
        #idxs = set()
        idxs = {}
        code = str(path.stem)
        if fast:
            with open(path, 'r') as f:
                for idx, line in enumerate(f):
                    if not load_all and len(sents) > minimum:
                        break

                    line = line.rstrip()
                    if not line:
                        continue

                    line = json.loads(line)
                    sents.append(line)

        else:
            for key, l in self.code2sent[code].items():
                if len(l) <= minimum:
                    tmps = l
                else:
                    tmps = random.sample(l, minimum)
                for t in tmps:
                    idxs[t] = key

            with open(path, 'r') as f:
                for idx, line in enumerate(f):
                    if idx not in idxs:
                        continue
                    line = line.rstrip()
                    if not line:
                        continue

                    line = json.loads(line)

                    if only:
                        for entity in line['entities']:
                            if entity['string'] == idxs[idx]:
                                line['entities'] = [entity.copy()]
                                break

                    sents.append(line)

        return sents

    def __pos_pair__(self, sents):
        pos_pair = []
        idx_list = list(range(0, len(sents)))
        random.shuffle(idx_list)
        bag = []

        for i, idx in enumerate(idx_list):
            bag.append(sents[idx])
            if i % 2 == 0:
                pos_pair.append(bag)
                bag = []

        if len(bag) == 1:
            pos_pair.append(bag)

        return pos_pair

    def __sample__(self, N=1000, minimum=50, fast=False, only=False):
        self.pos_pair = []
        for fn in tqdm(self.files):
            sents = self.__load__(fn, N, minimum, fast, only)
            pos_pair = self.__pos_pair__(sents)
            self.pos_pair.extend(pos_pair)

    def __sample_all__(self, only=False, batch_size=32):
        self.pos_pair = []
        all_sents = set()
        input_ent_idxs = []
        input_tokens = []
        input_tags = []
        for fn in tqdm(self.files):
            sents = self.__load__(fn, 1000, 50, True, only, load_all=True)
            tokens = [' '.join(s['tokens']) for s in sents]
            for s, t in zip(sents, tokens):
                t_hash = hash(t)
                if t_hash not in all_sents:
                    ent, token, tag = self.__preprocess__(s)
                    input_ent_idxs.append(ent)
                    input_tokens.append(token)
                    input_tags.append(tag)
                    all_sents.add(t_hash)

                    if len(input_tokens) > batch_size:
                        yield input_ent_idxs, input_tokens, input_tags
                        input_ent_idxs = []
                        input_tokens = []
                        input_tags = []

        yield input_ent_idxs, input_tokens, input_tags

    def __getitem__(self, idx):
        sents = self.pos_pair[idx]
        sents = [self.__preprocess__(sent) for sent in sents]

        ent_idxs = [t[0] for t in sents]
        tokens = [t[1] for t in sents]
        tags = [t[2] for t in sents]

        return ent_idxs, tokens, tags

    def __len__(self):
        return len(self.pos_pair)


def my_collate_fn_for_sent(batch):
    ent_idxs, tokens, tags = list(zip(*batch))
    tokens = [t for token in tokens for t in token]
    tags = [t for tag in tags for t in tag]
    ent_idxs = [e for ent in ent_idxs for e in ent]

    return ent_idxs, tokens, tags

