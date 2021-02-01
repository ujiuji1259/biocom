import argparse
import dartsclone
import json
import linecache
from pathlib import Path

import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, WordPunctTokenizer
from transformers import AutoTokenizer, AutoModel
word_tokenize = WordPunctTokenizer().tokenize

from nltk.corpus import stopwords



def parse_args():
    parser = argparse.ArgumentParser(description='Construct the entity-linked sentences for biocom')

    parser.add_argument('--dictionary_path', required=True, help='Dictionary path')
    parser.add_argument('--input_path', required=True, help='Input file path')
    parser.add_argument('--output_dir', required=True, help='Output directory. Note that this directory needs at least 35G memory space')
    parser.add_argument('--max_row', type=int, help='The number of the row of the input file')

    args = parser.parse_args()

    return args


def load_entities(path):
    entities = []
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            line = json.loads(line)
            if line[0].lower() not in sw:
                entities.append(line)

    return sorted(entities, key=lambda s: s[0])


def create_darts(entities, path):
    darts = dartsclone.DoubleArray()
    entities = [e[0].encode('utf-8') for e in entities]
    values = [i for i, e in enumerate(entities)]

    darts.build(entities, values=values)

    return darts

def select_candidates(candidates, text_len):
    is_used = np.zeros(text_len)
    outputs = []
    for c in candidates:
        if np.sum(is_used[c[2]:c[3]]) > 0:
            continue

        outputs.append(c)
        is_used[c[2]:c[3]] = 1
    
    return outputs


def search_match(text, darts):
    entity = {}

    tokens = text.split(' ')
    #tokens = [word_tokenize(t) for t in tokens]
    tokens = [tokenizer.tokenize(t) for t in tokens]

    start_idxs = []
    last_idxs = [[-1, -1]]
    total_tokens = 0
    for token in tokens:
        first = True
        for t in token:
            if first:
                start_idxs.append([last_idxs[-1][0] + 1, last_idxs[-1][1] + 1])
                last_idxs.append([start_idxs[-1][0] + len(t.replace('##', '')), start_idxs[-1][1]])
            else:
                last_idxs[-1][0] += len(t.replace('##', ''))
                last_idxs[-1][1] += 1
            first = False
    last_idxs = {i:j for i, j in last_idxs[1:]}

    all_candidates = []
    for idx, (i, token_idx) in enumerate(start_idxs):
        query = text[i:].lower()
        result = darts.common_prefix_search(query.encode('utf-8'), pair_type=False)
        candidates = [entity_list[r] for r in result]

        for c in candidates:
            if i + len(c[0]) in last_idxs:
                all_candidates.append([i, i+len(c[0]), token_idx, last_idxs[i + len(c[0])]+1] + c)

    all_candidates = sorted(all_candidates, key=lambda s: s[1] - s[0], reverse=True)
    tokens = [t for token in tokens for t in token]

    all_candidates = select_candidates(all_candidates, len(tokens))
    entity['tokens'] = tokens
    entity['entities'] = [{'span': [c[2], c[3]], 'string': c[4], 'descriptor': c[5]} 
            for c in all_candidates]

    all_concepts = list(set([c[5][0] for c in all_candidates]))

    return entity, all_concepts


if __name__ == '__main__':
    args = parse_args()
    MAX_ROW = args.max_row if args.max_ros else 112129518 
    BASE_DIR = Path(args.output_dir)

    sw = stopwords.words("english")
    sw = set([s.lower() for s in sw])

    entity_list = load_entities(args.dictionary_path)
    darts = create_darts(entity_list, args.trie_path)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    cnt = 0
    bar = tqdm(total = MAX_ROW)
    cnt = 0
    with open(args.input_path, 'r') as fin:
        bar.set_description('Progress rate')
        for line in fin:
            line = line.rstrip()
            sents = sent_tokenize(line)
            for s in sents:
                entities, concepts = search_match(s, darts)
                if not entities['entities']:
                    continue
                for concept in concepts:
                    with open(str(BASE_DIR / (concept + '.jsonl')), 'a') as g:
                        g.write(json.dumps(entities, ensure_ascii=False) + '\n')

            f.write(str(cnt)+'\n')
            cnt += 1

            bar.update(1)

