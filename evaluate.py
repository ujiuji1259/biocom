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

from data import SentTrainDataset, SentDataset, my_collate_fn_for_sent
from model import EntityBERT, FaissIndexer


def parse_args():
    parser = argparse.ArgumentParser(description='Precompute the embeddings')

    parser.add_argument('--concept_map', required=True, help='File path of concept mapping')
    parser.add_argument('--embedding_dir', required=True, help='Input directory path')
    parser.add_argument('--dataset_dir', required=True, help='dataset directory path')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--model_path', required=True, help='model save path')
    parser.add_argument('--shard_bsz', type=int, help='shard batch size')

    args = parser.parse_args()

    return args


def knn(preds, k):
    results = []
    for pred in preds:
        pred = pred[:k]
        pred = [','.join(p) for p in pred]
        counter = Counter(pred)
        label, count = counter.most_common(1)[0]
        results.append(label.split(','))

    return results


def update_knn(prev_scores, prev_tags, cur_scores, cur_tags, k):
    if prev_scores is None:
        return cur_tags, cur_scores

    prev_tags = [[','.join(p) for p in pp] for pp in prev_tags]
    cur_tags = [[','.join(p) for p in pp] for pp in cur_tags]
    total_tags = np.array([pp + cc for pp, cc in zip(prev_tags, cur_tags)])
    total_scores = np.hstack([prev_scores, cur_scores])

    new_indexs = np.argsort(total_scores, axis=1)[:, ::-1]
    total_tags = np.take_along_axis(total_tags, new_indexs[:, :k], 1).tolist()
    total_scores = np.take_along_axis(total_scores, new_indexs[:, :k], 1)

    total_tags = [[p.split(',') for p in pp] for pp in total_tags]

    return total_tags, total_scores


def evaluate_from_file(model, base_dir, datasets, ks = [15], only=False, bsz=5):
    accuracy = {}
    with torch.no_grad():
        train_datasets = load_all_embed(base_dir, bsz=bsz)
        results = {}
        test_datasets = {}
        for key in datasets.keys():
            test_dataset = SentTrainDataset(datasets[key][0], tokenizer, only=only, concept_set=base_valid_tags)
            test_embeds, test_tags = get_all_embed(model, test_dataset)
            test_embeds = normalize(torch.tensor(test_embeds, requires_grad=False), p=2, dim=-1)
            test_datasets[key] = datasets[key] + [test_embeds, test_tags]

        for train_embeds, train_tags in train_datasets:
            inference_model = FaissIndexer(is_gpu=False)
            inference_model.train_index(train_embeds, train_tags)

            for key, (train_dataset, test_dataset, output_path, offset, test_embeds, test_tags) in test_datasets.items():
                pred_tags, pred_scores = inference_model.search_nn_with_score(test_embeds, 30)

                if key in results:
                    prev_tags, prev_scores = results[key]['tags'], results[key]['scores']
                else:
                    prev_tags, prev_scores = None, None
                    results[key] = {}
                    results[key]['trues'] = test_tags
                    results[key]['output_path'] = output_path

                tags, scores = update_knn(prev_scores, prev_tags, pred_scores, pred_tags, 30)
                results[key]['tags'] = tags
                results[key]['scores'] = scores

                knn_test_preds = knn(tags, 10)
                acc = [any(p in tt['label'] for p in pp) for pp, tt in zip(knn_test_preds, test_tags) if any(t in base_valid_tags for t in tt['label'])]
                print(sum(acc) / (len(acc) + offset))


        for k in results.keys():
            test_tags = results[k]['trues']
            test_preds = results[k]['tags']
            valid_tags = base_valid_tags.copy()

            total_outputs = [{"true_label": tag['label'], "prediction": pred} for tag, pred in zip(test_tags, test_preds) if any(t in valid_tags for t in tag['label'])]
            with open(results[k]['output_path'], 'w') as f:
                json.dump(total_outputs, f, ensure_ascii=False, indent=4)

            accs = []
            for knn_k in ks:
                knn_test_preds = knn(test_preds, knn_k)
                acc = [any(p in tt['label'] for p in pp) for pp, tt in zip(knn_test_preds, test_tags) if any(t in valid_tags for t in tt['label'])]
                accs.append(acc)

            mlflow.log_metric(key + " acc", sum(acc) / len(acc), step=0)
            for ac, kn in zip(accs, ks):
                accuracy[k + str(kn)] = sum(ac) / (len(ac) + offset)

    return accuracy


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


def load_all_embed(base_dir, bsz=1):
    base_dir = Path(base_dir)
    file_num = len(list(base_dir.glob('*.npy')))
    batch_cnt = 0
    batch_tensor = np.zeros((1, 768))
    batch_tag = []
    for i in tqdm(range(file_num)):
        if batch_cnt > bsz:
            yield batch_tensor[1:, :], batch_tag
            batch_tensor = np.zeros((1, 768))
            batch_tag = []
            batch_cnt = 0

        result_tensor = np.load(str(base_dir / ("embedding" + str(i) + ".npy"))).astype(np.float16)
        with open(base_dir / ("label" + str(i) + ".tsv"), "r") as f:
            results = [l.split("\t") for l in f.read().split('\n') if l != '']
            result_tags = [r[0].split(',') for r in results]

        batch_tensor = np.concatenate([batch_tensor, result_tensor], axis=0)
        batch_tag.extend(result_tags)
        batch_cnt += 1

    if batch_cnt > 1:
        yield batch_tensor[1:, :], batch_tag


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]', '[/ENT]']})
    bert = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    bert.resize_token_embeddings(len(tokenizer))
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

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    datasets = {
            "ncbi": [datset_dir / "test_NCBID.jsonl", output_dir / "pred_NCBID.json", 4],
            "bc5cdr": [output_dir / "test_BC5CDR.jsonl", output_dir / "pred_BC5CDR.json", 0],
            "medmentions": [output_dir / "test_medmentions.jsonl", output_dir / "pred_medmentions.json", 1]
            }

    model.load_state_dict(torch.load(args.model_path))
    accuracy = evaluate_from_file(model, args.embedding_dir, datasets, only=False, bsz=args.shard_bsz)
    print(accuracy)

