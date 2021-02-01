# biocom
We present BioCoM, a contrastive learning framework for context-aware medical entity normalization. 
You can train the model by the following procedure: 1) construct the corpus (./corpus) 2) run traininig.

## Resource
### Dataset
You can download the [three dataset](http://aoi.naist.jp/biocom/) used in our experiments (NCBI disease corpus, BC5CDR, MedMentions).
These datasets is `.jsonl` format.
Please place these file into ./dataset folder.

Note that mentions whose concept is not in ``MeSH" or ``OMIM" are filtered out in the dataset.

### Corpus
You need to construct the corpus, a set of entity-linked sentences, since the corpus is too large to distribute.
Please see ./corpus/README.md and construct the corpus used in training and inference.


## Training

```
python train.py \
  --concept_map corpus/concept_map.jsonl \
  --input_dir (corpus directory that are built. Please see ./corpus/README.md) \
  --learning_rate 1e-5 \
  --batch_size 16 \
  --model_path (path to save model)
```


## Precompute the embeddings
You have to compute and save the embeddings of all the entity representations in the corpus.
```
python precompute_embeddings.py \
  --dictionary_path corpus/disease_down_half.tsv \
  --concept_map corpus/concept_map.jsonl \
  --input_dir (corpus directory that are built. Please see ./corpus/README.md) \
  --output_dir (save directory for entity representations)
```

## Evaluation
```
python evaluate.py \
  --concept_map corpus/concept_map.jsonl \
  --embedding_dir (directory of entity representations saved above) \
  --dataset_dir (dataset directory. Please see ./dataset/README.md) \
  --output_dir (Directory to save prediction samples for each dataset) \
  --model_path (model path used for inference) \
  --shard_bsz 10 # This means 10 embedding .npy (1.5G / file) at the same time.
```
