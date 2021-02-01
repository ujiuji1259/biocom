# BioCoM
We present BioCoM, a contrastive learning framework for context-aware medical entity normalization. 
You can train the model by the following procedure: 1) construct the corpus (./corpus) 2) run traininig.

## Requirments
```
numpy
tqdm
nltk
torch==1.5.1
transformers==2.11.0
pytorch-metric-learning==0.9.89
faiss==1.6.3
```
Please follow the instruction [here](https://github.com/facebookresearch/faiss) to install faiss.

## Resource
### Dataset
You can download the datasets used in our paper [here](http://aoi.naist.jp/biocom/) (NCBI disease corpus, BC5CDR, MedMentions).
Please move these file to `./dataset` folder to evaluate our model.
Note that the format of each dataset is different from the original data, and mentions whose concept is not in `MeSH` or `OMIM` are filtered out in the dataset.

### Corpus
The corpus, a set of entity-linked sentences, is too large to distribute, so please construct the corpus on your own.
Please see ./corpus/README.md and save the corpus in any directory.


## Training
The following example trains our model.
Please construct the corpus before training (see ./corpus/README.md).
```
MODEL=./biocom.model
INPUT_DIR=./corpus/pubmed_down_half

python train.py \
  --concept_map corpus/concept_map.jsonl \
  --input_dir ${INPUT_DIR} \
  --learning_rate 1e-5 \
  --batch_size 16 \
  --model_path ${MODEL}
```


## Precompute the embeddings
You can compute and save the embeddings of all the entity representations in the corpus by following example.
```
MODEL=./biocom.model
INPUT_DIR=./corpus/pubmed_down_half
OUTPUT_DIR=./corpus/precomputed_embeddings

python precompute_embeddings.py \
  --model_path ${MODEL} \
  --dictionary_path corpus/disease_down_half.tsv \
  --concept_map corpus/concept_map.jsonl \
  --input_dir ${INPUT_DIR} \
  --output_dir ${OUTPUT_DIR}
```

## Evaluation
The following example evaluates our model.
If you want to use trained model, you can download [trained model](http://aoi.naist.jp/biocom/sent_50_down_half.model) used in our experiments.

Specifically, `shard_bsz` means that we use the specified number of `.npy` file for normalization at the same time.
We iteratively retrieve the nearest neighbors and update them.
If you don't have enough memory space, you can set it to small number (e.g., 5)
```
MODEL=./biocom.model
EMBEDDING_DIR=./corpus/precomputed_embeddings
OUTPUT_DIR=./output

python evaluate.py \
  --concept_map corpus/concept_map.jsonl \
  --embedding_dir ${EMBEDDING_DIR} \
  --dataset_dir dataset \
  --output_dir ${OUTPUT_DIR} \
  --model_path ${MODEL} \
  --shard_bsz 10
```
