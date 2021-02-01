
# Constructing the corpus
Construction of the corpus, a set of entity-linked sentences, from PubMed abstracts.

## Download PubMed data
First, download all the articles from [National Library of Medicine](https://www.nlm.nih.gov/databases/download/pubmed_medline.html)

## Preprocess and Filter
Second, extract text from downloaded xml files and filter out the articles whose pmids are in the test dataset (NCBID, BC5CDR, MedMentions).

```bash
PUBMED_DIR=./pubmed
OUTPUT_FILE=./baseline.txt

python preprocess.py \
  --pubmed_path ${PUBMED_DIR} \
  --output_path ${OUTPUT_FILE} \
  --pmid_path test_pmids.txt
```

## Link the entity mentions
Third, link the entities in the sentences to the corresponding concepts by dictioanry matching.
Note that output directory needs to have at least 35G memory space.

```bash
INPUT_FILE=./baseline.txt
OUTPUT_DIR=./pubmed_down_half

python entity_matching.py \
  --dictionary_path disease_down_half.tsv \
  --input_path ${INPUT_FILE} \
  --output_dir ${OUTPUT_DIR}
```
