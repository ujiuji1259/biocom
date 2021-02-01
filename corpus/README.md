
# Construction of the corpus (a set of the entity-linked sentences)

## Download PubMed data
You can download all the articles used in our experiments from [National Library of Medicine](https://www.nlm.nih.gov/databases/download/pubmed_medline.html)

## Preprocess and Filter
Now that you download the PubMed articles, you need to extract text from xml files and filter out the articles whose pmids are in the test dataset (NCBID, BC5CDR, MedMentions).

```bash
python preprocess.py \
  --pubmed_path (download path of PubMed data) \
  --output_path (output file path) \
  --pmid_path test_pmids.txt
```

## Link the entity mentions
You can link the entities in the sentences to the corresponding sentences by dictioanry matching.
Note that output directory needs to have at least 35G memory space.

```bash
python entity_matching.py \
  --dictionary_path disease_down_half.tsv
  --input_path (Input file path (= output file of preprocess and filter))
  --output_dir (Output directory)
  --max_row (The number of rows of input file)
```
