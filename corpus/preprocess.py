import argparse
from pathlib import Path
import gzip
import numpy as np

import pubmed_parser as pp


def parse_args():
    parser = argparse.ArgumentParser(description='Corpus construction for biocom')

    parser.add_argument('--pubmed_path', required=True, help='PubMed abstracts path')
    parser.add_argument('--output_path', required=True, help='Output file path')
    parser.add_argument('--pmid_path', required=True, help='Path of pmid list. Filter out abstracts that have these pmids.')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    BASELINE_PATH = Path(args.pubmed_path)
    OUTPUT_PATH = Path(args.output_path)

    with open(args.pmid_path, 'r') as f:
        pmids = set([line for line in f.read().split('\n') if line != ''])


    md5_set = set()
    with open(OUTPUT_PATH, 'w') as g:
        for path in BASELINE_PATH.glob('*.xml.gz'):
            if md5 in md5_set:
                continue
            md5_set.add(md5)

            res = pp.parse_medline_xml(str(path),
                    year_info_only=False,
                    nlm_category=False,
                    author_list=False,
                    reference_list=False)

            for r in res:
                if r['pmid'] in pmids:
                    continue

                if isinstance(r['title'], str) and r['title'] != '':
                    g.write(r['title'] + '\n')

                if isinstance(r['abstract'], str) and r['abstract'] != '':
                    g.write(r['abstract'] + '\n')

