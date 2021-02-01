from pathlib import Path
import gzip
import numpy as np

import pubmed_parser as pp


BASELINE_PATH = Path('/data2/ftp.ncbi.nlm.nih.gov/pubmed/baseline')
OUTPUT_PATH = '/data1/ujiie/pubmed/baseline.txt'

md5_set = set()


if __name__ == '__main__':
    with open('test_pmids.txt', 'r') as f:
        pmids = set([line for line in f.read().split('\n') if line != ''])

    with open(OUTPUT_PATH, 'w') as g:
        for path in BASELINE_PATH.glob('*.xml.gz'):
            md5_file = str(path) + '.md5'
            with open(md5_file, 'r') as f:
                md5 = f.read().strip().split('=')[-1].strip()

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

