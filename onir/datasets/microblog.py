import os,re
import json
from pytools import memoize_method
from onir import datasets, util, indices
from onir.interfaces import trec, plaintext

# Generate pipeline
# python -m onir.bin.catfog config/catastrophic_forgetting/configX

VALIDATION_QIDS=[135, 127, 222, 112, 142, 118, 191, 174, 194, 199, 225, 119, 203, 168]
TEST_QIDS = [192, 115, 151, 180, 121, 156, 181, 177, 158, 122, 211, 147, 195, 224, 129, 111, 183, 137, 216, 164, 205, 223, 123]


_FILES = {
    'qrels_2013': dict(url='https://trec.nist.gov/data/microblog/2013/qrels.txt', expected_md5="4776a5dfd80b3f675184315ec989c02f"),
    'queries_2013': dict(url='https://trec.nist.gov/data/microblog/2013/topics.MB111-170.txt', expected_md5="0b78d99dfa2d655dca7e9f138a93c21a"),
    'queries_2014': dict(url='https://trec.nist.gov/data/microblog/2014/topics.MB171-225.txt', expected_md5="28e791895ce21469eabf1944668b26ef"),
    'qrels_2014': dict(url='https://trec.nist.gov/data/microblog/2014/qrels2014.txt', expected_md5="68d9a1920b244f6ccdc687ee1d473214"),
}


@datasets.register('microblog')
class MicroblogDataset(datasets.IndexBackedDataset):
    """
    Interface to the TREC Robust 2004 dataset.
     > Ellen M. Voorhees. 2004. Overview of TREC 2004. In TREC.
    """
    DUA = """Will begin downloading Robust04 dataset.
Please confirm you agree to the authors' data usage stipulations found at
https://trec.nist.gov/data/cd45/index.html"""

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'subset': 'all',
            'ranktopk': 100
        })
        return result

    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        base_path = util.path_dataset(self)
        self.index = indices.AnseriniIndex(os.path.join(base_path, 'anserini'), stemmer='none')
        self.index_stem = indices.AnseriniIndex(os.path.join(base_path, 'anserini.porter'), stemmer='porter')
        self.doc_store = indices.SqliteDocstore(os.path.join(base_path, 'docs.sqllite'))

    def _get_index(self, record):
        return self.index

    def _get_docstore(self):
        return self.doc_store

    def _get_index_for_batchsearch(self):
        return self.index_stem

    @memoize_method
    def _load_queries_base(self, subset):
        topics = self._load_topics(subset)
        return topics

    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt)

    @memoize_method
    def _load_qrels(self, subset, fmt):
        print("Reading qrels",f'{subset}.qrels.txt')
        return trec.read_qrels_fmt(os.path.join(util.path_dataset(self), f'{subset}.qrels.txt'), fmt)

    @memoize_method
    def _load_topics(self,subset):
        result = {}
        for qid, text in plaintext.read_tsv(os.path.join(util.path_dataset(self), 'topics.txt')):
            #nqid=int(qid.replace('MB','').strip())
            if subset=='valid' and (int(qid) in VALIDATION_QIDS):
                result[qid] = text
            elif subset=='test' and (int(qid) in TEST_QIDS):
                result[qid] = text
            elif subset=='train' and (int(qid) not in VALIDATION_QIDS) and (int(qid) not in TEST_QIDS):
                result[qid] = text
        return result

    def init(self, force=False):
        base_path = util.path_dataset(self)
        idxs = [self.index, self.index_stem, self.doc_store]
        self._init_indices_parallel(idxs, self._init_iter_collection(), force)
        train_qrels = os.path.join(base_path, 'train.qrels.txt')
        valid_qrels = os.path.join(base_path, 'valid.qrels.txt')
        test_qrels = os.path.join(base_path, 'test.qrels.txt')

        if (force or not os.path.exists(train_qrels) or not os.path.exists(valid_qrels)) and self._confirm_dua():
            source_stream = util.download_stream(**_FILES['qrels_2013'], encoding='utf8')
            source_stream2 = util.download_stream(**_FILES['qrels_2014'], encoding='utf8')
            with util.finialized_file(train_qrels, 'wt') as tf, \
                 util.finialized_file(valid_qrels, 'wt') as vf, \
                 util.finialized_file(test_qrels, 'wt') as Tf: 
                for line in source_stream:
                    cols = line.strip().split()
                    if int(cols[0]) in VALIDATION_QIDS:
                        vf.write(' '.join(cols) + '\n')
                    elif int(cols[0]) in TEST_QIDS:
                        Tf.write(' '.join(cols)+ '\n')
                    else:
                        tf.write(' '.join(cols) + '\n')
                for line in source_stream2:
                    cols = line.strip().split()
                    if cols[0] in VALIDATION_QIDS:
                        vf.write(' '.join(cols) + '\n')
                    elif int(cols[0]) in TEST_QIDS:
                        Tf.write(' '.join(cols)+ '\n')
                    else:
                        tf.write(' '.join(cols) + '\n')
            

        all_queries = os.path.join(base_path, 'topics.txt')

        if (force or not os.path.exists(all_queries) ) and self._confirm_dua():
            source_stream = util.download_stream(**_FILES['queries_2013'], encoding='utf8')
            source_stream2 = util.download_stream(**_FILES['queries_2014'], encoding='utf8')
            train, valid = [], []
            for _id,_query in trec.parse_query_mbformat(source_stream):
                nid = _id.replace('MB','').strip()
                train.append([nid,_query])
            
            for _id,_query in trec.parse_query_mbformat(source_stream2):
                nid = _id.replace('MB','').strip()
                train.append([nid,_query])

            plaintext.write_tsv(all_queries, train)


    def _init_iter_collection(self):
        # Using the trick here from capreolus, pulling document content out of public index:
        # <https://github.com/capreolus-ir/capreolus/blob/d6ae210b24c32ff817f615370a9af37b06d2da89/capreolus/collection/robust04.yaml#L15>
        index = indices.AnseriniIndex(f'../Tweets2013')
        for did in self.logger.pbar(index.docids(), desc='documents'):
            raw_doc = index.get_raw(did)
            #dict_doc = json.loads(raw_doc)
            #print("Doc",dict_doc,raw_doc, dict_doc['text'])
            pattern='"text":"(.*?)","source":'
            raw_txt = re.search(pattern,raw_doc).group(1)
            #print(raw_txt)
            #print(raw_txt)
            yield indices.RawDoc(did, raw_txt)

