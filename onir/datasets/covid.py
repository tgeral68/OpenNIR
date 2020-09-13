import os
import json
import contextlib
import functools
import pandas as pd
from bs4 import BeautifulSoup
from pytools import memoize_method
import onir
from onir import datasets, util, indices
from onir.interfaces import trec, plaintext

_kfolds=5

_FOLDS = {
    'f1': {'6','7','8','20','29','35','37','41','42'}, #{'2','11','12','15','23','37','41','44','45','49'},
    'f2': {'1','3','14','24','30','33','36','38','40'}, #{'1','4','5','9','19','20','24','38','40','42'},
    'f3': {'4','16','19','26','27','28','31','34','44'}, #{'8','10','21','25','26','28','33','35','36','50'},
    'f4': {'9','10','12','13','15','21','25','43','45'}, #{'3','7','13','30','31','32','34','39','47','48'},
    'f5': {'2','5','11','17','18','22','23','32','39'} #{'6','14','16','17','18','22','27','29','43','46'}
}

FOLDS = {
    'f1': {'6','8','12','14','15','18','23','30','38'}, #{'2','11','12','15','23','37','41','44','45','49'},
    'f2': {'3','4','7','9','11','17','29','31','35'}, #{'1','4','5','9','19','20','24','38','40','42'},
    'f3': {'13','16','19','24','26','33','37','43','45'}, #{'8','10','21','25','26','28','33','35','36','50'},
    'f4': {'2','21','27','28','32','40','41','42','44'}, #{'3','7','13','30','31','32','34','39','47','48'},
    'f5': {'1','5','10','20','22','25','34','36','39'} #{'6','14','16','17','18','22','27','29','43','46'}
}


# we use folds, but we are looking for the best split train/test similar to previous results already found
# we search for 36/18 splits, donc for training 36 = 27/9 train/valid
_ALL = set.union(*FOLDS.values())
_FOLD_IDS = list(sorted(FOLDS.keys()))
for i in range(len(_FOLDS)):
    nx = (i+1)% _kfolds
    # 27/18
    FOLDS['tr' + _FOLD_IDS[i]] = _ALL - _FOLDS[_FOLD_IDS[i]] - _FOLDS[_FOLD_IDS[i-1]] - _FOLDS[_FOLD_IDS[nx]] 
    FOLDS['va' + _FOLD_IDS[i]] = _FOLDS[_FOLD_IDS[i-1]]
    FOLDS[_FOLD_IDS[i]] = _FOLDS[_FOLD_IDS[i]].union( _FOLDS[_FOLD_IDS[nx]] )
    
    # 36/9
    #FOLDS['tr' + _FOLD_IDS[i]] = _ALL - _FOLDS[_FOLD_IDS[i]] - _FOLDS[_FOLD_IDS[i-1]] 
    #FOLDS['va' + _FOLD_IDS[i]] = _FOLDS[_FOLD_IDS[i-1]]
    #FOLDS[_FOLD_IDS[i]] = _FOLDS[_FOLD_IDS[i]]
    
    #FOLDS[_FOLD_IDS[i]] = _FOLDS[_FOLD_IDS[i]]
    
    #FOLDS['tr' + _FOLD_IDS[i]] = _ALL - FOLDS[_FOLD_IDS[i]] - FOLDS[_FOLD_IDS[i-1]]  
    #FOLDS['va' + _FOLD_IDS[i]] = FOLDS[_FOLD_IDS[i-1]]
FOLDS['all'] = _ALL



@datasets.register('covid')
class CovidDataset(datasets.IndexBackedDataset):
    """
    Interface to the CORD-19 dataset
     > Helia Hashemi, Mohammad Aliannejadi, Hamed Zamani, and W. Bruce Croft. 2019. ANTIQUE: A
     > Non-Factoid Question Answering Benchmark.ArXiv (2019).
    and the TREC-COVID dataset
     > Kirk Roberts, Tasmeer Alam, Steven Bedrick, Dina Demner-Fushman, Kyle Lo, Ian Soboroff, Ellen
     > Voorhees, Lucy Lu Wang, and William R Hersh. 2020. TREC-COVID: Rationale and Structure of an
     > Information Retrieval Shared Task for COVID-19. JAMIA.
    """
    DUA = """'Will begin downloading COVID-19 Open Research Dataset (CORD-19) and TREC-COVID topics,
relevance judgments, etc.
Please confirm you agree to the authors' data usage stipulations found at
https://pages.semanticscholar.org/coronavirus-research
and
https://ir.nist.gov/covidSubmit/"""

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            # subset of queries to use
            #'subset': onir.config.Choices(['rnd1-query', 'rnd1-quest', 'rnd1-narr', 'rnd1-udel', 'rnd2-query', 'rnd2-quest', 'rnd2-narr', 'rnd2-udel', 'rnd5-query', 'rnd5-quest', 'rnd5-narr', 'rnd5-udel']),
            'subset': 'all',
            # date of CORD-19 dump to use
            'date': onir.config.Choices(['2020-07-16']),
            '2020_filter': False, # filter documents to only 2020?
            # batch search (bs) / re-rank (rr) document field to use
            'bs_field': onir.config.Choices(['text', 'title', 'abstract', 'title_abs', 'body'], default='text'),
            'rr_field': onir.config.Choices(['text', 'title', 'abstract', 'title_abs', 'body'], default='title_abs'),
            # subset override for batch search (for using different query field)
            'bs_override': onir.config.Choices(['', 'rnd1-query', 'rnd1-quest', 'rnd1-narr', 'rnd2-query', 'rnd2-quest', 'rnd2-narr','rnd5-query', 'rnd5-quest', 'rnd5-narr']),
            'fold' : 'all',
        })
        return result


    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        base_path = os.path.join(util.path_dataset(self), config['date'])
        os.makedirs(base_path, exist_ok=True)
        self.index_stem = indices.MultifieldAnseriniIndex(os.path.join(base_path, 'anserini_multifield'), stemmer='porter', primary_field=config['bs_field'])
        self.index_stem_2020 = indices.MultifieldAnseriniIndex(os.path.join(base_path, 'anserini_multifield_2020'), stemmer='porter', primary_field=config['bs_field'])
        self.doc_store = indices.MultifieldSqliteDocstore(os.path.join(base_path, 'docs_multifield.sqlite'), primary_field=config['rr_field'])


    def qrels(self, fmt='dict'):
        return self._base_qrels(self.config['subset'], fmt)

    @memoize_method
    def _base_qrels(self, subset, fmt):
        fold,rnd, _ = subset.split('-', 2)
        path = os.path.join(util.path_dataset(self), f'{fold}-{rnd}.qrels')
        if os.path.exists(path):
            aaa = trec.read_qrels_fmt(path, fmt)
            return aaa
        self.logger.info(f'missing qrels for {rnd} -- returning empty qrels')
        return {}

    def path_segment(self):
        result = '{base}_{date}'.format(base=super().path_segment(), **self.config)
        result += '_bs-{bs_field}'.format(**self.config)
        if self.config['2020_filter']:
            result += '_2020filter'
        if self.config['bs_override']:
            result += '_bsoverride-{bs_override}'.format(**self.config)
        result += '_rr-{rr_field}'.format(**self.config)
        return result

   

    # added but not sure if correct
    def _get_index(self, record):
        return self.index_stem

    def _get_index_for_batchsearch(self):
        if self.config['2020_filter']:
            return self.index_stem_2020
        return self.index_stem

    def _get_docstore(self):
        return self.doc_store

    def _load_queries_base(self,  subset):
        fold, rnd, fields = subset.split('-', 2)
        fields = fields.split('-')
        path = os.path.join(util.path_dataset(self), f'rnd5.tsv')
        filter_queries = {str(qid): qtext for qid, qtype, qtext in plaintext.read_tsv(path) if qtype in fields and str(qid) in FOLDS[fold]}
        return filter_queries
        

    #def qrels(self):
    #    return self._base_qrels(self.config['subset'])

    

    def init(self, force=False):
        needs_docs = []
        for index in [self.index_stem, self.index_stem_2020, self.doc_store]:
            if force or not index.built():
                needs_docs.append(index)

        if needs_docs and self._confirm_dua():
            with contextlib.ExitStack() as stack:
                doc_iter = self._init_iter_collection()
                doc_iter = self.logger.pbar(doc_iter, desc='articles')
                doc_iters = util.blocking_tee(doc_iter, len(needs_docs))
                for idx, it in zip(needs_docs, doc_iters):
                    if idx is self.index_stem_2020:
                        it = (d for d in it if '2020' in d.data['date'])
                    stack.enter_context(util.CtxtThread(functools.partial(idx.build, it)))

        path = os.path.join(util.path_dataset(self), 'rnd1.tsv')
        if not os.path.exists(path) and self._confirm_dua():
            with util.download_tmp('https://ir.nist.gov/covidSubmit/data/topics-rnd1.xml', expected_md5="cf1b605222f45f7dbc90ca8e4d9b2c31") as f, \
                 util.finialized_file(path, 'wt') as fout:
                soup = BeautifulSoup(f.read(), 'lxml-xml')
                for topic in soup.find_all('topic'):
                    qid = topic['number']
                    plaintext.write_tsv(fout, [
                        (qid, 'query', topic.find('query').get_text()),
                        (qid, 'quest', topic.find('question').get_text()),
                        (qid, 'narr', topic.find('narrative').get_text()),
                    ])

        udel_flag = path + '.includes_udel'
        if not os.path.exists(udel_flag):
            with open(path, 'at') as fout, util.finialized_file(udel_flag, 'wt'):
                 with util.download_tmp('https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.covid-round1-udel.xml', expected_md5="2915cf59ae222f0aa20b2a671f67fd7a") as f:
                    soup = BeautifulSoup(f.read(), 'lxml-xml')
                    for topic in soup.find_all('topic'):
                        qid = topic['number']
                        plaintext.write_tsv(fout, [
                            (qid, 'udel', topic.find('query').get_text()),
                        ])

        path = os.path.join(util.path_dataset(self), 'rnd2.tsv')
        if not os.path.exists(path) and self._confirm_dua():
            with util.download_tmp('https://ir.nist.gov/covidSubmit/data/topics-rnd2.xml', expected_md5="550129e71c83de3fb4d6d29a172c5842") as f, \
                 util.finialized_file(path, 'wt') as fout:
                soup = BeautifulSoup(f.read(), 'lxml-xml')
                for topic in soup.find_all('topic'):
                    qid = topic['number']
                    plaintext.write_tsv(fout, [
                        (qid, 'query', topic.find('query').get_text()),
                        (qid, 'quest', topic.find('question').get_text()),
                        (qid, 'narr', topic.find('narrative').get_text()),
                    ])

        udel_flag = path + '.includes_udel'
        if not os.path.exists(udel_flag):
            with open(path, 'at') as fout, util.finialized_file(udel_flag, 'wt'):
                 with util.download_tmp('https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.covid-round2-udel.xml', expected_md5="a8988734e6f812921d5125249c197985") as f:
                    soup = BeautifulSoup(f.read(), 'lxml-xml')
                    for topic in soup.find_all('topic'):
                        qid = topic['number']
                        plaintext.write_tsv(fout, [
                            (qid, 'udel', topic.find('query').get_text()),
                        ])

        path = os.path.join(util.path_dataset(self), 'rnd5.tsv')
        if not os.path.exists(path) and self._confirm_dua():
            with util.download_tmp('https://ir.nist.gov/covidSubmit/data/topics-rnd5.xml', expected_md5="0307a37b6b9f1a5f233340a769d538ea") as f, \
                 util.finialized_file(path, 'wt') as fout:
                soup = BeautifulSoup(f.read(), 'lxml-xml')
                for topic in soup.find_all('topic'):
                    qid = topic['number']
                    plaintext.write_tsv(fout, [
                        (qid, 'query', topic.find('query').get_text()),
                        (qid, 'quest', topic.find('question').get_text()),
                        (qid, 'narr', topic.find('narrative').get_text()),
                    ])

        udel_flag = path + '.includes_udel'
        if not os.path.exists(udel_flag):
            with open(path, 'at') as fout, util.finialized_file(udel_flag, 'wt'):
                 with util.download_tmp('https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.covid-round5-udel.xml', expected_md5="966a49487348dc853634bcdd0829fd26") as f:
                    soup = BeautifulSoup(f.read(), 'lxml-xml')
                    for topic in soup.find_all('topic'):
                        qid = topic['number']
                        plaintext.write_tsv(fout, [
                            (qid, 'udel', topic.find('query').get_text()),
                        ])

        qrels_file = os.path.join(util.path_dataset(self), 'rnd5.qrels')
        if (force or not os.path.exists(qrels_file)) and self._confirm_dua():
            util.download('https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.covid-round4-cumulative.txt', qrels_file)

        for fold in FOLDS:
            fold_qrels_file = os.path.join(util.path_dataset(self), f'{fold}-rnd5.qrels')
            if (force or not os.path.exists(fold_qrels_file)):
                all_qrels = trec.read_qrels_dict(qrels_file)
                
                fold_qrels = {qid: dids for qid, dids in all_qrels.items() if str(qid) in FOLDS[fold]}
                trec.write_qrels_dict(fold_qrels_file, fold_qrels)


    def _init_iter_collection(self):
        files = {
            '2020-04-10': {
                'comm_use_subset': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-04-10/comm_use_subset.tar.gz', "253cecb4fee2582a611fb77a4d537dc5"),
                'noncomm_use_subset': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-04-10/noncomm_use_subset.tar.gz', "734b462133b3c00da578a909f945f4ae"),
                'custom_license': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-04-10/custom_license.tar.gz', "2f1c9864348025987523b86d6236c40b"),
                'biorxiv_medrxiv': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-04-10/biorxiv_medrxiv.tar.gz', "c12acdec8b3ad31918d752ba3db36121"),
            },
            '2020-05-01': {
                'comm_use_subset': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-05-01/comm_use_subset.tar.gz', "af4202340182209881d3d8cba2d58a24"),
                'noncomm_use_subset': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-05-01/noncomm_use_subset.tar.gz', "9cc25b9e8674197446e7cbd4381f643b"),
                'custom_license': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-05-01/custom_license.tar.gz', "1cb6936a7300a31344cd8a5ecc9ca778"),
                'biorxiv_medrxiv': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-05-01/biorxiv_medrxiv.tar.gz', "9d6c6dc5d64b01e528086f6652b3ccb7"),
                'arxiv': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-05-01/arxiv.tar.gz', "f10890174d6f864f306800d4b02233bc"),
            },
            '2020-07-16': {
                'document_parses': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/document_parses.tar.gz',"b47ddab873ee020414f86e05e18ea528"),
            }
        }
        metadata = {
            '2020-04-10': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-04-10/metadata.csv', "42a21f386be86c24647a41bedde34046"),
            '2020-05-01': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-05-01/metadata.csv', "b1d2e409026494e0c8034278bacd1248"),
            '2020-07-16': ('https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/2020-07-16/metadata.csv', "80d664e496b8b7e50a39c6f6bb92e0ef"),
        }
        meta_url, meta_md5 = metadata[self.config['date']]

        fulltexts = {}
        with contextlib.ExitStack() as stack:
            for fid, (file, md5) in files[self.config['date']].items():
                fulltexts[fid] = stack.enter_context(util.download_tmp(file, tarf=True, expected_md5=md5))
            meta = pd.read_csv(util.download_stream(meta_url, expected_md5=meta_md5))

            if self.config['date']=='2020-07-16':
                for _, row in meta.iterrows():
                    did = str(row['cord_uid'])
                    title = str(row['title'])
                    doi = str(row['doi']) 
                    abstract = str(row['abstract'])
                    date = str(row['publish_time'])
                    body = ''
                    heads = ''
                    if row['pmc_json_files'] and isinstance(row['pmc_json_files'],str):
                        path = row['pmc_json_files']
                        fulltextfile = path.split("/")[0]
                        
                        #path = os.path.join(row['full_text_file'], 'pmc_json', row['pmcid'] + '.xml.json')
                        data = json.load(fulltexts[fulltextfile].extractfile(path))
                        if 'body_text' in data:
                            body = '\n'.join(b['text'] for b in data['body_text'])
                            heads = '\n'.join(set(b['section'] for b in data['body_text']))
                    elif row['pdf_json_files'] and isinstance(row['pdf_json_files'],str):
                        path = row['pdf_json_files'].split(';')[0].strip()

                        #path = os.path.join(row['full_text_file'], 'pdf_json', row['sha'].split(';')[0].strip() + '.json')
                        fulltextfile = path.split("/")[0]
                        data = json.load(fulltexts[fulltextfile].extractfile(path))
                        if 'body_text' in data:
                            body = '\n'.join(b['text'] for b in data['body_text'])
                            heads = '\n'.join(set(b['section'] for b in data['body_text']))
                    contents = f'{title}\n\n{abstract}\n\n{body}\n\n{heads}'
                    doc = indices.RawDoc(did, text=contents, title=title, abstract=abstract, title_abs=f'{title}\n\n{abstract}', body=body, doi=doi, date=date)
                    yield doc
            else:
                for _, row in meta.iterrows():
                    did = str(row['cord_uid'])
                    title = str(row['title'])
                    doi = str(row['doi'])
                    abstract = str(row['abstract'])
                    date = str(row['publish_time'])
                    body = ''
                    heads = ''
                    if row['has_pmc_xml_parse']:
                        path = os.path.join(row['full_text_file'], 'pmc_json', row['pmcid'] + '.xml.json')
                        data = json.load(fulltexts[row['full_text_file']].extractfile(path))
                        if 'body_text' in data:
                            body = '\n'.join(b['text'] for b in data['body_text'])
                            heads = '\n'.join(set(b['section'] for b in data['body_text']))
                    elif row['has_pdf_parse']:
                        path = os.path.join(row['full_text_file'], 'pdf_json', row['sha'].split(';')[0].strip() + '.json')
                        data = json.load(fulltexts[row['full_text_file']].extractfile(path))
                        if 'body_text' in data:
                            body = '\n'.join(b['text'] for b in data['body_text'])
                            heads = '\n'.join(set(b['section'] for b in data['body_text']))
                    contents = f'{title}\n\n{abstract}\n\n{body}\n\n{heads}'
                    doc = indices.RawDoc(did, text=contents, title=title, abstract=abstract, title_abs=f'{title}\n\n{abstract}', body=body, doi=doi, date=date)
                    yield doc
