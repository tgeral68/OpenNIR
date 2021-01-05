import os
import io
import itertools
import gzip
import tarfile
import zipfile
import contextlib
import functools
from tqdm import tqdm
from pytools import memoize_method
import onir
from onir import util, datasets, indices
from onir.interfaces import trec, plaintext
import pandas as pd
import numpy as np
from onir.datasets import microblog, covid, msmarco
import copy

MINI_DEV = {'484694', '836399', '683975', '428803', '1035062', '723895', '267447', '325379', '582244', '148817', '44209', '1180950', '424238', '683835', '701002', '1076878', '289809', '161771', '807419', '530982', '600298', '33974', '673484', '1039805', '610697', '465983', '171424', '1143723', '811440', '230149', '23861', '96621', '266814', '48946', '906755', '1142254', '813639', '302427', '1183962', '889417', '252956', '245327', '822507', '627304', '835624', '1147010', '818560', '1054229', '598875', '725206', '811871', '454136', '47069', '390042', '982640', '1174500', '816213', '1011280', '368335', '674542', '839790', '270629', '777692', '906062', '543764', '829102', '417947', '318166', '84031', '45682', '1160562', '626816', '181315', '451331', '337653', '156190', '365221', '117722', '908661', '611484', '144656', '728947', '350999', '812153', '149680', '648435', '274580', '867810', '101999', '890661', '17316', '763438', '685333', '210018', '600923', '1143316', '445800', '951737', '1155651', '304696', '958626', '1043094', '798480', '548097', '828870', '241538', '337392', '594253', '1047678', '237264', '538851', '126690', '979598', '707766', '1160366', '123055', '499590', '866943', '18892', '93927', '456604', '560884', '370753', '424562', '912736', '155244', '797512', '584995', '540814', '200926', '286184', '905213', '380420', '81305', '749773', '850038', '942745', '68689', '823104', '723061', '107110', '951412', '1157093', '218549', '929871', '728549', '30937', '910837', '622378', '1150980', '806991', '247142', '55840', '37575', '99395', '231236', '409162', '629357', '1158250', '686443', '1017755', '1024864', '1185054', '1170117', '267344', '971695', '503706', '981588', '709783', '147180', '309550', '315643', '836817', '14509', '56157', '490796', '743569', '695967', '1169364', '113187', '293255', '859268', '782494', '381815', '865665', '791137', '105299', '737381', '479590', '1162915', '655989', '292309', '948017', '1183237', '542489', '933450', '782052', '45084', '377501', '708154'}

"""
Mixed dataset: MSMarco + Microblog + Cord19.
"""
@datasets.register('mixmsmbcord')
class MixmsmbcordDataset(datasets.IndexBackedDataset):
    DUA = """Will begin downloading MS-MARCO dataset.
Please confirm you agree to the authors' data usage stipulations found at
http://www.msmarco.org/dataset.aspx"""

    @staticmethod
    def default_config():
        result = datasets.IndexBackedDataset.default_config()
        result.update({
            'subset': onir.config.Choices(['train', 'train10', 'train_med', 'dev', 'minidev', 'judgeddev', 'eval', 'trec2019', 'judgedtrec2019']),
            'rankfn': onir.config.Ranker(),
            'ranktopk': 100,
            'special': onir.config.Choices(['', 'mspairs', 'msrun', 'validrun']),
            'bs_field':'text',
            'rr_field':'text',
            'bs_override':'text',
            '2020_filter': '',
        })
        return result

    def __init__(self, config, logger, vocab):
        super().__init__(config, logger, vocab)
        base_path = util.path_dataset(self)

        global_base_path = "/".join(base_path.split("/")[:-1])
        #setup msmarco
        _base_path = global_base_path+"/msmarco"
        self.ms_index_stem = indices.AnseriniIndex(os.path.join(_base_path, 'anserini.porter'), stemmer='porter')
        self.ms_index_doctttttquery_stem = indices.AnseriniIndex(os.path.join(_base_path, 'anserini.doctttttquery.porter'), stemmer='porter')
        self.ms_doc_store = indices.SqliteDocstore(os.path.join(_base_path, 'docs.sqllite'))
        
        #setup microblog
        _base_path = global_base_path+"/microblog"
        self.mb_index_stem = indices.AnseriniIndex(os.path.join(_base_path, 'anserini.porter'), stemmer='porter')
        self.mb_index = indices.AnseriniIndex(os.path.join(_base_path, 'anserini'), stemmer='none')
        self.mb_doc_store = indices.SqliteDocstore(os.path.join(_base_path, 'docs.sqllite'))
        
        #setup cord
        _base_path = global_base_path+"/covid/2020-07-16"
        self.cord_index_stem = indices.MultifieldAnseriniIndex(os.path.join(_base_path, 'anserini_multifield'), stemmer='porter', primary_field=config['bs_field'])
        self.cord_index_stem_2020 = indices.MultifieldAnseriniIndex(os.path.join(_base_path, 'anserini_multifield_2020'), stemmer='porter', primary_field=config['bs_field'])
        self.cord_doc_store = indices.MultifieldSqliteDocstore(os.path.join(_base_path, 'docs_multifield.sqlite'), primary_field=config['rr_field'])
    
        self.msds = msmarco.MsmarcoDataset(self.msmarco_config(self.config['subset'],config),logger,vocab)
        self.mbds = microblog.MicroblogDataset(self.microblog_config(self.config['subset'],config),logger,vocab)
        self.cordds = covid.CovidDataset(self.cord_config(self.config['subset'],config),logger,vocab)


    def msmarco_config(self, subset, config):
        config2 = copy.deepcopy(config)
        config2['rankfn']='bm25_k1-0.82_b-0.68'
        if subset=='train':
            config2['special']='mspairs'
            config2['ranktopk']=100
        elif subset=='valid' or subset=='minidev':
            config2['measures']='ndcg@20,map@100,p@1,p@20,rprec'
            config2['ranktopk']=20
            config2['subset']='minidev'
        elif subset=='test' or subset=='judgeddev':
            config2['measures']='ndcg@20,map@100,p@1,p@20,rprec'
            config2['ranktopk']=100
            config2['subset']='judgeddev'
        return config2


    def microblog_config(self, subset, config):
        config2 = copy.deepcopy(config)
        config2['rankfn']='bm25_k1-0.2_b-0.95'
        if subset=='train':
            config2['special']='mspairs'
            config2['ranktopk']=100
            config2['measures']='ndcg@20,map@100,p@20,ndcg@10'
        elif subset=='valid':
            config2['measures']='ndcg@20,map@100,p@20,ndcg@10'
            config2['ranktopk']=100
        elif subset=='test':
            config2['measures']='ndcg@20,map@100,p@20,ndcg@10'
            config2['ranktopk']=100
        return config2

    def cord_config(self, subset, config):
        config2 = copy.deepcopy(config)
        config2['rankfn']='bm25_k1-3.9_b-0.55'
        config2['2020_filter']=True
        config2['rr_field']='title_abs'
        config2['bs_field']='text'
        config2['bs_override']='rnd5-query'
        config2['date']='2020-07-16'
        if subset=='train':
            config2['special']='mspairs'
            config2['ranktopk']=100
            config2['subset']='trf2-rnd5-quest'
            config2['measures']='ndcg@20,map@100,p@20,ndcg@10'
        elif subset=='valid':
            config2['measures']='ndcg@20,map@100,p@20,ndcg@10'
            config2['ranktopk']=100
            config2['subset']='vaf2-rnd5-quest'
        elif subset=='test':
            config2['measures']='ndcg@20,map@100,p@20,ndcg@10'
            config2['ranktopk']=100
            config2['subset']='f2-rnd5-quest'
        return config2



    def all_query_ids(self,current='all'):
        yield from self._load_queries_base(current,self.config['subset']).keys()

    def all_queries_raw(self):
        return self._load_queries_base(current,self.config['subset']).items()
    
    def _load_run_base_query(self, current, index, subset, rankfn, ranktopk, run_path, fmt):
        queries = self._load_queries_base(current,subset).items()
        index.batch_query(queries, rankfn, ranktopk, destf=run_path)
        return trec.read_run_fmt(run_path, fmt)

    def run(self,fmt='dict'):
        
        msrun = self.msds.run()
        mbrun = self.mbds.run()
        cordrun = self.cordds.run()

        if fmt=='dict':
            final_run = {}
            for query, scores in msrun.items():
                final_run["ms-"+query] = scores
            for query, scores in mbrun.items():
                final_run["mb-"+query] = scores
            for query, scores in cordrun.items():
                final_run["cord-"+query] = scores
            return final_run


    @memoize_method
    def _load_qrels(self, subset, fmt):
        with self.logger.duration('loading qrels'):
            base_path = util.path_dataset(self)
            path = os.path.join(base_path, f'{subset}.qrels')
            return trec.read_qrels_fmt(path, fmt)


    def qrels(self, fmt='dict'):
        return self._load_qrels(self.config['subset'], fmt=fmt)


    @memoize_method
    def _load_queries_base(self, current,subset):
        base_path = util.path_dataset(self)
        path = os.path.join(base_path, f'{current}-{subset}.queries.tsv')
        return dict(self.logger.pbar(plaintext.read_tsv(path), desc=f'loading queries CURRENT={current}'))


    def pair_iter(self, fields, pos_source='intersect', neg_source='run', sampling='query', pos_minrel=1, unjudged_rel=0, num_neg=1, random=None, inf=False):
        
        it_ms = self.msds.pair_iter(fields, pos_source, neg_source, sampling, pos_minrel, unjudged_rel, num_neg, random, inf)
        it_mb = onir.datasets.pair_iter(self.mbds,fields, pos_source, neg_source, sampling, pos_minrel, unjudged_rel, num_neg, random, inf)
        it_cord = onir.datasets.pair_iter(self.cordds,fields, pos_source, neg_source, sampling, pos_minrel, unjudged_rel, num_neg, random, inf)

        counter = 0
        no_mb, no_ms, no_cord=False,False,False # collections finished
        while True: # each 10000 queries we use 1 Microblog, each 3000 we use 1 Cord19
            if (counter==2 or counter==11) and not no_mb:
                try:
                    yield next(it_mb)
                except StopIteration:
                    no_mb=True
            elif (counter==7) and not no_cord:
                try:
                    yield next(it_cord)
                except StopIteration:
                    no_cord=True
            elif not no_ms:
                try:
                    yield next(it_ms)
                except StopIteration:
                    no_ms=True
                    break
            counter+=1
            if counter==16:
                counter=0

        for element in it_mb:
            yield(element)
        for element in it_cord:
            yield(element)
                




    def record_iter(self, fields, source, minrel=None, shuf=True, random=None, inf=False, run_threshold=None):
        
        for element in onir.datasets.record_iter(self.cordds,fields=fields, source=source, minrel=minrel, shuf=shuf, random=random, inf=inf, run_threshold=run_threshold):
            element['query_id']="cord-"+element['query_id']
            yield(element)
        for element in onir.datasets.record_iter(self.mbds,fields=fields, source=source, minrel=minrel, shuf=shuf, random=random, inf=inf, run_threshold=run_threshold):
            element['query_id']="mb-"+element['query_id']
            yield(element)
        for element in onir.datasets.record_iter(self.msds,fields=fields, source=source, minrel=minrel, shuf=shuf, random=random, inf=inf, run_threshold=run_threshold):
            element['query_id']="ms-"+element['query_id']
            yield(element)
 
   
    def init(self, force=False):
        return None