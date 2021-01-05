import sys
import torch
import torch.nn.functional as F
import onir
from onir import trainers, spec, util
from tqdm import tqdm
from onir import util, trainers
from onir.interfaces import apex
import pickle, os
from shutil import copyfile


@trainers.register('pairwise_ewc')
class PairwiseEWCTrainer(trainers.Trainer):
    @staticmethod
    def default_config():
        result = trainers.Trainer.default_config()
        result.update({
            'lossfn': onir.config.Choices(['softmax', 'cross_entropy', 'nogueira_cross_entropy', 'hinge']),
            'pos_source': onir.config.Choices(['intersect', 'qrels']),
            'neg_source': onir.config.Choices(['run', 'qrels', 'union']),
            'sampling': onir.config.Choices(['query', 'qrel']),
            'pos_minrel': 1,
            'unjudged_rel': 0,
            'num_neg': 1,
            'margin': 0.,
        })
        return result

    def __init__(self, config, ranker, logger, train_ds, vocab, random):
        super().__init__(config, ranker, vocab, train_ds, logger, random)
        self.loss_fn = {
            'softmax': self.softmax,
            'cross_entropy': self.cross_entropy,
            'nogueira_cross_entropy': self.nogueira_cross_entropy,
            'hinge': self.hinge
        }[config['lossfn']]
        self.dataset = train_ds
        self.input_spec = ranker.input_spec()
        self.iter_fields = self.input_spec['fields'] | {'runscore'}
        self.train_iter_core = onir.datasets.pair_iter(
            train_ds,
            fields=self.iter_fields,
            pos_source=self.config['pos_source'],
            neg_source=self.config['neg_source'],
            sampling=self.config['sampling'],
            pos_minrel=self.config['pos_minrel'],
            unjudged_rel=self.config['unjudged_rel'],
            num_neg=self.config['num_neg'],
            random=self.random,
            inf=True)
        self.train_iter = util.background(self.iter_batches(self.train_iter_core))
        self.numneg = config['num_neg']

        self._ewc = False

    def path_segment(self):
        path = super().path_segment()
        pos = 'pos-{pos_source}-{sampling}'.format(**self.config)
        if self.config['pos_minrel'] != 1:
            pos += '-minrel{pos_minrel}'.format(**self.config)
        neg = 'neg-{neg_source}'.format(**self.config)
        if self.config['unjudged_rel'] != 0:
            neg += '-unjudged{unjudged_rel}'.format(**self.config)
        if self.config['num_neg'] != 1:
            neg += '-numneg{num_neg}'.format(**self.config)
        loss = self.config['lossfn']
        if loss == 'hinge':
            loss += '-{margin}'.format(**self.config)
        result = 'pairwise_{path}_{loss}_{pos}_{neg}'.format(**self.config, loss=loss, pos=pos, neg=neg, path=path)
        if self.config['gpu'] and not self.config['gpu_determ']:
            result += '_nondet'
        return result

    def iter_batches(self, it):
        while True: # breaks on StopIteration
            input_data = {}
            for _, record in zip(range(self.batch_size), it):
                for k, v in record.items():
                    assert len(v) == self.numneg + 1
                    for seq in v:
                        input_data.setdefault(k, []).append(seq)
            input_data = spec.apply_spec_batch(input_data, self.input_spec, self.device)
            yield input_data

    def train_batch(self):
        input_data = next(self.train_iter)
        rel_scores = self.ranker(**input_data)
        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error('nan or inf relevance score detected. Aborting.')
            sys.exit(1)
        rel_scores_by_record = rel_scores.reshape(self.batch_size, self.numneg + 1, -1)
        run_scores_by_record = input_data['runscore'].reshape(self.batch_size, self.numneg + 1)
        loss = self.loss_fn(rel_scores_by_record)

        losses = {'data': loss}
        loss_weights = {'data': 1.}

        return {
            'losses': losses,
            'loss_weights': loss_weights,
            'acc': self.acc(rel_scores_by_record),
            'unsup_acc': self.acc(run_scores_by_record)
        }

    def fast_forward(self, record_count):
        self._fast_forward(self.train_iter_core, self.iter_fields, record_count)

    @staticmethod
    def cross_entropy(rel_scores_by_record):
        target = torch.zeros(rel_scores_by_record.shape[0]).long().to(rel_scores_by_record.device)
        return F.cross_entropy(rel_scores_by_record, target, reduction='mean')

    @staticmethod
    def nogueira_cross_entropy(rel_scores_by_record):
        """
        cross entropy loss formulation for BERT from:
         > Rodrigo Nogueira and Kyunghyun Cho. 2019.Passage re-ranking with bert. ArXiv,
         > abs/1901.04085.
        """
        log_probs = -rel_scores_by_record.log_softmax(dim=2)
        return (log_probs[:, 0, 0] + log_probs[:, 1, 1]).mean()

    @staticmethod
    def softmax(rel_scores_by_record):
        return torch.mean(1. - F.softmax(rel_scores_by_record, dim=1)[:, 0])

    def hinge(self, rel_scores_by_record):

        return F.relu(self.config['margin'] - rel_scores_by_record[:, :1] + rel_scores_by_record[:, 1:])
        return F.relu(self.config['margin'] - rel_scores_by_record[:, :1] + rel_scores_by_record[:, 1:]).mean()

    @staticmethod
    def pointwise(rel_scores_by_record):
        log_probs = -rel_scores_by_record.log_softmax(dim=2)
        return (log_probs[:, 0, 0] + log_probs[:, 1, 1]).mean()

    @staticmethod
    def acc(scores_by_record):
        count = scores_by_record.shape[0] * (scores_by_record.shape[1] - 1)
        return (scores_by_record[:, :1] > scores_by_record[:, 1:]).sum().float() / count

    def setewc(self):
        self._ewc = True

    def iter_train(self, only_cached=False, _top_epoch=False, ewc_params=None):
        epoch = -1
        base_path = util.path_model_trainer(self.ranker, self.vocab, self, self.dataset)
        context = {
            'epoch': epoch,
            'batch_size': self.config['batch_size'],
            'batches_per_epoch': self.config['batches_per_epoch'],
            'num_microbatches': 1 if self.config['grad_acc_batch'] == 0 else self.config['batch_size'] // self.config['grad_acc_batch'],
            'device': self.device,
            'base_path': base_path,
        }

        files = trainers.misc.PathManager(base_path)
        self.logger.info(f'train path: {base_path}, batches_per_epoch : {context["batches_per_epoch"]}')
        b_count = context['batches_per_epoch'] * context['num_microbatches'] * self.batch_size

        ranker = self.ranker.to(self.device)
        optimizer = self.create_optimizer()

        if f'{epoch}.p' not in files['weights']:
            ranker.save(files['weights'][f'{epoch}.p'])
        if f'{epoch}.p' not in files['optimizer']:
            torch.save(optimizer.state_dict(), files['optimizer'][f'{epoch}.p'])
        
        context.update({
            'ranker': lambda: ranker,
            'ranker_path': files['weights'][f'{epoch}.p'],
            'optimizer': lambda: optimizer,
            'optimizer_path': files['optimizer'][f'{epoch}.p'],
        })

        #load ranker/optimizer
        if _top_epoch:
            # trainer.pipeline=msmarco_train_bm25_k1-0.82_b-0.68.100_mspairs
            __path="/".join(base_path.split("/")[:-1])+"/"+self.config['pipeline'] 
            self.logger.info(f'loading prev model : {__path}')
            w_path = os.path.join(__path, 'weights','-2.p')
            oppt_path = os.path.join(__path, 'optimizer','-2.p')
            ranker.load(w_path)
            optimizer.load_state_dict(torch.load(oppt_path))

            if w_path!=files['weights']['-2.p']:
                copyfile(w_path,files['weights']['-2.p'] )
                copyfile(oppt_path,files['optimizer']['-2.p'] )

            context.update({ 
                'ranker': lambda: ranker,
                'ranker_path': files['weights'][f'-2.p'],
                'optimizer': lambda: optimizer,
                'optimizer_path': files['optimizer'][f'-2.p'],
            })

        yield context # before training

        while True:
            context = dict(context)


            epoch = context['epoch'] = context['epoch'] + 1
            if epoch in files['complete.tsv']:
                context.update({
                    'loss': files['loss.txt'][epoch],
                    'data_loss': files['data_loss.txt'][epoch],
                    'losses': {},
                    'acc': files['acc.tsv'][epoch],
                    'unsup_acc': files['unsup_acc.tsv'][epoch],
                    'ranker': _load_ranker(ranker, files['weights'][f'{epoch}.p']),
                    'ranker_path': files['weights'][f'{epoch}.p'],
                    'optimizer': _load_optimizer(optimizer, files['optimizer'][f'{epoch}.p']),
                    'optimizer_path': files['optimizer'][f'{epoch}.p'],
                    'cached': True,
                })
                if not only_cached:
                    self.fast_forward(b_count) # skip this epoch
                yield context
                continue

            if only_cached:
                break # no more cached

            # forward to previous versions (if needed)
            ranker = context['ranker']()
            optimizer = context['optimizer']()
            ranker.train()

            context.update({
                'loss': 0.0,
                'losses': {},
                'acc': 0.0,
                'unsup_acc': 0.0,
            })


            # LOAD EWC
            _optpar_params = {}
            _fisher_params = {}
            print("EWC LOAD")
            for task in ewc_params.tasks:
                print("Aded task", task)
                _optpar_params[task] = {}
                _fisher_params[task] = {}

                for name, param in ranker.named_parameters():
                    if param.requires_grad:
                #for name, param in ranker.state_dict().items():   
                        optpar_path = ewc_params.getOptpar(task, name)
                        fisher_path = ewc_params.getFisher(task, name)
                        
                        _optpar_params[task][name] = pickle.load(open(optpar_path,"rb")).cuda()
                        _fisher_params[task][name] = pickle.load(open(fisher_path,"rb")).cuda()

            with tqdm(leave=False, total=b_count, ncols=100, desc=f'train {epoch}') as pbar:
                for b in range(context['batches_per_epoch']):
                    for _ in range(context['num_microbatches']):
                        self.epoch = epoch
                        train_batch_result = self.train_batch()
                        #print("BATCH FINALIZED")
                        losses = train_batch_result['losses']
                        loss_weights = train_batch_result['loss_weights']
                        acc = train_batch_result.get('acc')
                        unsup_acc = train_batch_result.get('unsup_acc')


                        #EWC
                        if not self._ewc:
                            for task in ewc_params.tasks:
                                #print("EWC task", task)
                                for name, param in ranker.named_parameters():
                                    if param.requires_grad:
                                        fisher = _fisher_params[task][name]
                                        optpar = _optpar_params[task][name]
                                        losses['data'] += (fisher * (optpar - param).pow(2)).sum()  * ewc_params.ewc_lambda

                                #for name, param in ranker.state_dict().items():
                                    #print(name,param)
                                    # optpar_path = ewc_params.getOptpar(task, name)
                                    # fisher_path = ewc_params.getFisher(task, name)

                                    # optpar = torch.load(optpar_path) #pickle.load(open(optpar_path,"rb"))
                                    # #torch.load('featurs.pkl',map_location=torch.device('cpu'))
                                    # param = (optpar - param)
                                    # #torch.cuda.empty_cache()
                                    # #gc.collect()

                                    # fisher = torch.load(fisher_path) #pickle.load(open(fisher_path,"rb"))
                                    # losses['data'] += (fisher * param.pow(2)).sum() * ewc_params.ewc_lambda
                                    # #torch.cuda.empty_cache()
                                    # #gc.collect()

                                    
                        losses['data'] = losses['data'].mean()
                        loss = sum(losses[k] * loss_weights.get(k, 1.) for k in losses) / context['num_microbatches']

                        context['loss'] += loss.item()
                        for lname, lvalue in losses.items():
                            context['losses'].setdefault(lname, 0.)
                            context['losses'][lname] += lvalue.item() / context['num_microbatches']

                        if acc is not None:
                            context['acc'] += acc.item() / context['num_microbatches']
                        if unsup_acc is not None:
                            context['unsup_acc'] += unsup_acc.item() / context['num_microbatches']

                        if loss.grad_fn is not None:
                            if hasattr(optimizer, 'backward'):
                                optimizer.backward(loss)
                            else:
                                loss.backward()
                        else:
                            self.logger.warn('loss has no grad_fn; skipping batch')
                        pbar.update(self.batch_size)

                    postfix = {
                        'loss': context['loss'] / (b + 1),
                    }
                    for lname, lvalue in context['losses'].items():
                        if lname in loss_weights and loss_weights[lname] != 1.:
                            postfix[f'{lname}({loss_weights[lname]})'] = lvalue / (b + 1)
                        else:
                            postfix[lname] = lvalue / (b + 1)

                    if postfix['loss'] == postfix['data']:
                        del postfix['data']

                    pbar.set_postfix(postfix)
                    optimizer.step()
                    optimizer.zero_grad()

            context.update({
                'ranker': lambda: ranker,
                'ranker_path': files['weights'][f'{epoch}.p'],
                'optimizer': lambda: optimizer,
                'optimizer_path': files['optimizer'][f'{epoch}.p'],
                'loss': context['loss'] / context['batches_per_epoch'],
                'losses': {k: v / context['batches_per_epoch'] for k, v in context['losses'].items()},
                'acc': context['acc'] / context['batches_per_epoch'],
                'unsup_acc': context['unsup_acc'] / context['batches_per_epoch'],
                'cached': False,
            })

            if self._ewc:
                #print("EWC iteration")
                params_for_ewc = {n: p for n, p in ranker.named_parameters() if p.requires_grad}
                yield params_for_ewc
                #yield  ranker.state_dict()
                # EWC implementation from https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb
                # fisher_dict = {}
                # optpar_dict = {}
                
                # # gradients accumulated can be used to calculate fisher
                # for name, param in ranker.state_dict().items():
                #     # print(name)
                #     # print(param)
                #     optpar_dict[name] = param.clone()
                #     fisher_dict[name] = param.clone().pow(2)

                # for name, param in ranker.named_parameters():
                #     print(name)
                #     print(param.grad)
                #     optpar_dict[name] = param.data.clone()
                #     fisher_dict[name] = param.grad.data.clone().pow(2)


                #pbar.set_postfix(postfix)
                optimizer.step()
                optimizer.zero_grad()
                #yield {'fisher':fisher_dict, 'optpar':optpar_dict}

            # save stuff
            ranker.save(files['weights'][f'{epoch}.p'])
            torch.save(optimizer.state_dict(), files['optimizer'][f'{epoch}.p'])
            files['loss.txt'][epoch] = context['loss']
            for lname, lvalue in context['losses'].items():
                files[f'loss_{lname}.txt'][epoch] = lvalue
            files['acc.tsv'][epoch] = context['acc']
            files['unsup_acc.tsv'][epoch] = context['unsup_acc']
            files['complete.tsv'][epoch] = 1 # mark as completed

            yield context



def _load_optimizer(optimizer, state_path):
    def _wrapped():
        optimizer.load_state_dict(torch.load(state_path))
        return optimizer
    return _wrapped


def _load_ranker(ranker, state_path):
    def _wrapped():
        ranker.load(state_path)
        return ranker
    return _wrapped
