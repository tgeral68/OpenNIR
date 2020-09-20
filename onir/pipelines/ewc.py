import os
import json
from onir import util, pipelines
import onir
import pickle

@pipelines.register('ewc')
class EWCPipeline(pipelines.BasePipeline):
    name = None

    @staticmethod
    def default_config():
        return {
            'max_epoch': 1000,
            'early_stop': 20,
            'warmup': -1,
            'val_metric': 'map',
            'purge_weights': True,
            'test': False,
            'initial_eval': False,
            'skip_ds_init': False,
            'only_cached': False,
            'onlytest': False,
            'finetune': False,
            'savefile': '_',
            'dataset': '',
        }

    def __init__(self, config, trainer, valid_pred, test_pred, logger):
        super().__init__(config, logger)
        self.trainer = trainer
        self.valid_pred = valid_pred
        self.test_pred = test_pred

    def run(self):
        validator = self.valid_pred.pred_ctxt()

        top_epoch, top_value, top_train_ctxt, top_valid_ctxt = None, None, None, None
        prev_train_ctxt = None

        file_output = {
            'ranker': self.trainer.ranker.path_segment(),
            'vocab': self.trainer.vocab.path_segment(),
            'trainer': self.trainer.path_segment(),
            'dataset': self.trainer.dataset.path_segment(),
            'valid_ds': self.valid_pred.dataset.path_segment(),
            'validation_metric': self.config['val_metric'],
            'logfile': util.path_log()
        }

        # initialize dataset(s)
        if not self.config['skip_ds_init']:
            self.trainer.dataset.init(force=False)
            self.valid_pred.dataset.init(force=False)
            if self.config['test']:
                self.test_pred.dataset.init(force=False)
    
        base_path_g = None
        ewc_params = {}

        # initialize EWC
        base_path = util.path_model_trainer(self.trainer.ranker, self.trainer.vocab, self.trainer, self.trainer.dataset)
        ewc_path = "/".join(base_path.split("/")[:-1])+"/ewc.pickle"
        task_name = self.trainer.dataset.path_segment().split("_")[0]

        try:
            my_ewc = pickle.load(open(ewc_path,"rb"))
        except (OSError, IOError) as e:
            my_ewc = EWCValues(path=ewc_path)
        #print("EWC PATH: ",ewc_path, task_name)

        _train_it = self.trainer.iter_train(only_cached=self.config['only_cached'], _top_epoch=self.config.get('finetune'), ewc_params=my_ewc)
        for train_ctxt in _train_it:
        
            if self.config.get('onlytest'):
                base_path_g = train_ctxt['base_path']
                self.logger.debug(f'[jesus] skipping training')
                top_train_ctxt=train_ctxt
                break


            if prev_train_ctxt is not None and top_epoch is not None and prev_train_ctxt is not top_train_ctxt:
                self._purge_weights(prev_train_ctxt)

            if train_ctxt['epoch'] >= 0 and not self.config['only_cached']:
                message = self._build_train_msg(train_ctxt)

                if train_ctxt['cached']:
                    self.logger.debug(f'[train] [cached] {message}')
                else:
                    self.logger.debug(f'[train] {message}')

            if train_ctxt['epoch'] == -1 and not self.config['initial_eval']:
                continue

            valid_ctxt = dict(validator(train_ctxt))

            message = self._build_valid_msg(valid_ctxt)
            if valid_ctxt['epoch'] >= self.config['warmup']:
                if self.config['val_metric'] == '':
                    top_epoch = valid_ctxt['epoch']
                    top_train_ctxt = train_ctxt
                    top_valid_ctxt = valid_ctxt
                elif top_value is None or valid_ctxt['metrics'][self.config['val_metric']] > top_value:
                    message += ' <---'
                    top_epoch = valid_ctxt['epoch']
                    top_value = valid_ctxt['metrics'][self.config['val_metric']]
                    if top_train_ctxt is not None:
                        self._purge_weights(top_train_ctxt)
                    top_train_ctxt = train_ctxt
                    top_valid_ctxt = valid_ctxt
            else:
                if prev_train_ctxt is not None:
                    self._purge_weights(prev_train_ctxt)

            if not self.config['only_cached']:
                if valid_ctxt['cached']:
                    self.logger.debug(f'[valid] [cached] {message}')
                else:
                    self.logger.info(f'[valid] {message}')

            if top_epoch is not None:
                epochs_since_imp = valid_ctxt['epoch'] - top_epoch
                if self.config['early_stop'] > 0 and epochs_since_imp >= self.config['early_stop']:
                    self.logger.warn('stopping after epoch {epoch} ({early_stop} epochs with no '
                                     'improvement to {val_metric})'.format(**valid_ctxt, **self.config))
                    break

            if train_ctxt['epoch'] >= self.config['max_epoch']:
                self.logger.warn('stopping after epoch {max_epoch} (max_epoch)'.format(**self.config))
                break

            prev_train_ctxt = train_ctxt

        if not self.config.get('onlytest'):
            self.logger.info('top validation epoch={} {}={}'.format(top_epoch, self.config['val_metric'], top_value))

            self.logger.info(f'[jesus: top_train_ctxt] {top_train_ctxt}')
            file_output.update({
                'valid_epoch': top_epoch,
                'valid_run': top_valid_ctxt['run_path'],
                'valid_metrics': top_valid_ctxt['metrics'],
            })



        # save top train epoch for faster testing without needing the retraining phase
        if not self.config.get('onlytest'):
            #pickle.dump(top_epoch, open( top_train_ctxt['base_path']+"/top_epoch.pickle", "wb") )
            # move best to -2.p

            self.trainer.save_best(top_epoch, top_train_ctxt['base_path'])

            # EWC
            # Recover parms from model after extra epoch
            self.trainer.setewc()
            ewc_params = next(_train_it)
            #ewc_params.cpu()

            # EWC implementation from https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb
            fisher_dict = {}
            optpar_dict = {}
            
            # gradients accumulated can be used to calculate fisher
            for name, param in ewc_params.items():
                # print(name)
                # print(param)

                optpar_path = my_ewc.getOptpar(task_name, name)
                pickle.dump(param, open(optpar_path, "wb"))
                #optpar_dict[name] = param.clone()
                
                fisher_path = my_ewc.getFisher(task_name, name)
                param = param.pow(2)
                pickle.dump(param, open(fisher_path, "wb"))
                #fisher_dict[name] = param.clone().pow(2)

                my_ewc.addNew(task_name, name)
                
            ####
            # load EWC object
            # get task ID  -> up before the loop train_iter
            ####
            #print("EWC params", ewc_params)
            #my_ewc.setValues(task_name, ewc_params)

            #my_ewc.setValues(task_name, {'fisher':fisher_dict, 'optpar':optpar_dict})
            ###
            # save EWC object
            ###
            pickle.dump(my_ewc, open(ewc_path, "wb"))

        

        if self.config.get('onlytest'): # for onlytest use also finetune=true, to load best epoch at first iteration
            self.logger.debug(f'[jesus] loading top context')
            #top_epoch = pickle.load(open(base_path_g+"/top_epoch.pickle", "rb"))
            #self.logger.debug(f'[jesus] loading top context ... {top_epoch} epoch')
            #top_train_ctxt = self.trainer.trainCtx(top_epoch)
            self.logger.debug(f'[jesus] Top epoch context: {dict(top_train_ctxt)}')

        
        if self.config['test']:
            self.logger.info(f'Starting load ranker')
            top_train_ctxt['ranker'] = onir.trainers.base._load_ranker(top_train_ctxt['ranker'](), top_train_ctxt['ranker_path'])

            self.logger.info(f'Starting test predictor run')
            with self.logger.duration('testing'):
                test_ctxt = self.test_pred.run(top_train_ctxt)

            file_output.update({
                'test_ds': self.test_pred.dataset.path_segment(),
                'test_run': test_ctxt['run_path'],
                'test_metrics': test_ctxt['metrics'],
            })

        with open(util.path_modelspace() + '/val_test.jsonl', 'at') as f:
            json.dump(file_output, f)
            f.write('\n')

        if not self.config.get('onlytest'):
            self.logger.info('valid run at {}'.format(valid_ctxt['run_path']))
        if self.config['test']:
            self.logger.info('test run at {}'.format(test_ctxt['run_path']))
        if not self.config.get('onlytest'):
            self.logger.info('valid ' + self._build_valid_msg(top_valid_ctxt))
        if self.config['test']:
            self.logger.info('test  ' + self._build_valid_msg(test_ctxt))
            self._write_metrics_file(test_ctxt)

    def _build_train_msg(self, ctxt):
        delta_acc = ctxt['acc'] - ctxt['unsup_acc']
        msg_pt1 = 'epoch={epoch} loss={loss:.4f}'.format(**ctxt)
        msg_pt2 = 'acc={acc:.4f} unsup_acc={unsup_acc:.4f} ' \
                  'delta_acc={delta_acc:.4f}'.format(**ctxt, delta_acc=delta_acc)
        losses = ''
        if ctxt['losses'] and ({'data'} != ctxt['losses'].keys() or ctxt['losses']['data'] != ctxt['loss']):
            losses = []
            for lname, lvalue in ctxt['losses'].items():
                losses.append(f'{lname}={lvalue:.4f}')
            losses = ' '.join(losses)
            losses = f' ({losses})'
        return f'{msg_pt1}{losses} {msg_pt2}'


    def _build_valid_msg(self, ctxt):
        message = ['epoch=' + str(ctxt['epoch'])]
        for metric, value in sorted(ctxt['metrics'].items()):
            message.append('{}={:.4f}'.format(metric, value))
            if metric == self.config['val_metric']:
                message[-1] = '[' + message[-1] + ']'
        return ' '.join(message)

    def _purge_weights(self, ctxt):
        if self.config['purge_weights']:
            if os.path.exists(ctxt['ranker_path']):
                os.remove(ctxt['ranker_path'])
            if os.path.exists(ctxt['optimizer_path']):
                os.remove(ctxt['optimizer_path'])

    def _write_metrics_file(self, ctxt):
        outputdir = ""
        filename = os.path.join(outputdir, self.config.get('savefile')) # format model-namemodel_train-dataset_test-dataset_train-dataset_test-dataset

        with open(filename, "w") as f:
            for metric, value in sorted(ctxt['metrics'].items()):
                f.write('{}\t{:.4f}'.format(metric, value))
                f.write('\n')


class EWCValues():

    def __init__(self, ewc_lambda=0.4, path=None):
        self.fisher_dict = {}
        self.optpar_dict = {}
        self.ewc_lambda = ewc_lambda
        self.tasks = set()
        self.param_name = set()
        self.path = "/".join(path.split("/")[:-1])

    def load(self):
        pass

    def save(self):
        pass


    def addNew(self, task_name, param_name):
        self.tasks.add(task_name)
        self.param_name.add(param_name)

    def getOptpar(self,task_name, param_name):
        return self.path+"/"+task_name+"_optpar_"+param_name+".pickle"

    def getFisher(self, task_name, param_name):
        return self.path+"/"+task_name+"_fisher_"+param_name+".pickle"

    def setValues(self, task_name, ewc_params):
        self.fisher_dict[task_name] = ewc_params['fisher']
        self.optpar_dict[task_name] = ewc_params['optpar']
        self.tasks.add(task_name)