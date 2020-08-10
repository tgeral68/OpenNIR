#!/bin/bash
BERT_MODEL_PARAMS="trainer.grad_acc_batch=1 valid_pred.batch_size=4 test_pred.batch_size=4"

#------------
# commands to calculate 1st results

# drmm

# Trains msmarco -> Test Msmarco, from scratch
#CUDA_VISIBLE_DEVICES=1 python -m onir.bin.pipeline pipeline=jesus modelspace=try1 vocab.source=glove vocab.variant=cc-42b-300d config/msmarco ranker=drmm pipeline.test=true ranker.add_runscore=True data_dir=../data

# from trained msmarco -> Only test msmarco
#CUDA_VISIBLE_DEVICES=0 python -m onir.bin.pipeline pipeline=jesus modelspace=try1-ft vocab.source=glove vocab.variant=cc-42b-300d config/msmarco  ranker=drmm pipeline.onlytest=true ranker.add_runscore=True data_dir=../data pipeline.test=true pipeline.finetune=true trainer.pipeline=msmarco_train_bm25_k1-0.82_b-0.68.100_mspairs


# from trained msmarco -> Only test cord19 (5 folds)
#for fold in {2..2}; do
#CUDA_VISIBLE_DEVICES=0 python -m onir.bin.pipeline pipeline=jesus modelspace=try1-ft vocab.source=glove vocab.variant=cc-42b-300d config/msmarco config/covidj/test${fold} ranker=drmm pipeline.onlytest=true ranker.add_runscore=True data_dir=../data pipeline.test=true pipeline.finetune=true trainer.pipeline=msmarco_train_bm25_k1-0.82_b-0.68.100_mspairs >drmm_$fold.out 2>drmm_$fold.err
#done


# from trained msmarco -> retrain cor19 -> test cord19
prev_models=("msmarco_train_bm25_k1-0.82_b-0.68.100_mspairs"  
"covid_trf1-rnd5-quest_bm25_k1-3.9_b-0.55.1000_2020-07-16_bs-text_2020filter_bsoverride-rnd5-query_rr-title_abs" 
"covid_trf2-rnd5-quest_bm25_k1-3.9_b-0.55.1000_2020-07-16_bs-text_2020filter_bsoverride-rnd5-query_rr-title_abs" 
"covid_trf3-rnd5-quest_bm25_k1-3.9_b-0.55.1000_2020-07-16_bs-text_2020filter_bsoverride-rnd5-query_rr-title_abs" 
"covid_trf4-rnd5-quest_bm25_k1-3.9_b-0.55.1000_2020-07-16_bs-text_2020filter_bsoverride-rnd5-query_rr-title_abs" 
"covid_trf5-rnd5-quest_bm25_k1-3.9_b-0.55.1000_2020-07-16_bs-text_2020filter_bsoverride-rnd5-query_rr-title_abs" 
)

#for fold in {2..2}; do
#    CUDA_VISIBLE_DEVICES=0 python -m onir.bin.pipeline pipeline=jesus modelspace=try1-ft vocab.source=glove vocab.variant=cc-42b-300d config/covidj/fold${fold} ranker=drmm ranker.add_runscore=True data_dir=../data pipeline.test=true pipeline.finetune=true trainer.pipeline=${prev_models[0]} >drmm_$fold.retrained.out 2>drmm_$fold.retrained.err
#done




# from trained msmarco -> retrain cor19 -> test msmarco
for fold in {2..2}; do
CUDA_VISIBLE_DEVICES=1 python -m onir.bin.pipeline pipeline=jesus modelspace=try1-ft vocab.source=glove vocab.variant=cc-42b-300d config/covidj/fold${fold} config/msmarco/judgeddev ranker=drmm pipeline.onlytest=true ranker.add_runscore=True data_dir=../data pipeline.test=true pipeline.finetune=true trainer.pipeline=${prev_models[2]}  >drmm_$fold.out 2>drmm_$fold.err
done

#--------

