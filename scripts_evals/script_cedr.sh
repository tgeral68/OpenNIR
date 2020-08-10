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
#    CUDA_VISIBLE_DEVICES=1 python -m onir.bin.pipeline pipeline=jesus modelspace=try1-ft config/msmarco config/covidj/test${fold} config/cedr/knrm $BERT_MODEL_PARAMS vocab.bert_weights=try1 pipeline.onlytest=true ranker.add_runscore=True data_dir=../data pipeline.test=true pipeline.finetune=true trainer.pipeline=msmarco_train_bm25_k1-0.82_b-0.68.100_mspairs >cedr_$fold.out 2>cedr_$fold.err
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
#    CUDA_VISIBLE_DEVICES=0 python -m onir.bin.extract_bert_weights modelspace=try1-ft pipeline.bert_weights=try1 config/covidj/fold${fold} pipeline.test=true config/vanilla_bert $BERT_MODEL_PARAMS ranker.add_runscore=True data_dir=../data >weightbert_$fold.out 2>weightbert_$fold.err
#    CUDA_VISIBLE_DEVICES=0 python -m onir.bin.pipeline pipeline=jesus modelspace=try1-ft config/covidj/fold${fold} config/cedr/knrm $BERT_MODEL_PARAMS vocab.bert_weights=try1 ranker.add_runscore=True data_dir=../data pipeline.test=true pipeline.finetune=true trainer.pipeline=${prev_models[0]} >cedr_$fold.retrained.out 2>cedr_$fold.retrained.err
#done







# from trained msmarco -> retrain cor19 -> test msmarc0 (onlytest)
for fold in {2..2}; do
    CUDA_VISIBLE_DEVICES=1 python -m onir.bin.pipeline pipeline=jesus modelspace=try1-ft config/covidj/fold${fold} config/msmarco/judgeddev config/cedr/knrm $BERT_MODEL_PARAMS vocab.bert_weights=try1 ranker.add_runscore=True data_dir=../data pipeline.test=true pipeline.finetune=true trainer.pipeline=${prev_models[2]} pipeline.onlytest=true >cedr_$fold.out 2>cedr_$fold.err
done


# from trained msmarco -> retrain cor19 -> test msmarco

#--------





#------------
# commands to calculate 1st results


#CUDA_VISIBLE_DEVICES=2 python -m onir.bin.extract_bert_weights modelspace=try1 pipeline.bert_weights=try1 config/msmarco pipeline.test=true config/vanilla_bert $BERT_MODEL_PARAMS ranker.add_runscore=True data_dir=../data

