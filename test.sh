#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m onir.bin.pipeline pipeline=catfog modelspace=configX data_dir=../data vocab.source=glove vocab.variant=cc-42b-300d ranker=drmm ranker.add_runscore=True   config/mix3/

