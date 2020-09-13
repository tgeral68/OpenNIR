#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -m onir.bin.pipeline data_dir=../data  vocab.source=glove vocab.variant=cc-42b-300d   config/trivial/bm25 ranker.add_runscore=True   config/microblog >output/bm25_mb.out 2>output/bm25_mb.err
