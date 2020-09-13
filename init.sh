#!/bin/bash
python -m onir.bin.init_dataset vocab.source=glove vocab.variant=cc-42b-300d config/microblog data_dir=../data
