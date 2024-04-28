#!/usr/bin/bash
python train.py \
--msb hdb --lsb hd  \
--upscale 2 2 \
--act-fn relu --n-filters 64 \