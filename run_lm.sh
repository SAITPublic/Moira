#!/usr/bin/env bash

N_TARGET="3.0 2.5 2.0 1.5 1.1 1.0"
LLAMA="meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf huggyllama/llama-30b meta-llama/Llama-2-70b-hf huggyllama/llama-65b"
QWEN="Qwen/Qwen3.5-27B"
MISTRAL="mistralai/Mistral-7B-v0.3"

for model in $LLAMA
do
    for var in $N_TARGET
    do
        echo $var
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py --model $model \
                                                             --data wikitext2 \
                                                             --target_wires $var \
                                                             --save_model "./ckpt/${model##*/}-moira-${var}" \
                                                             --eval_zero_shot
    done
done
for var in $N_TARGET
do
    echo $var
    CUDA_VISIBLE_DEVICES=0 python3 main.py --model mistralai/Mistral-7B-v0.3 \
                                           --data wikitext2 \
                                           --target_wires $var \
                                           --save_model "./ckpt/${model##*/}-moira-${var}" \
                                           --eval_zero_shot
    done
for var in $N_TARGET
do
    echo $var
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py --model Qwen/Qwen3.5-27B \
                                                 --data wikitext2 \
                                                 --target_wires $var \
                                                 --save_model "./ckpt/${model##*/}-moira-${var}" \
                                                 --eval_zero_shot
done

