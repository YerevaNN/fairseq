#!/bin/bash
TOTAL_NUM_UPDATES=2036 # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=61 # 6 percent of the number of updates
LR=1e-05 # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=2 # Batch size.
DATASET_NAME="freesolv-bin"
PATH1="/home/gayane/BartLM/"
PATH2="/processed"
NUM_CLASSES=1
BART_PATH=/home/gayane/BartLM/Bart/chemical/checkpoints/chemical/checkpoints/ft.bart_large.sentpred.ms16.uf1.mu2296.dr0.1.atdr0.1.actdr0.0.wd0.01.adam.beta9999.eps1e-08.clip0.1.lr3e-05.warm367.fp16.ngpu8/
declare -a lr=(0.0005)
for i in $lr
do
    echo $i
    CUDA_VISIBLE_DEVICES=0 fairseq-train $PATH1$DATASET_NAME$PATH2 --update-freq 8 \
        --restore-file $BART_PATH --wandb-project Fine_Tune_freesolv \
        --batch-size $MAX_SENTENCES --max-tokens 4400 --task sentence_prediction \
        --add-prev-output-tokens --layernorm-embedding --share-all-embeddings \
        --share-decoder-input-output-embed --reset-optimizer --reset-dataloader \
        --reset-meters --required-batch-size-multiple 1 --init-token 0 --arch bart_large \
        --criterion sentence_prediction --dropout 0.1 --attention-dropout 0.1 \
        --weight-decay 0.01 --optimizer sgd --clip-norm 0.0 --lr-scheduler polynomial_decay \
        --lr $i --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES --fp16 \
        --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 --max-epoch 11 \
        --find-unused-parameters --regression-target --num-classes $NUM_CLASSES 
        # --best-checkpoint-metric RMSE

done
# --no-epoch-checkpoints 
# CUDA_VISIBLE_DEVICES=1 fairseq-train $PATH1$DATASET_NAME$PATH2 --update-freq 8 \
#         --restore-file $BART_PATH --wandb-project Fine_Tune_freesolv \
#         --batch-size $MAX_SENTENCES --max-tokens 4400 --task sentence_prediction \
#         --add-prev-output-tokens --layernorm-embedding --share-all-embeddings \
#         --share-decoder-input-output-embed --reset-optimizer --reset-dataloader \
#         --reset-meters --required-batch-size-multiple 1 --init-token 0 --arch bart_large \
#         --criterion sentence_prediction --dropout 0.1 \
#         --attention-dropout 0.1 --weight-decay 0.01 --optimizer sgd --clip-norm 0.0 \
#         --lr-scheduler polynomial_decay --lr 6e-07 --total-num-update $TOTAL_NUM_UPDATES \
#         --warmup-updates $WARMUP_UPDATES --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 \
#         --fp16-scale-window 128 --max-epoch 10 --find-unused-parameters --regression-target \
#         # --separator_token 2 --no_epoch_checkpoints 
# ##--best-checkpoint-metric accuracy --maximize-best-checkpoint-metric
# CUDA_VISIBLE_DEVICES=1 fairseq-train $PATH1$DATASET_NAME$PATH2 --update-freq 8 --no-epoch-checkpoints \
#         --restore-file $BART_PATH --wandb-project Fine_Tune_freesolv \
#         --batch-size $MAX_SENTENCES --max-tokens 4400 --task sentence_prediction \
#         --add-prev-output-tokens --layernorm-embedding --share-all-embeddings \
#         --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters \
#         --required-batch-size-multiple 1 --init-token 0 --arch bart_large \
#         --criterion sentence_prediction --num-classes $NUM_CLASSES --dropout 0.1 \
#         --attention-dropout 0.1 --weight-decay 0.01 --optimizer sgd --clip-norm 0.0 \
#         --lr-scheduler polynomial_decay --lr 8e-07 --total-num-update $TOTAL_NUM_UPDATES \
#         --warmup-updates $WARMUP_UPDATES --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 \
#         --fp16-scale-window 128 --max-epoch 10 --find-unused-parameters --regression-target 
# ##--best-checkpoint-metric rmse --maximize-best-checkpoint-metric
# CUDA_VISIBLE_DEVICES=1 fairseq-train $PATH1$DATASET_NAME$PATH2 --update-freq 8 --no-epoch-checkpoints \
#         --restore-file $BART_PATH --wandb-project Fine_Tune_freesolv \
#         --batch-size $MAX_SENTENCES --max-tokens 4400 --task sentence_prediction \
#         --add-prev-output-tokens --layernorm-embedding --share-all-embeddings \
#         --share-decoder-input-output-embed --reset-optimizer --reset-dataloader \
#         --reset-meters --required-batch-size-multiple 1 --init-token 0 --arch bart_large \
#         --criterion sentence_prediction --num-classes $NUM_CLASSES --dropout 0.1 \
#         --attention-dropout 0.1 --weight-decay 0.01 --optimizer sgd --clip-norm 0.0 \
#         --lr-scheduler polynomial_decay --lr 1e-06 --total-num-update $TOTAL_NUM_UPDATES \
#         --warmup-updates $WARMUP_UPDATES --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 \
#         --fp16-scale-window 128 --max-epoch 10 --find-unused-parameters --regression-target 
# ##--best-checkpoint-metric rmse --maximize-best-checkpoint-metric
# CUDA_VISIBLE_DEVICES=1 fairseq-train $PATH1$DATASET_NAME$PATH2 --update-freq 8 --no-epoch-checkpoints \
#         --restore-file $BART_PATH --wandb-project Fine_Tune_freesolv \
#         --batch-size $MAX_SENTENCES --max-tokens 4400 --task sentence_prediction \
#         --add-prev-output-tokens --layernorm-embedding --share-all-embeddings \
#         --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters \
#         --required-batch-size-multiple 1 --init-token 0 --arch bart_large \
#         --criterion sentence_prediction --num-classes $NUM_CLASSES --dropout 0.1 \
#         --attention-dropout 0.1 --weight-decay 0.01 --optimizer sgd --clip-norm 0.0 \
#         --lr-scheduler polynomial_decay --lr 2e-06 --total-num-update $TOTAL_NUM_UPDATES \
#         --warmup-updates $WARMUP_UPDATES --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 \
#         --fp16-scale-window 128 --max-epoch 10 --find-unused-parameters --regression-target 
# ##--best-checkpoint-metric rmse --maximize-best-checkpoint-metric
# CUDA_VISIBLE_DEVICES=1 fairseq-train $PATH1$DATASET_NAME$PATH2 --update-freq 8 --no-epoch-checkpoints \
#         --restore-file $BART_PATH --wandb-project Fine_Tune_freesolv \
#         --batch-size $MAX_SENTENCES --max-tokens 4400 --task sentence_prediction \
#         --add-prev-output-tokens --layernorm-embedding --share-all-embeddings \
#         --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters \
#         --required-batch-size-multiple 1 --init-token 0 --arch bart_large \
#         --criterion sentence_prediction --num-classes $NUM_CLASSES --dropout 0.1 \
#         --attention-dropout 0.1 --weight-decay 0.01 --optimizer sgd --clip-norm 0.0 \
#         --lr-scheduler polynomial_decay --lr 4e-06 --total-num-update $TOTAL_NUM_UPDATES \
#         --warmup-updates $WARMUP_UPDATES --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 \
#         --fp16-scale-window 128 --max-epoch 10 --find-unused-parameters --regression-target 
# ##--best-checkpoint-metric rmse --maximize-best-checkpoint-metric
# CUDA_VISIBLE_DEVICES=1 fairseq-train $PATH1$DATASET_NAME$PATH2 --update-freq 8 --no-epoch-checkpoints \
#         --restore-file $BART_PATH --wandb-project Fine_Tune_freesolv \
#         --batch-size $MAX_SENTENCES --max-tokens 4400 --task sentence_prediction \
#         --add-prev-output-tokens --layernorm-embedding --share-all-embeddings \
#         --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters \
#         --required-batch-size-multiple 1 --init-token 0 --arch bart_large \
#         --criterion sentence_prediction --num-classes $NUM_CLASSES --dropout 0.1 \
#         --attention-dropout 0.1 --weight-decay 0.01 --optimizer sgd --clip-norm 0.0 \
#         --lr-scheduler polynomial_decay --lr 6e-06 --total-num-update $TOTAL_NUM_UPDATES \
#         --warmup-updates $WARMUP_UPDATES --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 \
#         --fp16-scale-window 128 --max-epoch 10 --find-unused-parameters --regression-target 
# ##--best-checkpoint-metric rmse --maximize-best-checkpoint-metric