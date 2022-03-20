#!/bin/bash
TOTAL_NUM_UPDATES=2296 # 10 epochs through RTE for bsz 16
WARMUP_UPDATES=143 # 6 percent of the number of updates  int(total_num_udpates*0.16)
LR=1e-05 # Peak LR for polynomial LR scheduler.
NUM_CLASSES=2
MAX_SENTENCES=2 # Batch size.
DATASET_NAME="bbbp"
PATH1="/home/gayane/BartLM/"
PATH2="/processed"
num_data_loaders=1

echo $LR 3e-05
# PATH="$PATH1$DATASET_NAME$PATH2"
BART_PATH=/home/gayane/BartLM/Bart/chemical/checkpoints/chemical/checkpoints/ft.bart_large.sentpred.ms16.uf1.mu2296.dr0.1.atdr0.1.actdr0.0.wd0.01.adam.beta9999.eps1e-08.clip0.1.lr3e-05.warm367.fp16.ngpu8/
CUDA_VISIBLE_DEVICES=0 fairseq-train $PATH1$DATASET_NAME$PATH2 \
         --update-freq 8 --max-target-positions 128 --max-source-positions 128  --restore-file $BART_PATH \
         --wandb-project Fine_Tune_BBBP --batch-size $MAX_SENTENCES --max-tokens 4400 \
         --task sentence_prediction --add-prev-output-tokens --layernorm-embedding \
         --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer \
         --reset-dataloader --reset-meters --required-batch-size-multiple 1 --init-token 0 \
         --arch bart_large --criterion sentence_prediction --num-classes $NUM_CLASSES --dropout 0.2 \
         --attention-dropout 0.2 --relu-dropout 0.1 --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --clip-norm 0.1 \
         --lr-scheduler polynomial_decay --lr 3e-05 --total-num-update $TOTAL_NUM_UPDATES \
         --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 \
         --fp16-scale-window 128 --max-epoch 10 --find-unused-parameters --max-update  $TOTAL_NUM_UPDATES \
         --num-workers $num_data_loaders --skip-invalid-size-inputs-valid-test \
         --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
         --no-epoch-checkpoints --warmup-updates $WARMUP_UPDATES
#  --log-format json --log-interval 1 
        
# CUDA_VISIBLE_DEVICES=1 fairseq-train $PATH1$DATASET_NAME$PATH2 --update-freq 8 --no-epoch-checkpoints \
#         --restore-file $BART_PATH --wandb-project Fine_Tune_BBBP --batch-size $MAX_SENTENCES \
#         --max-tokens 4400 --task sentence_prediction --add-prev-output-tokens --layernorm-embedding \
#         --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader \
#         --reset-meters --required-batch-size-multiple 1 --init-token 0 --arch bart_large \
#         --criterion sentence_prediction --num-classes $NUM_CLASSES --dropout 0.1 \
#         --attention-dropout 0.1 --weight-decay 0.01 --optimizer sgd --clip-norm 0.0 \
#         --lr-scheduler polynomial_decay --lr 1e-05 --total-num-update $TOTAL_NUM_UPDATES \
#         --warmup-updates $WARMUP_UPDATES --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 \
#         --fp16-scale-window 128 --max-epoch 10 --find-unused-parameters \
#         --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric
# CUDA_VISIBLE_DEVICES=0 fairseq-train $PATH1$DATASET_NAME$PATH2 --update-freq 8 --no-epoch-checkpoints \
#         --restore-file $BART_PATH --wandb-project Fine_Tune_BBBBP --batch-size $MAX_SENTENCES \
#         --max-tokens 4400 --task sentence_prediction --add-prev-output-tokens --layernorm-embedding \
#         --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader \
#         --reset-meters --required-batch-size-multiple 1 --init-token 0 --arch bart_large \
#         --criterion sentence_prediction --num-classes $NUM_CLASSES --dropout 0.1 \
#         --attention-dropout 0.1 --weight-decay 0.01 --optimizer sgd --clip-norm 0.0 \
#         --lr-scheduler polynomial_decay --lr 5e-05 --total-num-update $TOTAL_NUM_UPDATES \
#         --warmup-updates $WARMUP_UPDATES --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 \
#         --fp16-scale-window 128 --max-epoch 10 --find-unused-parameters \
#         --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric
# CUDA_VISIBLE_DEVICES=0 fairseq-train $PATH1$DATASET_NAME$PATH2 --update-freq 8 --no-epoch-checkpoints \
#         --restore-file $BART_PATH --wandb-project Fine_Tune_BBBP --batch-size $MAX_SENTENCES \
#         --max-tokens 4400 --task sentence_prediction --add-prev-output-tokens --layernorm-embedding \
#         --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader \
#         --reset-meters --required-batch-size-multiple 1 --init-token 0 --arch bart_large \
#         --criterion sentence_prediction --num-classes $NUM_CLASSES --dropout 0.1 \
#         --attention-dropout 0.1 --weight-decay 0.01 --optimizer sgd --clip-norm 0.0 \
#         --lr-scheduler polynomial_decay --lr 1e-04 --total-num-update $TOTAL_NUM_UPDATES \
#         --warmup-updates $WARMUP_UPDATES --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 \
#         --fp16-scale-window 128 --max-epoch 10 --find-unused-parameters \
#         --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric 