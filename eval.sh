export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=meta-llama/Llama-2-7b-hf
export DATASET_PATH=c4

python main.py $MODEL_PATH $DATASET_PATH\
    --nsamples=1024 \
    --val_size=128 \
    --num_codebooks=1 \
    --nbits_per_codebook=16 \
    --in_group_size=8 \
    --relative_mse_tolerance=0.01 \
    --finetune_batch_size=32 \
    --finetune_max_epochs=10 \
    --finetune_early_stop=3 \
    --finetune_keep_best \
    --local_batch_size=1 \
    --offload_activations \
    --dtype float32