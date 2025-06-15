export TOKENIZERS_PARALLELISM=false
# export WANDB_MODE=offline
gpu="0,1"
export CUDA_VISIBLE_DEVICES=$gpu

if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

echo "GPU number: $gpu_count"

current_script=$(readlink -f "$0")
current_dir=$(dirname "$current_script")
code_dir=$(realpath "$current_dir/../../../../")
cd ${code_dir}/semst


source=covost4en
wandb_exp_name=Semst_${source}_asr_adp_1e4lr_32bsz_${gpu_count}gpu_5000warmupWarmupCosine
output_dir=/Work21/2024/lixuanchen/project/semst/examples/st_covost2/outputs/${wandb_exp_name}
ds_config=/Work21/2024/lixuanchen/project/semst/examples/st_covost2/conf/ds_config.json


encoder_path_hf=/Work21/2024/lixuanchen/models/Whisper-large-v3
llm_path=/Work21/2024/lixuanchen/models/Qwen2-7B  # TODO

#change your train data
train_data_path=/Work21/2024/lixuanchen/project/ic26/data/covost2/covost_en_train_clean_asr.jsonl  # TODO
val_data_path=/Work21/2024/lixuanchen/project/ic26/data/covost2/covost_en_dev_asr.jsonl



# Find End checkpoint
# max_epoch=$(ls -d ${checkpoint_dir}/asr_epoch_*_step_* | sed -n 's/.*asr_epoch_\([0-9]*\)_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)
# max_step=$(ls -d ${checkpoint_dir}/asr_epoch_${max_epoch}_step_* | sed -n 's/.*asr_epoch_[0-9]*_step_\([0-9]*\).*/\1/p' | sort -n | tail -1)


# final_path="${checkpoint_dir}/asr_epoch_${max_epoch}_step_${max_step}"


# ckpt_name=$final_path/model.pt

# echo $ckpt_name




hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=Qwen \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=3584 \
++model_config.encoder_name=whisper \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_path_hf=$encoder_path_hf \
++model_config.encoder_dim=1280 \
++model_config.encoder_projector=q-former \
++model_config.query_len=80 \
++dataset_config.dataset=st_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=mel \
++dataset_config.mel_size=128  \
++dataset_config.fix_length_audio=80 \
++dataset_config.source=$source \
++train_config.model_name=asr \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.num_workers_dataloader=8 \
++train_config.output_dir=$output_dir \
++metric=acc \
"
train_args="
++train_config.num_epochs=3 \
++train_config.gradient_accumulation_steps=4 \
++train_config.validation_interval=5000 \
++train_config.warmup_steps=2000 \
++train_config.total_steps=21837 \
++train_config.lr=1e-4 \
++train_config.batch_size_training=4 \
++train_config.val_batch_size=16 \
"
log_args="
++log_config.use_wandb=true \
++log_config.wandb_dir=$output_dir \
++log_config.wandb_entity_name=sslnon \
++log_config.wandb_project_name=Semst_asr_adp \
++log_config.wandb_exp_name=${wandb_exp_name} \
++deepspeed_config=$ds_config \
"


deepspeed \
    --include localhost:${gpu} \
    --master_port=29502 \
    ${code_dir}/semst/examples/st_covost2/deepspeed_finetune_asr.py \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++fsdp_config.pure_bf16=true \
    ++train_config.use_peft=false \
    $hydra_args \
    $train_args \
    $log_args
fi