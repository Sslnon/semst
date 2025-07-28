export TOKENIZERS_PARALLELISM=false
# export WANDB_MODE=offline
gpu="6,7"
export CUDA_VISIBLE_DEVICES=$gpu
export PYTHONDONTWRITEBYTECODE=1

if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

echo "GPU number: $gpu_count"

current_script=$(readlink -f "$0")
current_dir=$(dirname "$current_script")
code_dir=$(realpath "$current_dir/../../../../")
cd ${code_dir}/SLAM-LLM

per_gpu_bsz=4
gas=8
bsz=$((gpu_count * per_gpu_bsz * gas))

lr=1e-5
warmup=200
num_epochs=20

source=fleurs_enjpeskoru
# wandb_exp_name=Semst_${source}_st_${lr}lr_${bsz}bsz_${gpu_count}gpu_${warmup}warmup_icu
wandb_exp_name=Semst_${source}_${bsz}bsz_${lr}lr_qwen25_ori_icu
# wandb_exp_name=test
output_dir=/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/output/${wandb_exp_name}
ds_config=/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/conf/ds_config.json


encoder_path_hf=/work/2024/lixuanchen/models/whisper-large-v3
llm_path=/work/2024/lixuanchen/models/Qwen2.5-7B  # TODO

#change your train data
train_data_path=/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/manifest2/fleurs_enjpeskoru_train.jsonl # TODO
val_data_path=/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/manifest2/fleurs_enjpeskoru_test.jsonl




line_count=$(wc -l < "$train_data_path")
total_steps=$((line_count * num_epochs))



ckpt_name=/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/output/Semst_covost2enzh_st_5e-6lr_16bsz_qwen25_icu/asr_epoch_2_step_18833/global_step27500/mp_rank_00_model_states.pt
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
++model_config.file=examples/st_covost2/model/slam_model_st.py \
++dataset_config.dataset=st_dataset \
++dataset_config.file=examples/st_covost2/dataset/st_dataset.py:get_speech_dataset \
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
++ckpt_path=$ckpt_name \
"
train_args="
++train_config.num_epochs=${num_epochs} \
++train_config.gradient_accumulation_steps=${gas} \
++train_config.validation_interval=1000 \
++train_config.warmup_steps=${warmup} \
++train_config.total_steps=${total_steps} \
++train_config.lr=${lr} \
++train_config.batch_size_training=${per_gpu_bsz} \
++train_config.val_batch_size=4 \
"
log_args="
++log_config.use_swanlab=true \
++log_config.swanlab_dir=$output_dir \
++log_config.swanlab_entity_name=16426917 \
++log_config.swanlab_project_name=Semst_allst_nlang \
++log_config.swanlab_exp_name=${wandb_exp_name} \
++deepspeed_config=$ds_config \
"


deepspeed \
    --include localhost:${gpu} \
    --master_port=29587 \
    ${code_dir}/SLAM-LLM/examples/st_covost2/deepspeed_finetune_asr.py \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++fsdp_config.pure_bf16=true \
    ++train_config.use_peft=false \
    $hydra_args \
    $train_args \
    $log_args
fi
