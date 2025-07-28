export MASTER_ADDR=localhost
export MASTER_PORT=12345
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=2,3,4,5
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




#supported translation languages are Chinese (zh), German (de), and Japanese (ja).
# Check if command-line arguments are provided
# if [ $# -eq 0 ]; then
#   echo "Usage: $0 <source_language>"
#   exit 1
# fi
source=co4en


ckpt_path="/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/output/Semst_covost4en_asr_adp_1e-4lr_32bsz_2gpu_icu/asr_epoch_3_step_12654/global_step21250/mp_rank_00_model_states.pt"

# if [ ! -f "$ckpt_path" ]; then
#     echo "Download ckpt..."
#     git clone https://huggingface.co/yxdu/cotst
# fi

echo $ckpt_path


decode_log=${code_dir}/SLAM-LLM/examples/st_covost2/${source}_qwen25_ori.jsonl

echo "Decode log saved to: ${decode_log}"

encoder_path_hf=/work/2024/lixuanchen/models/whisper-large-v3
llm_path=/work/2024/lixuanchen/models/Qwen2.5-7B  # TODO
# val_data_path=/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/manifest2/fleurs_jpesen_test.jsonl
val_data_path=/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/manifest/en_asr_test.jsonl
# fleurs_jpen_test_wox.jsonl
# fleurs_jpen_dev+test_wox.jsonl
# fleurs_jpen_test_wox.jsonl fleurs_jpen_st_test.jsonl
# ++model_config.encoder_path=$speech_encoder_path \
# ++dataset_config.dataset=hf_dataset \
# ++dataset_config.file=examples/st_covost2/dataset/hf_dataset.py:get_speech_dataset \


torchrun \
    --nnodes 1 \
    --nproc_per_node ${gpu_count} \
    --master_port=29412 \
    ${code_dir}/SLAM-LLM/examples/st_covost2/inference_asr_batch.py \
    --config-path "conf" \
    --config-name "prompt.yaml" \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++fsdp_config.pure_bf16=true \
    ++model_config.llm_name="Qwen2-7B" \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=3584 \
    ++model_config.query_len=80 \
    ++model_config.encoder_name=whisper \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.encoder_path_hf=$encoder_path_hf \
    ++model_config.encoder_dim=1280 \
    ++model_config.encoder_projector=q-former \
    ++dataset_config.dataset=cl_st_dataset \
    ++dataset_config.file=examples/st_covost2/dataset/cl_st_dataset.py:get_speech_dataset \
    ++dataset_config.val_data_path=$val_data_path \
    ++dataset_config.input_type=mel \
    ++dataset_config.fix_length_audio=80 \
    ++dataset_config.mel_size=128 \
    ++dataset_config.inference_mode=true \
    ++dataset_config.source=$source \
    ++train_config.model_name=asr \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.batching_strategy=custom \
    ++train_config.num_epochs=1 \
    ++train_config.val_batch_size=16 \
    ++train_config.num_workers_dataloader=8 \
    ++log_config.decode_log=$decode_log \
    ++ckpt_path=$ckpt_path \
    $hydra_args
fi

# python ${code_dir}/SLAM-LLM/examples/st_covost2/test_werbleu.py --file $decode_log 
# python /work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/utils/test_werbleu_fixprompt.py --file $decode_log