import csv
import json
import os

# 设置输入输出路径
split="train"
lang="zh-CN"
tsv_path = f"/work/2024/lixuanchen/data/cv-corpus-20.0-2024-12-06/{lang}/validated.tsv"
output_jsonl_path = f"/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/manifest/big_{lang}_asr_{split}.jsonl"

with open(tsv_path, encoding="utf-8") as tsv_file, open(output_jsonl_path, "w", encoding="utf-8") as out_file:
    reader = csv.DictReader(tsv_file, delimiter="\t")
    
    for row in reader:
        audio_file = row["path"]
        sentence = row["sentence"]

        # 构造新的 JSON 字典
        json_obj = {
            "audio": f"/work/2024/lixuanchen/data/cv-corpus-20.0-2024-12-06/zh-CN/clips/{audio_file}",
            "prompt": f"<|en|><|asr|>",
            "gt": sentence,
            "source": "common_voice"
        }

        # 写入 JSONL 文件
        out_file.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
    print("DONE")