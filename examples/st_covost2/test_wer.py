import json
from typing import List
from evaluate import load
from transformers.models.whisper.english_normalizer import BasicTextNormalizer


def compute_wer_from_file(jsonl_path: str, task: str = "asr") -> dict:

    response_asr = []
    gt_asr = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            gt = data['gt']
            response = data['response']

            if task == "st":
                text_lan = gt.split("|>")[0].split("<|")[-1]
                text_lan = "<|" + text_lan + "|>"

                gt_parts = gt.split(text_lan)
                response_parts = response.split(text_lan)

                if len(gt_parts) == 2 and len(response_parts) == 2:
                    gt_asr.append(gt_parts[0].strip())
                    response_asr.append(response_parts[0].strip())
                else:
                    # fallback to whole
                    gt_asr.append(gt.strip())
                    response_asr.append(response.strip())
            else:
                gt_asr.append(gt.strip())
                response_asr.append(response.strip())

    # 评估
    wer_metric = load("wer")
    normalizer = BasicTextNormalizer()

    wer_ortho = 100 * wer_metric.compute(predictions=response_asr, references=gt_asr)

    pred_str_norm = [normalizer(pred) for pred in response_asr]
    label_str_norm = [normalizer(label) for label in gt_asr]

    # 过滤空引用
    filtered_pred = [p for p, r in zip(pred_str_norm, label_str_norm) if r.strip()]
    filtered_ref = [r for r in label_str_norm if r.strip()]

    wer_norm = 100 * wer_metric.compute(predictions=filtered_pred, references=filtered_ref)

    return {
        "wer_ortho": round(wer_ortho, 2),
        "wer": round(wer_norm, 2),
        "count": len(filtered_pred)
    }


res = compute_wer_from_file("/Work21/2024/lixuanchen/project/semst/examples/st_covost2/asr_covost2_en_dev.jsonl")
print(res)  # {'wer_ortho': 18.23, 'wer': 15.77, 'count': 999}
