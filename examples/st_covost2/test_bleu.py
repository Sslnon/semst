import json
import sacrebleu
from collections import defaultdict
from typing import Dict, List


def extract_by_source(jsonl_path: str, lang_tag: str = "<|zh|>") -> Dict[str, Dict[str, List[str]]]:
    """
    按照 source 分组提取参考和预测文本。
    返回结构：{source: {"refs": [...], "hyps": [...]} }
    """
    grouped = defaultdict(lambda: {"refs": [], "hyps": []})

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            gt = data.get("gt", "").strip()
            response = data.get("response", "").strip()
            source = data.get("source", "unknown")
            # print(source)

            if lang_tag not in response:
                print(f"[WARN] Missing tag in response: {response}")
                continue

            parts = response.split(lang_tag)
            if len(parts) != 2:
                print(f"[WARN] Unexpected response format: {response}")
                continue

            hyp_zh = parts[1].strip()
            grouped[source]["refs"].append(gt)
            grouped[source]["hyps"].append(hyp_zh)

    return grouped


def compute_bleu_per_source(jsonl_path: str, lang_tag: str = "<|zh|>") -> Dict[str, Dict[str, float]]:
    """
    针对每个 source 分组计算 BLEU
    返回格式：{source: {"bleu": x, "count": y}}
    """
    grouped_data = extract_by_source(jsonl_path, lang_tag)
    bleu_results = {}

    for source, pairs in grouped_data.items():
        refs = pairs["refs"]
        hyps = pairs["hyps"]

        if not refs:
            continue

        bleu = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True, tokenize="zh")
        bleu_results[source] = {
            "bleu": round(bleu.score, 2),
            "count": len(refs)
        }

    return bleu_results


results = compute_bleu_per_source("/Work21/2024/lixuanchen/project/semst/examples/st_covost2/asr_fleurs_en_test_fix.jsonl")

for source, metrics in results.items():
    print(f"[{source}] BLEU: {metrics['bleu']} (count: {metrics['count']})")
