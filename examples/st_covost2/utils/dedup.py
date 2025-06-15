import json

# Step 1: 收集 dev 和 test 的 audio 字段
exclude_audio = set()
for fn in ['/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/manifest/zh-CN_asr_test.jsonl', '/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/manifest/zh-CN_asr_dev.jsonl']:
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            exclude_audio.add(item['audio'])   # 替换成你的主键字段名

# Step 2: 读 train 文件，排除在 exclude_audio 中的
input_train = '/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/manifest/big_zh-CN_asr_train.jsonl'
output_train = '/work/2024/lixuanchen/project/SLAM-LLM/examples/st_covost2/manifest/filtered_big_zh-CN_asr_train.jsonl'

with open(input_train, 'r', encoding='utf-8') as fin, open(output_train, 'w', encoding='utf-8') as fout:
    for line in fin:
        item = json.loads(line)
        if item['audio'] not in exclude_audio:   # 同上，如有不同主键请替换
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

print('过滤完成，已输出到', output_train)
