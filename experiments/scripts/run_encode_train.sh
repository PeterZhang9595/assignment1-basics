#!/bin/bash
# 任务 d：把训练集编码成 uint16 npy —— 先 TinyStories，再 OWT
#   用 Tokenizer.from_files 读已有的 vocab/merges pickle，encode_iterable 流式处理
#   边编码边落盘（常驻内存恒定），每 5 秒打印一次进度/速度/ETA/RSS
# 用法（建议放 tmux 里跑）：
#   bash experiments/scripts/run_encode_train.sh
# 资源：单进程 + nice，约占 1 个核 / <1GB RAM；可随时 Ctrl-C，
#       已产出的 .tmp 会被中断留下，重跑会覆盖。
set -e
cd /share/project/fjc/zyy/cs336/assignment1-basics

echo "==================== 1/2 TinyStories train (2.1 GB, TS-10k) ===================="
nice -n 10 uv run python experiments/scripts/encode_to_npy.py \
    --data data/TinyStoriesV2-GPT4-train.txt \
    --vocab experiments/TinyStoriesV2/output/tinystories_train_10000_vocab.pkl \
    --merges experiments/TinyStoriesV2/output/tinystories_train_10000_merges.pkl \
    --out experiments/TinyStoriesV2/output/tinystories_train_10000_ids.npy

echo "==================== 2/2 OWT train (12 GB, OWT-32k) ===================="
nice -n 10 uv run python experiments/scripts/encode_to_npy.py \
    --data data/owt_train.txt \
    --vocab experiments/owt/output/owt_train_32000_vocab.pkl \
    --merges experiments/owt/output/owt_train_32000_merges.pkl \
    --out experiments/owt/output/owt_train_32000_ids.npy

echo "✅ 任务d 完成"
ls -lh experiments/TinyStoriesV2/output/*_ids.npy experiments/owt/output/*_ids.npy
