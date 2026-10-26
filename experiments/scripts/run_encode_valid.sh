#!/bin/bash
# 任务 1：把两个 valid 集编码成 uint16 npy（各自用自己的 tokenizer）
#   产物统一放到 experiments/TinyStoriesV2/output/ 下（按 TODO 要求）
# 单进程串行 + nice，避免影响机器上其它训练任务
set -e
cd /share/project/fjc/zyy/cs336/assignment1-basics

OUT="experiments/TinyStoriesV2/output"
mkdir -p "$OUT"

echo "==================== TinyStories valid (tokenizer: TS-10k) ===================="
nice -n 10 uv run python experiments/scripts/encode_to_npy.py \
    --data data/TinyStoriesV2-GPT4-valid.txt \
    --vocab experiments/TinyStoriesV2/output/tinystories_train_10000_vocab.pkl \
    --merges experiments/TinyStoriesV2/output/tinystories_train_10000_merges.pkl \
    --out "$OUT/tinystories_valid_10000_ids.npy"

echo "==================== OWT valid (tokenizer: OWT-32k) ===================="
nice -n 10 uv run python experiments/scripts/encode_to_npy.py \
    --data data/owt_valid.txt \
    --vocab experiments/owt/output/owt_train_32000_vocab.pkl \
    --merges experiments/owt/output/owt_train_32000_merges.pkl \
    --out "$OUT/owt_valid_32000_ids.npy"

echo "✅ 任务1 完成：$OUT"
ls -lh "$OUT"/*.npy
