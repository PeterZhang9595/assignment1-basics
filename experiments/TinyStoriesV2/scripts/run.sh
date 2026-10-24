#!/bin/bash

# ===== 参数配置区 =====
INPUT_PATH="/share/project/fjc/zyy/cs336/assignment1-basics/data/owt_train.txt"
VOCAB_SIZE=32000
OUTPUT_DIR="/share/project/fjc/zyy/cs336/assignment1-basics/experiments/owt/output"
RUN_NAME="owt_train_32000"

# ===== 执行命令 =====
echo "🚀 开始直接运行 BPE，不使用 py-spy..."

# 给整个脚本加上 time 命令，方便最后看总耗时
time uv run python /share/project/fjc/zyy/cs336/assignment1-basics/experiments/TinyStoriesV2/scripts/run_bpe.py \
    --input_path "$INPUT_PATH" \
    --vocab_size $VOCAB_SIZE \
    --use_special_token \
    --output_dir "$OUTPUT_DIR" \
    --name "$RUN_NAME"

echo "✅ 运行结束！"