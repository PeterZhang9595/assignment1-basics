# /share/project/fjc/zyy/cs336/assignment1-basics/experiments/TinyStoriesV2/scripts/check_vocab_merges.py

import pickle
import itertools
import os

def print_pickle_preview(file_path: str, num_lines: int = 10):
    """
    读取 pickle 文件，并打印前 num_lines 行的内容。
    自动兼容 vocab (字典) 和 merges (列表) 两种格式。
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"\n{'='*50}")
    print(f"📄 文件路径: {file_path}")
    print(f"📊 总数据量: {len(data)} 条")
    print(f"🔍 预览前 {num_lines} 条:")
    print(f"{'-'*50}")

    if isinstance(data, dict):
        # 处理 vocab 字典
        for k, v in itertools.islice(data.items(), num_lines):
            print(f"ID: {k:<6} -> Token: {v}")
            
    elif isinstance(data, list):
        # 处理 merges 列表
        for i, item in enumerate(itertools.islice(data, num_lines)):
            print(f"[{i:>4}] -> {item}")
            
    else:
        print(f"⚠️ 未知的数据类型: {type(data)}")
    
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # 在这里填入你刚才跑出来的 pickle 路径
    VOCAB_PATH = "/share/project/fjc/zyy/cs336/assignment1-basics/experiments/TinyStoriesV2/output/tinystories_train_10000_vocab.pkl"
    MERGES_PATH = "/share/project/fjc/zyy/cs336/assignment1-basics/experiments/TinyStoriesV2/output/tinystories_train_10000_merges.pkl"
    
    # 预览前 15 条
    print_pickle_preview(VOCAB_PATH, 300)
    print_pickle_preview(MERGES_PATH, 15)