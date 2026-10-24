import argparse
import pickle
import os
import time
import threading
import psutil
from cs336_basics.bpe import train_bpe

class MemoryMonitor(threading.Thread):
    """后台守护线程：每隔 0.5 秒记录一次当前进程及所有子进程的物理内存(RSS)总和"""
    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_memory = 0
        self._stop_event = threading.Event()
        self.process = psutil.Process(os.getpid())

    def run(self):
        while not self._stop_event.is_set():
            try:
                procs = [self.process] + self.process.children(recursive=True)
                current_mem = sum(p.memory_info().rss for p in procs) / (1024 * 1024)
                if current_mem > self.peak_memory:
                    self.peak_memory = current_mem
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop_event.set()

def parse_args():
    parser = argparse.ArgumentParser(description="BPE Training Script")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Target vocabulary size")
    # 修复布尔参数写法：只要命令行出现 --use_special_token，就为 True
    parser.add_argument("--use_special_token", action="store_true", help="Use special token <|endoftext|>")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save vocab and merges")
    parser.add_argument("--name", type=str, default="default", help="Prefix for output files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    special_tokens = ["<|endoftext|>"] if args.use_special_token else []
    
    print(f"\n{'='*40}")
    print(f"🚀 开始 BPE 训练")
    print(f"📂 数据源: {args.input_path}")
    print(f"🎯 目标词表大小: {args.vocab_size}")
    print(f"🔤 Special Token: {special_tokens}")
    print(f"{'='*40}\n")
    
    # === 启动内存监控 ===
    monitor = MemoryMonitor(interval=0.5)
    monitor.start()
    
    start_time = time.time()
    
    vocab, merges = train_bpe(
        input_path=args.input_path, 
        vocab_size=args.vocab_size,
        special_tokens=special_tokens
    )
    
    end_time = time.time()
    
    # === 停止监控 ===
    monitor.stop()
    monitor.join()
    
    print(f"\n{'='*40}")
    print(f"✅ 训练完成！")
    print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒 ({(end_time - start_time)/60:.2f} 分钟)")
    print(f"💾 内存峰值 (RSS): {monitor.peak_memory:.2f} MB")
    print(f"📚 最终词表大小: {len(vocab)}，合并次数: {len(merges)}")
    print(f"{'='*40}\n")
    
    # ================= 保存结果 =================
    vocab_path = os.path.join(args.output_dir, f"{args.name}_vocab.pkl")
    merges_path = os.path.join(args.output_dir, f"{args.name}_merges.pkl")
    
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
        
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
        
    print(f"💾 词表已保存至: {vocab_path}")
    print(f"💾 合并规则已保存至: {merges_path}")