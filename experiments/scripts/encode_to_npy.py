#!/usr/bin/env python
"""把一份文本语料编码成 uint16 的 .npy（流式，常驻内存与语料大小无关）。

用途：
  任务 1（valid 集）和 任务 d（train 集）共用。
  用 Tokenizer.from_files 读 pickle 词表/merges，用 encode_iterable 逐行流式编码，
  先把 id 追加写入一个 raw uint16 临时文件（普通 buffered 写，不用 memmap，
  因为 memmap 的脏页会累计到 RSS 里），结束后一次性加上 .npy header 流式拷贝成标准 .npy。

峰值内存 ≈ FLUSH_EVERY 个 Python int（约 40MB）+ 一个 2MB 的 uint16 数组 + 拷贝时的 64MB 块，
与语料是 22MB 还是 12GB 无关。

用法：
  python experiments/scripts/encode_to_npy.py \
      --data data/owt_valid.txt \
      --vocab experiments/owt/output/owt_train_32000_vocab.pkl \
      --merges experiments/owt/output/owt_train_32000_merges.pkl \
      --out experiments/TinyStoriesV2/output/owt_valid_32000_ids.npy
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np

REPO = "/share/project/fjc/zyy/cs336/assignment1-basics"
sys.path.insert(0, REPO)

from cs336_basics.tokenizer import Tokenizer  # noqa: E402

FLUSH_EVERY = 1_000_000       # 每攒 100 万个 id 落一次盘（约 2MB）
COPY_BLOCK = 64 << 20         # 最终拷贝时每块 64MB


class Writer:
    """顺序追加 uint16 的写入器（背后是普通文件，内存恒定）。"""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "wb")
        self.n = 0
        self.buf: list[int] = []

    def add(self, tid: int) -> None:
        self.buf.append(tid)
        if len(self.buf) >= FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        self.f.write(np.asarray(self.buf, dtype=np.uint16).tobytes())
        self.n += len(self.buf)
        self.buf.clear()

    def close_to_npy(self, out_path: str) -> None:
        """把 raw 临时文件加上 .npy header 流式拷贝成标准 npy（内存恒定）。"""
        self.flush()
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()

        header = {"descr": "<u2", "fortran_order": False, "shape": (self.n,)}
        with open(out_path, "wb") as out, open(self.path, "rb") as src:
            np.lib.format.write_array_header_1_0(out, header)
            shutil.copyfileobj(src, out, COPY_BLOCK)
        os.remove(self.path)


class ByteCountingReader:
    """逐行吐文本，同时精确累计已读字节数（用于进度/ETA）。

    用 readline() 而不是直接用文件迭代器，因为 TextIOWrapper.tell() 在迭代路径上
    返回的位置不可靠（实测一直给 0），而多算一次 line.encode() 的开销可以忽略。
    """

    def __init__(self, path: str):
        self.f = open(path, "r", encoding="utf-8")
        self.nbytes = 0
        self.nlines = 0

    def __iter__(self):
        return self

    def __next__(self):
        line = self.f.readline()
        if not line:
            self.f.close()
            raise StopIteration
        self.nbytes += len(line.encode("utf-8"))
        self.nlines += 1
        return line


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except Exception:
        return 0.0


def encode_file(tok: Tokenizer, in_path: str, out_path: str, report_every: float = 5.0) -> dict:
    total = os.path.getsize(in_path)
    w = Writer(out_path + ".tmp")
    reader = ByteCountingReader(in_path)
    t0 = t_last = time.time()
    last_bytes = 0
    peak_rss = 0.0

    for tid in tok.encode_iterable(reader):
        w.add(tid)
        now = time.time()
        if now - t_last >= report_every:
            w.flush()
            bytes_done = max(reader.nbytes, 1)
            elapsed = now - t0
            speed = bytes_done / elapsed / 2**20 if elapsed else 0.0
            tok_s = w.n / elapsed if elapsed else 0.0
            pct = 100.0 * bytes_done / total if total else 0.0
            rate = (bytes_done - last_bytes) / (now - t_last) if now > t_last else 0.0
            eta = (total - bytes_done) / rate if rate > 0 else float("inf")
            peak_rss = max(peak_rss, _rss_mb())
            print(
                f"  [{pct:5.1f}%] {bytes_done/2**20:8.0f}/{total/2**20:.0f} MB  "
                f"{w.n/1e6:9.1f}M tok  {speed:5.2f} MB/s  {tok_s/1e3:7.1f}K tok/s  "
                f"ETA {eta/60:6.1f} min  RSS {peak_rss:6.0f} MB",
                flush=True,
            )
            t_last = now
            last_bytes = bytes_done

    w.flush()
    elapsed = time.time() - t0
    n = w.n
    w.close_to_npy(out_path)

    ratio = total / n if n else 0.0
    stats = {
        "bytes": total,
        "tokens": n,
        "seconds": elapsed,
        "mb_per_s": total / elapsed / 2**20,
        "tok_per_s": n / elapsed,
        "bytes_per_token": ratio,
        "peak_rss_mb": max(peak_rss, _rss_mb()),
        "out": out_path,
    }
    print(
        f"  [100.0%] 完成: {n:,} tokens, {total/2**20:.1f} MB, "
        f"{elapsed:.1f}s, {stats['mb_per_s']:.2f} MB/s, {ratio:.3f} bytes/token, "
        f"峰值 RSS {stats['peak_rss_mb']:.0f} MB",
        flush=True,
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="输入文本文件")
    ap.add_argument("--vocab", required=True, help="vocab pickle")
    ap.add_argument("--merges", required=True, help="merges pickle")
    ap.add_argument("--out", required=True, help="输出 .npy")
    ap.add_argument("--special", default="<|endoftext|>")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    specials = [args.special] if args.special else None

    print(f"数据: {args.data}  ({os.path.getsize(args.data)/2**20:.1f} MB)", flush=True)
    print(f"词表: {args.vocab}", flush=True)
    tok = Tokenizer.from_files(args.vocab, args.merges, specials)
    print(f"  vocab {len(tok.vocab):,} / merges {len(tok.merges):,} / special {specials}", flush=True)

    encode_file(tok, args.data, args.out,report_every=120.0)


if __name__ == "__main__":
    main()
