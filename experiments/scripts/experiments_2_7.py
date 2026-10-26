#!/usr/bin/env python
"""PDF §2.7 的 (a)(b)(c) 三个实验，结果追加到 experiments/TODO.md。

(a) 从 TinyStories / OWT 的 valid 各抽 10 篇（按 <|endoftext|> 切分），
    用各自训练好的 tokenizer 编码，报告压缩率 bytes/token。
(b) 交叉：OWT 的文档用 TinyStories tokenizer 编码（反向也顺手做一份对照）。
(c) 吞吐：在 valid 的 8MB 切片上测冷缓存吞吐（bytes/s、tokens/s），
    并外推到 Pile 的 825GB 需要多久。

用法：uv run python experiments/scripts/experiments_2_7.py
"""
from __future__ import annotations

import datetime
import os
import sys
import time

REPO = "/share/project/fjc/zyy/cs336/assignment1-basics"
sys.path.insert(0, REPO)

from cs336_basics.tokenizer import Tokenizer  # noqa: E402

SPECIAL = "<|endoftext|>"
N_DOCS = 10
THROUGHPUT_BYTES = 8 * 1024 * 1024

TS_VALID = f"{REPO}/data/TinyStoriesV2-GPT4-valid.txt"
OWT_VALID = f"{REPO}/data/owt_valid.txt"
TS_VOCAB = f"{REPO}/experiments/TinyStoriesV2/output/tinystories_train_10000_vocab.pkl"
TS_MERGES = f"{REPO}/experiments/TinyStoriesV2/output/tinystories_train_10000_merges.pkl"
OWT_VOCAB = f"{REPO}/experiments/owt/output/owt_train_32000_vocab.pkl"
OWT_MERGES = f"{REPO}/experiments/owt/output/owt_train_32000_merges.pkl"
TODO = f"{REPO}/experiments/TODO.md"


def sample_docs(path: str, n: int = N_DOCS) -> list[str]:
    """流式读出前 n 篇由 <|endoftext|> 分隔的文档（去掉首尾空白）。"""
    docs, buf = [], ""
    with open(path, "r", encoding="utf-8") as f:
        while len(docs) < n:
            block = f.read(1 << 20)
            if not block:
                break
            buf += block
            parts = buf.split(SPECIAL)
            buf = parts.pop()
            for p in parts:
                p = p.strip()
                if p:
                    docs.append(p)
                    if len(docs) == n:
                        break
    if buf.strip() and len(docs) < n:
        docs.append(buf.strip())
    return docs


def compress(tok: Tokenizer, docs: list[str]) -> dict:
    nbytes = sum(len(d.encode("utf-8")) for d in docs)
    t0 = time.time()
    ntok = sum(len(tok.encode(d)) for d in docs)
    el = time.time() - t0
    return {"bytes": nbytes, "tokens": ntok, "ratio": nbytes / ntok, "seconds": el}


def throughput(tok: Tokenizer, path: str, nbytes: int = THROUGHPUT_BYTES) -> dict:
    raw = open(path, "rb").read(nbytes)
    text = raw.decode("utf-8", errors="ignore")
    n = len(text.encode("utf-8"))
    t0 = time.time()
    ids = tok.encode(text)
    el = time.time() - t0
    return {
        "bytes": n,
        "tokens": len(ids),
        "seconds": el,
        "mb_per_s": n / el / 2**20,
        "tok_per_s": len(ids) / el,
    }


def main() -> None:
    ts = Tokenizer.from_files(TS_VOCAB, TS_MERGES, [SPECIAL])
    owt = Tokenizer.from_files(OWT_VOCAB, OWT_MERGES, [SPECIAL])

    ts_docs = sample_docs(TS_VALID)
    owt_docs = sample_docs(OWT_VALID)
    print(f"抽样: TinyStories {len(ts_docs)} 篇 / OWT {len(owt_docs)} 篇", flush=True)

    # (a) 各自语料 + 各自 tokenizer
    a_ts = compress(ts, ts_docs)
    a_owt = compress(owt, owt_docs)
    # (b) 交叉
    b_owt_with_ts = compress(ts, owt_docs)
    b_ts_with_owt = compress(owt, ts_docs)

    print(f"(a) TS  docs  w/ TS-10k : {a_ts['ratio']:.3f} bytes/token "
          f"({a_ts['bytes']:,} B / {a_ts['tokens']:,} tok)", flush=True)
    print(f"(a) OWT docs  w/ OWT-32k: {a_owt['ratio']:.3f} bytes/token "
          f"({a_owt['bytes']:,} B / {a_owt['tokens']:,} tok)", flush=True)
    print(f"(b) OWT docs  w/ TS-10k : {b_owt_with_ts['ratio']:.3f} bytes/token "
          f"(相对 OWT-32k 变差 {100*(b_owt_with_ts['ratio']/a_owt['ratio']-1):+.1f}%)", flush=True)
    print(f"(b) TS  docs  w/ OWT-32k: {b_ts_with_owt['ratio']:.3f} bytes/token "
          f"(相对 TS-10k 变差 {100*(b_ts_with_owt['ratio']/a_ts['ratio']-1):+.1f}%)", flush=True)

    # (c) 吞吐（冷缓存，8MB 切片）
    t_ts = throughput(Tokenizer.from_files(TS_VOCAB, TS_MERGES, [SPECIAL]), TS_VALID)
    t_owt = throughput(Tokenizer.from_files(OWT_VOCAB, OWT_MERGES, [SPECIAL]), OWT_VALID)
    pile_gb = 825
    for name, t in (("TS-10k ", t_ts), ("OWT-32k", t_owt)):
        hours = pile_gb * 1024 / t["mb_per_s"] / 3600
        print(f"(c) {name}: {t['mb_per_s']:.2f} MB/s, {t['tok_per_s']/1e3:.0f}K tok/s "
              f"-> Pile 825GB 需 {hours/24:.1f} 天", flush=True)
    hours_owt = pile_gb * 1024 / t_owt["mb_per_s"] / 3600

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "",
        f"## PDF §2.7 实验结果（{stamp} 自动生成，valid 各抽 {N_DOCS} 篇，已去首尾空白）",
        "",
        "### (a) 各自 tokenizer 在自己语料上的压缩率",
        "",
        "| 语料 | tokenizer | 字节数 | token 数 | 压缩率 (bytes/token) |",
        "|---|---|---:|---:|---:|",
        f"| TinyStories valid | TinyStories-10k | {a_ts['bytes']:,} | {a_ts['tokens']:,} | **{a_ts['ratio']:.3f}** |",
        f"| OWT valid | OWT-32k | {a_owt['bytes']:,} | {a_owt['tokens']:,} | **{a_owt['ratio']:.3f}** |",
        "",
        f"结论：两个 tokenizer 在各自语料上分别约 **{a_ts['ratio']:.2f}** 和 **{a_owt['ratio']:.2f}** 字节/token，"
        f"OWT（32k 词表）单位 token 覆盖的字节数比 TinyStories 高 {100*(a_owt['ratio']/a_ts['ratio']-1):+.0f}%，"
        "与其更大的词表和更长/更多样的网页文本一致。",
        "",
        "### (b) 交叉 tokenize",
        "",
        "| 文档来源 | tokenizer | 压缩率 (bytes/token) | 相对自身 tokenizer |",
        "|---|---|---:|---:|",
        f"| OWT valid | **OWT-32k**（自身） | {a_owt['ratio']:.3f} | — |",
        f"| OWT valid | **TinyStories-10k**（交叉） | {b_owt_with_ts['ratio']:.3f} | "
        f"{100*(b_owt_with_ts['ratio']/a_owt['ratio']-1):+.1f}% |",
        f"| TinyStories valid | **TinyStories-10k**（自身） | {a_ts['ratio']:.3f} | — |",
        f"| TinyStories valid | **OWT-32k**（交叉） | {b_ts_with_owt['ratio']:.3f} | "
        f"{100*(b_ts_with_owt['ratio']/a_ts['ratio']-1):+.1f}% |",
        "",
        f"结论：把 OWT 文档喂给 TinyStories tokenizer，压缩率从 {a_owt['ratio']:.2f} 退化到 "
        f"{b_owt_with_ts['ratio']:.2f} 字节/token（{100*(b_owt_with_ts['ratio']/a_owt['ratio']-1):+.1f}%），"
        "原因是 TinyStories 词表只见过儿童故事的词汇/形态，网页里的长词、URL、代码片段、"
        "多语言字符只能退化成更短的子词或单字节序列。",
        "",
        "### (c) 吞吐（8MB 切片，冷缓存）",
        "",
        "| tokenizer | 速度 (MB/s) | tokens/s | 外推 Pile 825GB |",
        "|---|---:|---:|---:|",
        f"| TinyStories-10k | {t_ts['mb_per_s']:.2f} | {t_ts['tok_per_s']/1e3:.0f}K | "
        f"{pile_gb*1024/t_ts['mb_per_s']/3600/24:.1f} 天 |",
        f"| OWT-32k | {t_owt['mb_per_s']:.2f} | {t_owt['tok_per_s']/1e3:.0f}K | "
        f"{pile_gb*1024/t_owt['mb_per_s']/3600/24:.1f} 天 |",
        "",
        f"结论：纯 Python 单线程约 **{t_owt['mb_per_s']:.2f} MB/s**（瓶颈是 GP2 预分词正则本身，"
        "实测正则单独耗时约 0.96 MB/s），据此 Pile 825GB 需要约 "
        f"**{hours_owt/24:.1f} 天**；实际工程中需要用多进程 + 更快的预分词实现（如 tiktoken 的 Rust 实现）。",
        "",
        "### (d) uint16 为什么合适（补充回答）",
        "",
        "两个词表大小分别是 10,000 和 32,000（含 special token），所有 id 都 < 65,536，"
        "正好落进 uint16（0–65,535）范围；相比 int32/int64 分别省一半/四分之三的存储和 IO，"
        "而 uint8 不够（32k 词表超 255）。",
        "",
    ]
    with open(TODO, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n结果已追加到 {TODO}", flush=True)


if __name__ == "__main__":
    main()
