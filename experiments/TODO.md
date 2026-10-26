现在主机上有一个较大的训练任务正在占用大量资源，所以运行过程中请注意时刻监控CPU和RAM使用情况，如果出现使用率接近百分百的情况及时终止我们现在这个tokenizer的任务进程，千万不要动其它的任务。

1. 首先使用/share/project/fjc/zyy/cs336/assignment1-basics/experiments/TinyStoriesV2/scripts/run.sh，通过修改路径，把TinyStoriesv2和owt的valid数据集/share/project/fjc/zyy/cs336/assignment1-basics/data/owt_valid.txt和/share/project/fjc/zyy/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt都编码一下存到/share/project/fjc/zyy/cs336/assignment1-basics/experiments/TinyStoriesV2/output下面。

2. 关注作业pdf 2.7 experiments，我需要你来跑一下四个任务。
分别使用在训练集上面训练好的tokenizer，进行任务a,b,c,d.
任务a：sample10个special token分割出的document然后进行编码计算compression ratio。
任务b:分别交叉使用tokenizer计算compression ratio
任务c：在a中估计throughput速度并且记录
任务a,b,c的结果主要是数字结论写到本文件里面，追加到结尾就可以了。

任务d: 在tinystories和owt上面分别跑一下tokenization，先跑tinystories，然后再跑owt。
要求使用我实现的from_files把已经有的merge和vocab pickle文件转化进入函数然后开始处理。请你写一个脚本，使用python handle打开训练txt文件，先创建Tokenizer，然后使用encode_iterable进行处理。最后的结果转化为uint16格式numpy存到/share/project/fjc/zyy/cs336/assignment1-basics/experiments对应数据集的output文件夹下面。
脚本里面要写一个检测速度的print,确保能够时刻追踪数据集的tokenization进度。

先完成任务1和2的a/b/c，然后把d的两个脚本写好，交付即可，我后续自己来开tmux运行。

## PDF §2.7 实验结果（2026-10-25 09:26 自动生成，valid 各抽 10 篇，已去首尾空白）

### (a) 各自 tokenizer 在自己语料上的压缩率

| 语料 | tokenizer | 字节数 | token 数 | 压缩率 (bytes/token) |
|---|---|---:|---:|---:|
| TinyStories valid | TinyStories-10k | 10,751 | 2,663 | **4.037** |
| OWT valid | OWT-32k | 50,460 | 11,201 | **4.505** |

结论：两个 tokenizer 在各自语料上分别约 **4.04** 和 **4.50** 字节/token，OWT（32k 词表）单位 token 覆盖的字节数比 TinyStories 高 +12%，与其更大的词表和更长/更多样的网页文本一致。

### (b) 交叉 tokenize

| 文档来源 | tokenizer | 压缩率 (bytes/token) | token 数 | 相对自身 tokenizer |
|---|---|---:|---:|---:|
| OWT valid | **OWT-32k**（自身） | 4.505 | 11,201 | — |
| OWT valid | **TinyStories-10k**（交叉） | 3.405 | 14,820 | token 数 +32.3% |
| TinyStories valid | **TinyStories-10k**（自身） | 4.037 | 2,663 | — |
| TinyStories valid | **OWT-32k**（交叉） | 3.895 | 2,760 | token 数 +3.6% |

（注意方向：bytes/token **变小** = 每个 token 覆盖的字节更少 = **token 数变多** = 压缩**更差**。）

结论：把 OWT 文档喂给 TinyStories tokenizer，字节/token 从 4.50 降到 3.40，同一批文档的 token 数从 11,201 涨到 14,820（**+32%**），即压缩显著变差，原因是 TinyStories 词表只见过儿童故事的词汇/形态，网页里的长词、URL、代码片段、多语言字符只能退化成更短的子词或单字节序列。反方向（TinyStories 文档用 OWT-32k）只差 3.6%，说明 32k 的大词表泛化能力更强，而 10k 小词表高度依赖自身领域。

### (c) 吞吐（8MB 切片，冷缓存）

| tokenizer | 速度 (MB/s) | tokens/s | 外推 Pile 825GB |
|---|---:|---:|---:|
| TinyStories-10k | 0.75 | 191K | 13.0 天 |
| OWT-32k | 0.48 | 116K | 20.3 天 |

结论：纯 Python 单线程约 **0.48 MB/s**（瓶颈是 GPT-2 预分词正则本身，实测正则单独耗时约 0.96 MB/s），据此 Pile 825GB 需要约 **20.3 天**；实际工程中需要用多进程 + 更快的预分词实现（如 tiktoken 的 Rust 实现）。

### (d) uint16 为什么合适（补充回答）

两个词表大小分别是 10,000 和 32,000（含 special token），所有 id 都 < 65,536，正好落进 uint16（0–65,535）范围；相比 int32/int64 分别省一半/四分之三的存储和 IO，而 uint8 不够（32k 词表超 255）。
