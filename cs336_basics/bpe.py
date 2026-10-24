#import numpy
#import torch
import regex as re
from multiprocessing import Pool
from collections import Counter,defaultdict
import heapq
import os
import time

from cs336_basics.pretokenization_example import find_chunk_boundaries

# regex from gpt2: https://github.com/openai/tiktoken/pull/234/files
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

_worker_pattern = None # global val for compilation cache reuse
_worker_special_set = None
_worker_split_pattern = None

# init each process worker to save cache
def init_worker(special_tokens:list[str]):
    global _worker_pattern,_worker_special_set,_worker_split_pattern

    sorted_special_tokens = sorted(special_tokens,key=len,reverse=True)
    if sorted_special_tokens:
        split_rex = "|".join(re.escape(tok) for tok in sorted_special_tokens)
        _worker_split_pattern = re.compile(split_rex)
    #pattern = rf"(?:{special_pattern})|{PAT}" if special_tokens else PAT
    else:
        _worker_split_pattern = None

    _worker_pattern = re.compile(PAT)
    _worker_special_set = set(special_tokens)

# pretokenize and count on each text chunk
def process_chunk(args):
    file_path,start,end = args
    with open(file_path,"rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    if _worker_split_pattern is not None:
        texts = _worker_split_pattern.split(chunk)
    else:
        texts = [chunk]
    
    counter = Counter()

    return Counter(
        tuple(m.group(0).encode('utf-8'))
        for text in texts if text          # 1. 过滤掉空字符串
        for m in _worker_pattern.finditer(text)  # 2. 对每个非空段跑 PAT
    )

# 也有可能是超大的key的原因,是的！
def make_bytes_key(L):
    def key(b:bytes):
        k = 0
        assert len(b) <= L
        for i in range(L):
            d = (b[i] + 1) if i < len(b) else 0
            k = k * 257 + d
        return k
    return key


def merge_all(pretokenization_dict:dict[tuple[int,...],int],vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],vocab_size):
    # 考虑使用倒排索引，维护每个pair对应的pretoken单词
    # every pretoken is represented by a tuple of integers
    assert all(isinstance(w, tuple) for w, f in pretokenization_dict.items())
    pair_to_pretokens:dict[tuple[int,int],set[tuple[int,...]]] = defaultdict(set)
    #pretoken_to_pairs:dict[tuple[int,...],set[tuple[int,int]]] = {}
    pair_count:Counter = Counter()

    print("start merge initialization")
    start_time1 = time.time()
    # initialize dictionaries
    for token,num in pretokenization_dict.items():
        if len(token) <= 1:
            continue
        pc = Counter(zip(token,token[1:])) # use zip to generalize all adjcent pairs
        #pretoken_to_pairs[token] = pairs
        for pair,appearance in pc.items():
            pair_to_pretokens[pair].add(token)
            pair_count[pair] += appearance * num

    # initialize a heap,sorted by the cnt and key_in_heap
    L = 256
    bytes_key = make_bytes_key(L)
    key_in_heap ={int_id:bytes_key(b) for int_id,b in vocab.items() if int_id < 256}
    heap = [(-cnt,-key_in_heap[pair_int[0]],-key_in_heap[pair_int[1]],pair_int) for pair_int,cnt in pair_count.items()]
    heapq.heapify(heap)
    #initial_len = len(heap)
    end_time1 = time.time()
    print(end_time1 - start_time1)
    print("merge initialization finished")
    

    def push_pair(pair_int):
        new_cnt = pair_count.get(pair_int,0)
        if new_cnt <= 0:
            pair_count.pop(pair_int,None)
            return
        left,right = pair_int
        heapq.heappush(
            heap,
            (-new_cnt,-key_in_heap[left],-key_in_heap[right],pair_int),
        )

    # 惰性删除闹麻了，或许不再需要堆？
    def pop_max():
        pop_iters = 0  # 新增：记录本次清理了多少个过期元素
        while heap:
            pop_iters += 1
            neg_cnt,_,_,pair_int = heapq.heappop(heap)
            cnt = - neg_cnt
            if pair_count.get(pair_int,0) == cnt and cnt > 0:
                return (cnt,pair_int), pop_iters
        return None, pop_iters

    def rebuild_heap():
        nonlocal heap
        heap = [(-cnt,-key_in_heap[pair_int[0]],-key_in_heap[pair_int[1]],pair_int) for pair_int,cnt in pair_count.items()]
        heapq.heapify(heap)
        
    total_t_pop = 0.0       # 1. 取最大值、生成新 token 时间
    total_t_remove = 0.0    # 2. 扣除旧 pair 贡献时间
    total_t_add = 0.0       # 3. 替换并添加新 pair 贡献时间
    total_t_update = 0.0    # 4. 更新堆及重建时间
    merges_count = 0        # 辅助计数器，用于每 100 次打印
    
    # ===== 新增的统计累加器 =====
    total_t_pop_only = 0.0      # 纯 pop_max 耗时
    total_t_bytes_key = 0.0     # bytes_key 耗时
    total_pop_iters = 0         # 堆循环清理次数
    # ==========================

    # keep merging till len(vocab)==vocab_size
    while len(vocab) < vocab_size:
        
        # ---------------- 模块 1：出堆与基础更新 ----------------
        t0 = time.perf_counter()
        result, current_pop_iters = pop_max()  # 修改：接收迭代次数
        t_pop_only = time.perf_counter() - t0  # 修改：单独记录 pop_max 耗时
        
        if result is None:
            break
        count, pair = result
        A, B = pair

        new_token_id = len(vocab)
        vocab[new_token_id] = vocab[A] + vocab[B]
        merges.append((vocab[A], vocab[B]))
        
        # 修改：单独记录 bytes_key 耗时
        t_bk_0 = time.perf_counter()
        key_in_heap[new_token_id] = bytes_key(vocab[new_token_id])
        t_bytes_key = time.perf_counter() - t_bk_0
        
        t_pop = t_pop_only + t_bytes_key

        # ---------------- 模块 2：扣除旧贡献 ----------------
        t1 = time.perf_counter()
        affected = list(pair_to_pretokens.pop(pair, set()))
        snap_shots = []
        pairs_delta = Counter()
        
        for old_word in affected:
            if old_word not in pretokenization_dict:
                assert False
            freq = pretokenization_dict.pop(old_word)
            # 毒性代码预警：这里创建了 Counter
            pc_old = Counter(zip(old_word, old_word[1:]))
            for p, c in pc_old.items():
                pairs_delta[p] -= freq * c
                if p in pair_to_pretokens:
                    pair_to_pretokens[p].discard(old_word)
                    if not pair_to_pretokens[p]:
                        del pair_to_pretokens[p]
            snap_shots.append((old_word, freq))
        t_remove = time.perf_counter() - t1

        # ---------------- 模块 3：替换并添加新贡献 ----------------
        t2 = time.perf_counter()
        for old_word, freq in snap_shots:
            new_word = replace_non_overlapping(old_word, A, B, new_token_id)
            
            if len(new_word) <= 1:
                pretokenization_dict[new_word] = pretokenization_dict.get(new_word, 0) + freq
                continue
            
            pretokenization_dict[new_word] = freq
            # 毒性代码预警：这里又创建了 Counter
            pc_new = Counter(zip(new_word, new_word[1:]))
            for p, c in pc_new.items():
                pairs_delta[p] += freq * c
                pair_to_pretokens[p].add(new_word)
        t_add = time.perf_counter() - t2

        # ---------------- 模块 4：堆更新与重建 ----------------
        t3 = time.perf_counter()
        for p, c in pairs_delta.items():
            if c == 0:
                continue
            else:
                pair_count[p] += c
                push_pair(p)
        
        # 定期重建堆
        if len(heap) > 2 * len(pair_count):
            rebuild_heap()      
        t_update = time.perf_counter() - t3

        # ================= 统计与打印 =================
        total_t_pop += t_pop
        total_t_remove += t_remove
        total_t_add += t_add
        total_t_update += t_update
        
        # 累加新增的统计
        total_t_pop_only += t_pop_only
        total_t_bytes_key += t_bytes_key
        total_pop_iters += current_pop_iters
        
        merges_count += 1

        # 每 100 次打印一下各个模块的平均耗时
        if merges_count % 100 == 0:
            avg_pop = total_t_pop / 100
            avg_remove = total_t_remove / 100
            avg_add = total_t_add / 100
            avg_update = total_t_update / 100
            
            # 计算新增的平均值
            avg_pop_only = total_t_pop_only / 100
            avg_bytes_key = total_t_bytes_key / 100
            avg_pop_iters = total_pop_iters / 100
            
            print(f"--- 已完成 {len(vocab) - 256} 次合并 ---")
            print(f"  [1] 弹出与新建: {avg_pop:.3f}s (其中 pop_max: {avg_pop_only:.3f}s, bytes_key: {avg_bytes_key:.3f}s)")
            print(f"      -> 平均每轮 pop_max 迭代次数: {avg_pop_iters:.1f} 次") # 新增打印
            print(f"  [2] 扣除旧 Pair: {avg_remove:.3f}s")
            print(f"  [3] 替换与新加: {avg_add:.3f}s")
            print(f"  [4] 堆更新重建: {avg_update:.3f}s")
            print(f"  -> 平均每步总耗时: {avg_pop+avg_remove+avg_add+avg_update:.3f}s\n")
            
            # 重置这 100 步的累加器
            total_t_pop = 0.0
            total_t_remove = 0.0
            total_t_add = 0.0
            total_t_update = 0.0
            # 重置新增累加器
            total_t_pop_only = 0.0
            total_t_bytes_key = 0.0
            total_pop_iters = 0

def replace_non_overlapping(old_word,A,B,new_token_id):
    new_word = []
    n = len(old_word)
    i = 0
    while i < n:
        if i + 1 < n and old_word[i] == A and old_word[i+1] == B:
            new_word.append(new_token_id)
            i += 2
        else:
            new_word.append(old_word[i])
            i += 1
    return tuple(new_word)
    
def train_bpe(input_path:str,vocab_size:int,special_tokens:list[str]):
    """
    Read the training data from the given text file directed by the input_path,
    use special_tokens to divide chunks, then simply add them to the vocabulary and
    delete them from the texts.

    After that, use the gpt2 regex to pro-tokenize training data, then run bpe algorithms
    on tokens until the vocab_size reaches its limit.

    This method returns two output:
    - vocab: dict[int,bytes]
    - merges: list[tuple[bytes,bytes]] a list of bpe merges produced from training, it is 
    ordered by order of creation.
    """
    # step1: use the predefined function to find chunk boundaries
    # only use 1 special token as the boundary token, but need to
    # add all spetial_tokens into the vocabulary and then isolate them
    # from the text chunk
    print("start pretokenization")
    start_time = time.time()
    if special_tokens:
        # 优先找作业约定的分隔符
        doc_sep = "<|endoftext|>"
        split_special_token = doc_sep.encode('utf-8') if doc_sep in special_tokens else special_tokens[0].encode('utf-8')
    else:
        # 没有 special token，退化为按换行符切分
        split_special_token = b"\n"

    vocab = {i:bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode('utf-8')
    merges = []
    

    with open(input_path,"rb") as f:
        num_processes = 16
        boundaries = find_chunk_boundaries(f,num_processes,split_special_token)

        # use multiprocessing to paralize the task on multi-core cpu
        chunks = [(input_path,start,end) for start,end in zip(boundaries[:-1],boundaries[1:])]

        # Use multiprocessing pool to paralize
        with Pool(processes=num_processes,
                    initializer=init_worker,
                    initargs=(special_tokens,)) as pool:
            # merge need all chunks to be done,so here use sync map rather than async
            chunk_counters = pool.map(process_chunk,chunks)
        # reduce them to one counter
        total_counts = Counter()
        for c in chunk_counters:
            total_counts.update(c)
        end_time = time.time()
        print(end_time - start_time)
        print("finished pretokenization")
        
        # pretokenization finished, now that we have a big counter dictionary
        # the key is bytes(int) token and the value is its counts
        
        merge_all(total_counts,vocab,merges,vocab_size)

    return vocab,merges

