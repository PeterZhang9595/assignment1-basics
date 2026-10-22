import numpy
import torch
import regex as re
from multiprocessing import Pool
from collections import Counter,defaultdict
import heapq

from cs336_basics.pretokenization_example import find_chunk_boundaries

# regex from gpt2: https://github.com/openai/tiktoken/pull/234/files
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

_worker_pattern = None # global val for compilation cache reuse
_worker_special_set = None

# init each process worker to save cache
def init_worker(special_tokens:list[str]):
    global _worker_pattern,_worker_special_set

    sorted_special_tokens = sorted(special_tokens,key=len,reverse=True)
    special_pattern = "|".join(re.escape(tok) for tok in sorted_special_tokens)
    pattern = rf"(?:{special_pattern})|{PAT}" if special_tokens else PAT

    _worker_pattern = re.compile(pattern)
    _worker_special_set = set(special_tokens)

# pretokenize and count on each text chunk
def process_chunk(args):
    file_path,start,end = args
    with open(file_path,"rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    return Counter(tuple(m.group(0).encode('utf-8'))
                     for m in _worker_pattern.finditer(chunk)
                     if m.group(0) not in _worker_special_set
    )

def merge_all(pretokenization_dict:dict[tuple[int,...],int],vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],vocab_size):
    # 考虑使用倒排索引，维护每个pair对应的pretoken单词
    # every pretoken is represented by a tuple of integers
    assert all(isinstance(w, tuple) for w, f in pretokenization_dict.items())
    pair_to_pretokens:dict[tuple[int,int],set[tuple[int,...]]] = defaultdict(set)
    #pretoken_to_pairs:dict[tuple[int,...],set[tuple[int,int]]] = {}
    pair_count:Counter = Counter()

    # initialize all three dictionary
    for token,num in pretokenization_dict.items():
        if len(token) <= 1:
            continue
        pc = Counter(zip(token,token[1:])) # use zip to generalize all adjcent pairs
        pairs = set(pc)
        #pretoken_to_pairs[token] = pairs
        for pair,appearance in pc.items():
            pair_to_pretokens[pair].add(token)
            pair_count[pair] += appearance * num
    
    
    # keep merging till len(vocab)==vocab_size
    while len(vocab) < vocab_size:
        # get the most frequent pair from the heap data structure
        result = pop_max(heap,pair_count)
        if result is None:
            break
        pair,count = result
        A,B = pair

        # this pair is a new bytes token in vocab
        new_token_id = len(vocab)
        vocab[new_token_id] = vocab[A] + vocab[B]
        merges.append((vocab[A],vocab[B]))

        # 先维护这个pair对应的所有pretokens的快照list
        affected = list(pair_to_pretokens.pop(pair,set()))

        # 找出受影响的pretokens对应的pair,扣除旧贡献
        snap_shots = []
        for old_word in affected:
            if old_word not in pretokenization_dict:
                continue
            freq = pretokenization_dict.pop(old_word)
            pc_old = Counter(zip(old_word,old_word[1:]))
            for p,c in pc_old.items():
                pair_count[p] -= freq * c  
                if p in pair_to_pretokens:
                    pair_to_pretokens[p].discard(old_word)
                    if not pair_to_pretokens[p]:
                        del pair_to_pretokens[p]
            snap_shots.append((old_word,freq))

        # 把所有old_word替换成为new_word，
        # 记录到pretokenization_dict
        # 更新pair_to_pretokens
        for old_word,freq in snap_shots:
            new_word = replace_non_overlapping(old_word,A,B,new_token_id)
            
            if len(new_word) <= 1:
                pretokenization_dict[new_word] = pretokenization_dict.get(new_word,0) + freq
                continue
            
            pretokenization_dict[new_word] = freq
            pc_new = Counter(zip(new_word,new_word[1:]))
            for p,c in pc_new.items():
                pair_to_pretokens[p].add(new_word)
                pair_count[p] += freq * c







            
        pair_count.pop(pair,None)
        pair_to_pretokens.pop(pair,None)

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
        total_counts = Counter()
        for c in chunk_counters:
            total_counts.update(c)
        
        # pretokenization finished, now that we have a big counter dictionary
        # the key is bytes(int) token and the value is its counts
        
        merge_all(total_counts,vocab,merges,vocab_size)

    return vocab,merges

