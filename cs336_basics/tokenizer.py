from __future__ import annotations
import json
from typing import Iterable,Iterator
import regex as re
from cs336_basics.bpe import PAT
import pickle


class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, 
        and (optionally) a list of special tokens. 
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        #assert len(vocab) == len(set(vocab.values()))
        #assert len(mergse) == len(vocab) - 256 - len(special_tokens)
        self.token_to_ids = {v:k for k,v in vocab.items()}
        #self.merge_id_rank = {(self.token_to_ids[merge[0]],self.token_to_ids[merge[1]]):i for i,merge in enumerate(merges)}
        self._worker_pattern = re.compile(PAT)
        
        self._worker_split_pattern = None
        sorted_special_tokens = sorted(self.special_tokens,key=len,reverse=True)
        if sorted_special_tokens:
            split_rex = "(" + "|".join(re.escape(tok) for tok in sorted_special_tokens) + ")" # 现在是捕获组
            self._worker_split_pattern = re.compile(split_rex)
        
        self.merge_id_rank = {}
        self.pair_to_merged = {}  # 新增：直接 (left_id, right_id) -> merged_id

        for i, merge in enumerate(merges):
            left_id = self.token_to_ids[merge[0]]
            right_id = self.token_to_ids[merge[1]]
            merged_id = self.token_to_ids[merge[0] + merge[1]]
            
            self.merge_id_rank[(left_id, right_id)] = i
            self.pair_to_merged[(left_id, right_id)] = merged_id
        
        self._cache = {}
        self.CACHE_MAX = 1 << 18   # ~26 万条上限（约 80MB）；满了就不再写入，避免大语料把内存撑爆

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                special_tokens: list[str] | None = None):
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the 
        same format that your BPE training code output) and (optionally) a list of special tokens. 
        """
        # all read from pickle,since my own implementation is save as pickle
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
            
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        # however,remember that test use json and txt file 

        return cls(vocab, merges, special_tokens)

    def process_pretoken_to_ids(self,pretoken):
        if not pretoken:
            return []

        if pretoken:
            # use merge and vocab process the pretoken into token id list
            # but it is time-consuming to iterate through all merges to see whether it is in the pretoken
            # so we construct a rank diction to back search the rank of merges.
            cache = self._cache.get(pretoken,-1)
            if cache != -1:
                return cache
            all_bytes = pretoken.encode('utf-8')
            # in practice,average pretokens are small,so it is ok to use list
            parts = [self.token_to_ids[bytes([b])] for b in all_bytes]

            n = len(parts)
            while(n > 1):
                best_rank = float('inf')
                best_idx = -1

                length = len(parts)
                for idx in range(length - 1):
                    rank = self.merge_id_rank.get((parts[idx],parts[idx+1]),-1)
                    if rank != -1:
                        if rank < best_rank:
                            best_rank = rank
                            best_idx = idx
                # only find one pair to merge
                if best_idx == -1:
                    if len(self._cache) < self.CACHE_MAX:
                        self._cache[pretoken] = parts
                    return parts
                
                parts[best_idx] = self.pair_to_merged[(parts[best_idx], parts[best_idx+1])]
                parts.pop(best_idx + 1)
                n -= 1
            if len(self._cache) < self.CACHE_MAX:
                self._cache[pretoken] = parts
            return parts



    def process_normal_text(self,text):
        
        ids = []
        for match in self._worker_pattern.finditer(text):
            pretoken = match.group(0)
            ids.extend(self.process_pretoken_to_ids(pretoken))
        return ids

    def encode(self,text:str)->list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        # here we simply use single thread process
        # first, pretokenization the text into a list of pretokens and convert them to bytes.
        # second, iterate over all pretokens,in each pretoken,use merge to generate pairs then look up in token_to_ids

        # use regex to divide 
        id_list = []
        last_index = 0

        if self._worker_split_pattern is not None:
            for part in self._worker_split_pattern.finditer(text):
                normal_text = text[last_index:part.start()]
                if normal_text:
                    id_list.extend(self.process_normal_text(normal_text))
                special_token = part.group(0).encode('utf-8')
                id_list.append(self.token_to_ids[special_token])
                last_index = part.end()
        
        normal_text = text[last_index:]
        if normal_text:
            id_list.extend(self.process_normal_text(normal_text))
        return id_list

    def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is 
        required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        # A chunk boundary is an artificial cut: the pretokenizer looks one character ahead
        # (the \s+(?!\S) rule), so the trailing whitespace run of a chunk is *undecided* until we
        # know what follows it. Keep the trailing [whitespace run][non-whitespace run] block and
        # prepend it to the next chunk, so that decision is made with full context.
        pending = ""
        for chunk in iterable:
            buf = pending + chunk
            cut = len(buf)
            while cut > 0 and not buf[cut - 1].isspace():   # trailing word: may still grow across chunks
                cut -= 1
            while cut > 0 and buf[cut - 1].isspace():       # the whitespace run right before it: identity undecided
                cut -= 1
            pending = buf[cut:]
            if cut:
                yield from self.encode(buf[:cut])
        # end of stream: the remaining tail is now genuinely followed by nothing
        if pending:
            yield from self.encode(pending)


    def decode(self,ids:list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        all_bytes =b"".join(self.vocab[i] for i in ids)
        return all_bytes.decode("utf-8",errors="replace")