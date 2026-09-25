import math
from collections import defaultdict
import numpy as np
import pandas as pd
from .preprocessing import clean_tokens, clean_compact_name, extract_digits, extract_addr_tokens

class CandidateGenerator:
    """Scalable Multi-Key Blocking and Candidate Indexer."""
    
    def __init__(self, max_cand_per_key=100, default_top_k=15):
        self.max_cand_per_key = max_cand_per_key
        self.default_top_k = default_top_k
        self.name_tok_index = defaultdict(list)
        self.compact_index = defaultdict(list)
        self.addr_key_index = defaultdict(list)
        self.t_ids = None
        self.t_names = None
        self.t_addrs = None
        self.N_targets = 0

    def fit(self, targets_df):
        """Build high-speed inverted indexes over target entity records."""
        self.t_ids = targets_df['entity_id'].values
        self.t_names = targets_df['business_name'].values
        self.t_addrs = targets_df['business_address'].values
        self.N_targets = len(targets_df)
        
        self.name_tok_index = defaultdict(list)
        self.compact_index = defaultdict(list)
        self.addr_key_index = defaultdict(list)
        
        for i in range(self.N_targets):
            name = self.t_names[i]
            addr = self.t_addrs[i]
            
            # 1. Unique name tokens
            toks = set(clean_tokens(name))
            for tok in toks:
                self.name_tok_index[tok].append(i)
                
            # 2. Compact name prefix (len >= 5)
            cname = clean_compact_name(name)
            if len(cname) >= 5:
                self.compact_index[cname[:8]].append(i)
                
            # 3. Numeric address keys: (digit, addr_word)
            digs = extract_digits(addr)
            atoks = extract_addr_tokens(addr)
            if digs and atoks:
                for d in digs[:2]:
                    for w in atoks[:2]:
                        self.addr_key_index[(d, w)].append(i)

    def query(self, s1_name, s1_addr, top_k=None):
        """Retrieve top candidate target indices for a Source 1 entity."""
        if top_k is None:
            top_k = self.default_top_k
            
        cand_scores = defaultdict(float)
        
        # 1. Compact prefix
        cname = clean_compact_name(s1_name)
        if len(cname) >= 5:
            postings = self.compact_index.get(cname[:8], [])
            if len(postings) <= self.max_cand_per_key:
                for idx in postings:
                    cand_scores[idx] += 15.0
                    
        # 2. Rare name tokens
        toks = clean_tokens(s1_name)
        toks.sort(key=lambda t: len(self.name_tok_index.get(t, [])))
        for t in toks:
            postings = self.name_tok_index.get(t, [])
            if 0 < len(postings) <= self.max_cand_per_key:
                w = 10.0 / (1.0 + math.log1p(len(postings)))
                for idx in postings:
                    cand_scores[idx] += w
                if len(cand_scores) >= 40:
                    break
                    
        # 3. Addr key: (digit, addr_word)
        digs = extract_digits(s1_addr)
        atoks = extract_addr_tokens(s1_addr)
        if digs and atoks:
            for d in digs[:2]:
                for w in atoks[:2]:
                    postings = self.addr_key_index.get((d, w), [])
                    if 0 < len(postings) <= self.max_cand_per_key:
                        for idx in postings:
                            cand_scores[idx] += 8.0
                            
        if not cand_scores:
            return []
            
        if top_k is not None and len(cand_scores) > top_k:
            return sorted(cand_scores, key=cand_scores.get, reverse=True)[:top_k]
        else:
            return list(cand_scores.keys())
