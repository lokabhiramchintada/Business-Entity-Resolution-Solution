import pandas as pd
from unidecode import unidecode
import rapidfuzz.fuzz as fuzz
from .preprocessing import clean_tokens, extract_digits, extract_addr_tokens

FEATURE_NAMES = [
    'name_ratio', 'name_token_sort', 'name_token_set', 'name_partial',
    'name_jaccard', 'name_overlap_count',
    'addr_token_sort', 'addr_token_set', 'addr_partial',
    'addr_jaccard', 'addr_overlap_count',
    'digit_status', 'addr_missing', 'is_s2', 'len_diff'
]

def extract_pair_features(s1_name, s1_addr, tgt_name, tgt_addr, tgt_id):
    """Extract fine-grained similarity and structural features between a pair."""
    s1_n_str = str(s1_name) if pd.notna(s1_name) else ""
    t_n_str = str(tgt_name) if pd.notna(tgt_name) else ""
    s1_a_str = str(s1_addr) if pd.notna(s1_addr) else ""
    t_a_str = str(tgt_addr) if pd.notna(tgt_addr) else ""
    
    s1_n_clean = unidecode(s1_n_str).lower()
    t_n_clean = unidecode(t_n_str).lower()
    s1_a_clean = unidecode(s1_a_str).lower()
    t_a_clean = unidecode(t_a_str).lower()
    
    # 1. Name features
    name_ratio = fuzz.ratio(s1_n_clean, t_n_clean) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_n_clean, t_n_clean) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_n_clean, t_n_clean) / 100.0
    name_partial = fuzz.partial_ratio(s1_n_clean, t_n_clean) / 100.0
    
    s1_toks = set(clean_tokens(s1_n_str))
    t_toks = set(clean_tokens(t_n_str))
    tok_union = len(s1_toks | t_toks)
    name_jaccard = (len(s1_toks & t_toks) / tok_union) if tok_union > 0 else 0.0
    name_overlap_count = float(len(s1_toks & t_toks))
    
    # 2. Address features
    addr_missing = 1.0 if not t_a_clean or t_a_clean == 'nan' or '<null>' in t_a_clean else 0.0
    if not addr_missing and s1_a_clean:
        addr_token_sort = fuzz.token_sort_ratio(s1_a_clean, t_a_clean) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1_a_clean, t_a_clean) / 100.0
        addr_partial = fuzz.partial_ratio(s1_a_clean, t_a_clean) / 100.0
        
        s1_atoks = set(extract_addr_tokens(s1_a_str))
        t_atoks = set(extract_addr_tokens(t_a_str))
        at_union = len(s1_atoks | t_atoks)
        addr_jaccard = (len(s1_atoks & t_atoks) / at_union) if at_union > 0 else 0.0
        addr_overlap_count = float(len(s1_atoks & t_atoks))
    else:
        addr_token_sort = 0.0
        addr_token_set = 0.0
        addr_partial = 0.0
        addr_jaccard = 0.0
        addr_overlap_count = 0.0
        
    # 3. Numeric street address consistency
    s1_digs = set(extract_digits(s1_a_str))
    t_digs = set(extract_digits(t_a_str))
    if s1_digs and t_digs:
        if s1_digs == t_digs:
            digit_status = 2.0
        elif s1_digs & t_digs:
            digit_status = 1.0
        else:
            digit_status = -1.0 # Conflict
    else:
        digit_status = 0.0
        
    is_s2 = 1.0 if tgt_id.startswith('S2-') else 0.0
    len_diff = float(abs(len(s1_n_clean) - len(t_n_clean)))
    
    return [
        name_ratio, name_token_sort, name_token_set, name_partial,
        name_jaccard, name_overlap_count,
        addr_token_sort, addr_token_set, addr_partial,
        addr_jaccard, addr_overlap_count,
        digit_status, addr_missing, is_s2, len_diff
    ]
