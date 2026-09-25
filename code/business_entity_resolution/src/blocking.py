import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .preprocessing import (
    clean_tokens,
    clean_compact_name,
    extract_digits,
    extract_addr_tokens,
)


class CandidateGenerator:
    """
    High-recall multi-pass candidate generator.

    Retrieval passes:
      1. Exact compact-name
      2. Compact-name prefixes
      3. Rare name tokens
      4. Name token pairs
      5. Address digit + token
      6. Address digit only
      7. Address token
      8. Name + address evidence

    The important difference from the old blocker is that candidates are
    generated through multiple independent paths before the final top-K cut.
    """

    def __init__(
        self,
        max_cand_per_key=300,
        default_top_k=96,
        max_raw_candidates=500,
    ):
        self.max_cand_per_key = max_cand_per_key
        self.default_top_k = default_top_k
        self.max_raw_candidates = max_raw_candidates

        self.name_tok_index = defaultdict(list)
        self.name_pair_index = defaultdict(list)

        self.compact_index = defaultdict(list)
        self.compact_prefix6_index = defaultdict(list)
        self.compact_prefix5_index = defaultdict(list)

        self.addr_key_index = defaultdict(list)
        self.addr_digit_index = defaultdict(list)
        self.addr_token_index = defaultdict(list)

        self.t_ids = None
        self.t_names = None
        self.t_addrs = None
        self.N_targets = 0

    def fit(self, targets_df):
        """Build inverted indexes over the complete target population."""

        self.t_ids = targets_df["entity_id"].values
        self.t_names = targets_df["business_name"].values
        self.t_addrs = targets_df["business_address"].values
        self.N_targets = len(targets_df)

        self.name_tok_index = defaultdict(list)
        self.name_pair_index = defaultdict(list)

        self.compact_index = defaultdict(list)
        self.compact_prefix6_index = defaultdict(list)
        self.compact_prefix5_index = defaultdict(list)

        self.addr_key_index = defaultdict(list)
        self.addr_digit_index = defaultdict(list)
        self.addr_token_index = defaultdict(list)

        for i in range(self.N_targets):
            name = self.t_names[i]
            addr = self.t_addrs[i]

            # ---------------------------------------------------------
            # NAME INDEXES
            # ---------------------------------------------------------

            toks = sorted(set(clean_tokens(name)))

            # Individual name tokens
            for tok in toks:
                if tok:
                    self.name_tok_index[tok].append(i)

            # Token pairs
            if len(toks) >= 2:
                for a in range(len(toks)):
                    for b in range(a + 1, len(toks)):
                        key = (toks[a], toks[b])
                        self.name_pair_index[key].append(i)

            # Compact name
            cname = clean_compact_name(name)

            if cname:
                self.compact_index[cname].append(i)

                if len(cname) >= 6:
                    self.compact_prefix6_index[cname[:6]].append(i)

                if len(cname) >= 5:
                    self.compact_prefix5_index[cname[:5]].append(i)

            # ---------------------------------------------------------
            # ADDRESS INDEXES
            # ---------------------------------------------------------

            digs = list(dict.fromkeys(extract_digits(addr)))
            atoks = list(dict.fromkeys(extract_addr_tokens(addr)))

            # digit + address token
            for d in digs[:3]:
                for w in atoks[:3]:
                    self.addr_key_index[(d, w)].append(i)

            # digit-only
            for d in digs[:3]:
                self.addr_digit_index[d].append(i)

            # address-token-only
            for w in atoks[:4]:
                if w:
                    self.addr_token_index[w].append(i)

    # ================================================================
    # INTERNAL HELPERS
    # ================================================================

    def _add_postings(
        self,
        cand_scores,
        postings,
        score,
        max_items=None,
    ):
        """
        Add postings to candidate scores.

        Large posting lists are skipped because they have poor
        discriminative value and can create millions of candidates.
        """

        if not postings:
            return

        limit = self.max_cand_per_key if max_items is None else max_items

        if len(postings) > limit:
            return

        for idx in postings:
            cand_scores[idx] += score

    # ================================================================
    # QUERY
    # ================================================================

    def query(self, s1_name, s1_addr, top_k=None):
        """
        Retrieve candidates using multiple independent blocking passes.
        """

        if top_k is None:
            top_k = self.default_top_k

        cand_scores = defaultdict(float)

        # -------------------------------------------------------------
        # Normalize query
        # -------------------------------------------------------------

        cname = clean_compact_name(s1_name)
        toks = list(dict.fromkeys(clean_tokens(s1_name)))
        toks = [t for t in toks if t]

        digs = list(dict.fromkeys(extract_digits(s1_addr)))
        atoks = list(dict.fromkeys(extract_addr_tokens(s1_addr)))

        # =============================================================
        # PASS 1: EXACT COMPACT NAME
        # =============================================================

        if cname:
            postings = self.compact_index.get(cname, [])

            self._add_postings(
                cand_scores,
                postings,
                score=40.0,
                max_items=self.max_cand_per_key * 2,
            )

        # =============================================================
        # PASS 2: COMPACT PREFIX 6
        # =============================================================

        if len(cname) >= 6:
            postings = self.compact_prefix6_index.get(cname[:6], [])

            self._add_postings(
                cand_scores,
                postings,
                score=24.0,
            )

        # =============================================================
        # PASS 3: COMPACT PREFIX 5
        # =============================================================

        if len(cname) >= 5:
            postings = self.compact_prefix5_index.get(cname[:5], [])

            self._add_postings(
                cand_scores,
                postings,
                score=16.0,
            )

        # =============================================================
        # PASS 4: NAME TOKEN PAIRS
        # =============================================================

        if len(toks) >= 2:
            # Prefer rare tokens for pair construction.
            tok_freq = []

            for t in toks:
                freq = len(self.name_tok_index.get(t, []))
                if freq > 0:
                    tok_freq.append((freq, t))

            tok_freq.sort()

            selected = [t for _, t in tok_freq[:4]]

            pair_candidates = []

            for a in range(len(selected)):
                for b in range(a + 1, len(selected)):
                    pair_candidates.append(
                        tuple(sorted((selected[a], selected[b])))
                    )

            for pair in pair_candidates:
                postings = self.name_pair_index.get(pair, [])

                self._add_postings(
                    cand_scores,
                    postings,
                    score=22.0,
                )

        # =============================================================
        # PASS 5: RARE INDIVIDUAL NAME TOKENS
        # =============================================================

        token_info = []

        for tok in toks:
            freq = len(self.name_tok_index.get(tok, []))

            if freq > 0:
                token_info.append((freq, tok))

        token_info.sort()

        # Use several rare tokens instead of stopping after one.
        for freq, tok in token_info[:6]:

            postings = self.name_tok_index.get(tok, [])

            if not postings:
                continue

            if freq <= self.max_cand_per_key:
                # Rare token: strong evidence.
                score = 14.0 / (1.0 + math.log1p(freq))

                self._add_postings(
                    cand_scores,
                    postings,
                    score=score,
                )

        # =============================================================
        # PASS 6: ADDRESS DIGIT + TOKEN
        # =============================================================

        for d in digs[:3]:
            for w in atoks[:3]:

                postings = self.addr_key_index.get((d, w), [])

                self._add_postings(
                    cand_scores,
                    postings,
                    score=18.0,
                )

        # =============================================================
        # PASS 7: ADDRESS DIGIT ONLY
        # =============================================================

        for d in digs[:3]:

            postings = self.addr_digit_index.get(d, [])

            self._add_postings(
                cand_scores,
                postings,
                score=7.0,
            )

        # =============================================================
        # PASS 8: ADDRESS TOKEN ONLY
        # =============================================================

        # Prefer rare address tokens.
        addr_info = []

        for w in atoks:
            freq = len(self.addr_token_index.get(w, []))

            if freq > 0:
                addr_info.append((freq, w))

        addr_info.sort()

        for freq, w in addr_info[:5]:

            postings = self.addr_token_index.get(w, [])

            if freq <= self.max_cand_per_key:

                score = 5.0 / (1.0 + math.log1p(freq))

                self._add_postings(
                    cand_scores,
                    postings,
                    score=score,
                )

        # =============================================================
        # NO CANDIDATES
        # =============================================================

        if not cand_scores:
            return []

        # =============================================================
        # LIMIT RAW CANDIDATE POOL
        # =============================================================

        if len(cand_scores) > self.max_raw_candidates:

            ranked = sorted(
                cand_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            cand_scores = dict(
                ranked[:self.max_raw_candidates]
            )

        # =============================================================
        # FINAL TOP-K
        # =============================================================

        ranked = sorted(
            cand_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [idx for idx, _ in ranked[:top_k]]