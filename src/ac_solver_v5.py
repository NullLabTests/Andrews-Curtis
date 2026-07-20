#!/usr/bin/env python3
"""
Andrews–Curtis Conjecture — V5 Merged Solver
==============================================

Merges the best ideas from all prior versions:

  V3 (working dir): two-track beam, stabilization, restart injection
  V3 (zip):        depth-aware scoring tl + 0.7*cl + 0.01*depth
  V4:              structural potential, per-class diversity, subst verifier
  PRO:             genetic crossover operator
  NEW:             bidirectional (meet-in-the-middle) search

AK(3) = ⟨a,b | a³b⁴, abab⁻¹a⁻¹b⁻¹⟩ remains the primary open target.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional

import ac_solver_v3 as v3

# Re-export key primitives from V3
Letter = v3.Letter
Presentation = v3.Presentation
free_reduce = v3.free_reduce
invert = v3.invert
cyclic_rotate = v3.cyclic_rotate
cyclic_canonical = v3.cyclic_canonical
cyclic_length = v3.cyclic_length
word_str = v3.word_str
enumerate_moves_12rule = v3.enumerate_moves_12rule
enumerate_moves_full = v3.enumerate_moves_full


def weak_state_key(pres: Presentation) -> tuple:
    """Weaker state key: free-reduce only, NO cyclic canonical, NO sorting.

    Two presentations that differ only by relator rotation or inverse
    are treated as DISTINCT, opening a richer search graph for AK(3).
    """
    red = tuple(
        tuple(free_reduce(r)) for r in pres.relators
    )
    return (pres.n_gens, red)


# ═══════════════════════════════════════════════════════════════════════
# 1.  Structural potential analysis (from V4)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PotentialFeatures:
    total_length: int = 0
    cyclic_length: int = 0
    max_relator_length: int = 0
    adjacent_cancellations: int = 0
    shared_letters: int = 0
    isolated_letters: int = 0
    letter_jaccard: float = 0.0
    single_letter_relators: int = 0
    empty_relators: int = 0
    subword_contains: int = 0
    exponent_balance: float = 0.0

    def composite_score(self, depth: int = 0,
                        use_potential: bool = True) -> tuple:
        tl = self.total_length
        cl = self.cyclic_length
        if not use_potential:
            return (tl + cl, 0, 0, tl)

        structural_bonus = (
            + self.single_letter_relators * 5
            + self.empty_relators * 3
            + self.adjacent_cancellations * 1
            + self.shared_letters * 2
            - self.isolated_letters * 3
        )

        # Depth-aware length budget (from zip V3 scoring)
        depth_tolerance = depth // 6
        length_score = max(0, tl - depth_tolerance)

        return (cl, -structural_bonus, length_score, tl)

    def scalar_score(self, depth: int = 0) -> float:
        """Combined scalar for quick ranking (zip V3 style + potential)."""
        tl = self.total_length
        cl = self.cyclic_length
        bonus = (
            self.single_letter_relators * 5
            + self.empty_relators * 3
            + self.adjacent_cancellations * 0.5
            + self.shared_letters * 1
            - self.isolated_letters * 2
        )
        return tl + 0.7 * cl + 0.01 * depth - bonus


def analyze_potential(pres: Presentation) -> PotentialFeatures:
    f = PotentialFeatures()
    f.total_length = pres.total_length()
    f.cyclic_length = pres.total_cyclic_length()
    f.max_relator_length = pres.max_length()

    rels = pres.relators
    n = pres.n_gens

    f.single_letter_relators = sum(1 for r in rels if len(r) == 1)
    f.empty_relators = sum(1 for r in rels if len(r) == 0)

    for r in rels:
        if len(r) >= 2:
            for k in range(len(r)):
                rot = cyclic_rotate(r, k)
                if rot[0] == -rot[-1]:
                    f.adjacent_cancellations += 1

    gen_sets = [set(abs(a) for a in r) for r in rels]
    if len(gen_sets) >= 2:
        common = set.intersection(*gen_sets)
        all_gens = set.union(*gen_sets)
        f.shared_letters = len(common)
        f.isolated_letters = len(all_gens) - len(common)
        f.letter_jaccard = len(common) / max(len(all_gens), 1)

    for i, r in enumerate(rels):
        for j, s in enumerate(rels):
            if i != j and len(s) > 0 and len(s) <= len(r):
                for k in range(len(r)):
                    rot_k = tuple(cyclic_rotate(r, k))
                    if len(rot_k) >= len(s) and tuple(s) == rot_k[:len(s)]:
                        f.subword_contains += 1
                        break

    exp_sums = []
    for g in range(1, n + 1):
        total = 0
        for r in rels:
            total += sum(1 for a in r if a == g) - sum(1 for a in r if a == -g)
        exp_sums.append(abs(total))
    f.exponent_balance = sum(exp_sums) / max(n, 1)

    return f


# ═══════════════════════════════════════════════════════════════════════
# 2.  Cached substitution moves (performance fix)
# ═══════════════════════════════════════════════════════════════════════

_subst_cache: dict[tuple, list[tuple[str, Presentation]]] = {}


def get_cached_substitution_moves(pres: Presentation) -> list[tuple[str, Presentation]]:
    """Cached version of all_substitution_moves()."""
    key = pres.state_key()
    if key not in _subst_cache:
        _subst_cache[key] = pres.all_substitution_moves()
    return _subst_cache[key]


def clear_subst_cache():
    _subst_cache.clear()


# ═══════════════════════════════════════════════════════════════════════
# 3.  Move enumeration (with optional stabilization, from V3 working dir)
# ═══════════════════════════════════════════════════════════════════════

MoveRecord = tuple[str, Callable[[Presentation], Presentation]]


def enumerate_moves_with_stabilization(p: Presentation) -> list[MoveRecord]:
    """Full move set: 12-rule + permute + generator Nielsen + stabilization."""
    moves = enumerate_moves_12rule(p)
    n = p.n_gens

    for i in range(n - 1):
        perm = list(range(n))
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        moves.append((f"SWAP r{i}↔r{i+1}", lambda q, p_=perm: q.permute(p_)))

    for g in range(1, n + 1):
        gm = {a: [a] for a in range(1, n + 1)}
        gm[g] = [-g]
        moves.append((f"GENINV x{g}", lambda q, m=gm: q.map_gens(m)))

    for g in range(1, n + 1):
        for h in range(1, n + 1):
            if g == h:
                continue
            gm = {a: [a] for a in range(1, n + 1)}
            gm[g] = [g, h]
            moves.append((f"GENMUL x{g}→x{g}x{h}", lambda q, m=gm: q.map_gens(m)))

    moves.append(("STAB_ADD", lambda q: q.stab_add()))
    sr = p.stab_remove()
    if sr is not None:
        moves.append(("STAB_REMOVE", lambda q: q.stab_remove()))

    return moves


# ═══════════════════════════════════════════════════════════════════════
# 4.  Bidirectional beam search
# ═══════════════════════════════════════════════════════════════════════

SearchResult = tuple[bool, Optional[list[tuple[str, Presentation]]], dict]


@dataclass
class BidirectionalBeamSearcher:
    """Bidirectional beam search with potential-guided scoring.

    Runs forward from start and backward from trivial simultaneously.
    Detects intersection to find long non-monotonic paths.
    """

    beam_width: int = 2000
    max_depth: int = 500
    max_states: int = 50_000_000
    timeout: float = 7200.0
    use_substitution_moves: bool = True
    use_generator_moves: bool = True
    use_potential_scoring: bool = True
    max_per_class: int = 5
    restart_patience: int = 30
    restart_fraction: float = 0.3
    verbose: bool = True
    results_dir: str = "results"
    seed: int = 0
    weak_nf: bool = False

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        if self.seed:
            random.seed(self.seed)

    def _key(self, pres: Presentation) -> tuple:
        return weak_state_key(pres) if self.weak_nf else pres.state_key()

    def _make_move_set(self, p: Presentation) -> list[MoveRecord]:
        if self.use_generator_moves:
            return enumerate_moves_with_stabilization(p)
        return enumerate_moves_12rule(p)

    def _save_checkpoint(self, run_id: str, data: dict):
        fname = f"{self.results_dir}/{run_id}_checkpoint.json"
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)

    def _expand_frontier(
        self,
        frontier: list[tuple[list[tuple[str, Presentation]], Presentation]],
        visited: dict,
        depth: int,
        use_subst: bool,
        forward: bool,
        total_states: int,
        t0: float,
        pres_store: Optional[dict] = None,
    ) -> tuple[list[tuple], int, Optional[tuple]]:
        """Expand one level of the frontier. Returns (candidates, total_states, found_intersection)."""
        candidates: list[tuple] = []

        for path, pres in frontier:
            if time.monotonic() - t0 > self.timeout:
                break

            moves = self._make_move_set(pres)

            for move_desc, move_fn in moves:
                new_pres = move_fn(pres)
                key = self._key(new_pres)

                if key in visited and visited[key] <= depth + 1:
                    continue
                visited[key] = depth + 1
                total_states += 1
                if pres_store is not None:
                    pres_store[key] = new_pres

                new_path = path + [(move_desc, new_pres)]

                if forward and new_pres.is_solved():
                    return candidates, total_states, ("solved", new_path, new_pres)

                feat = analyze_potential(new_pres)
                if self.use_potential_scoring:
                    score = feat.composite_score(depth + 1, True)
                else:
                    score = (feat.total_length + feat.cyclic_length, 0, 0, feat.total_length)

                eclass = (new_pres.total_length(),
                          new_pres.total_cyclic_length(),
                          feat.shared_letters)

                candidates.append(
                    (score, eclass, len(new_path), total_states, new_path, new_pres))

            # Substitution super-moves (forward only — cached)
            if use_subst and forward and pres.n_gens == 2:
                for sub_desc, sub_res in get_cached_substitution_moves(pres):
                    key = self._key(sub_res)
                    if key in visited and visited[key] <= depth + 1:
                        continue
                    visited[key] = depth + 1
                    total_states += 1
                    if pres_store is not None:
                        pres_store[key] = sub_res

                    new_path = path + [(sub_desc, sub_res)]

                    if forward and sub_res.is_solved():
                        return candidates, total_states, ("solved", new_path, sub_res)

                    feat = analyze_potential(sub_res)
                    if self.use_potential_scoring:
                        score = feat.composite_score(depth + 1, True)
                    else:
                        score = (feat.total_length + feat.cyclic_length, 0, 0, feat.total_length)
                    eclass = (sub_res.total_length(),
                              sub_res.total_cyclic_length(),
                              feat.shared_letters)

                    candidates.append(
                        (score, eclass, len(new_path), total_states, new_path, sub_res))

        return candidates, total_states, None

    def _select_frontier(
        self, candidates: list[tuple], beam_width: int, max_per_class: int
    ) -> list[tuple[list, Presentation]]:
        candidates.sort(key=lambda x: x[0])
        class_count: dict = defaultdict(int)
        new_frontier: list = []
        for _, eclass, _, _, path, pres in candidates:
            if len(new_frontier) >= beam_width:
                break
            if max_per_class > 0:
                if class_count[eclass] >= max_per_class:
                    continue
                class_count[eclass] += 1
            new_frontier.append((path, pres))
        return new_frontier

    def search(self, start: Presentation,
               run_id: Optional[str] = None) -> SearchResult:
        if run_id is None:
            run_id = f"bidir_{int(time.time())}"
        t0 = time.monotonic()

        if start.is_solved():
            return True, [], {
                "states_explored": 0, "depth": 0, "reason": "already trivial"
            }

        target = Presentation.trivial(start.n_gens)
        t_key = self._key(target)
        s_key = self._key(start)

        # Forward search: from start toward trivial
        f_visited: dict = {s_key: 0}
        f_frontier: list[tuple[list[tuple[str, Presentation]], Presentation]] = [
            ([], start)
        ]
        f_total = 1
        f_parent: dict = {}  # state_key -> (parent_key, move_desc)
        f_pres: dict = {s_key: start}  # state_key -> Presentation

        # Backward search: from trivial toward start
        b_visited: dict = {t_key: 0}
        b_frontier: list = [([], target)]
        b_total = 1
        b_parent: dict = {}  # state_key -> (parent_key, move_desc)
        b_pres: dict = {t_key: target}

        best_len = start.total_length()
        best_cyclic = start.total_cyclic_length()
        last_report = 0.0
        last_checkpoint = 0.0
        depths_since_improvement = 0

        for depth in range(self.max_depth):
            if time.monotonic() - t0 > self.timeout:
                break

            # --- Expand forward frontier ---
            f_cands, f_total, f_result = self._expand_frontier(
                f_frontier, f_visited, depth, self.use_substitution_moves,
                True, f_total, t0, pres_store=f_pres,
            )
            if f_result is not None:
                typ, f_path, f_state = f_result
                return self._found(f_path, f_state, f_total + b_total,
                                   t0, run_id)

            # Track parents for forward expansion
            for _, _, _, _, path, pres in f_cands:
                if len(path) > 0:
                    child_key = self._key(pres)
                    if child_key in f_parent:
                        continue
                    # parent is state before last move in path
                    if len(path) >= 2:
                        parent_key = self._key(path[-2][1])
                    else:
                        parent_key = s_key
                    f_parent[child_key] = (parent_key, path[-1][0])

            # Check intersection: forward × backward
            for _, _, _, _, path, pres in f_cands:
                key = self._key(pres)
                if key in b_visited:
                    full = self._reconstruct(
                        path, pres, key, f_parent, b_parent,
                        start, target, f_pres, b_pres)
                    return self._found(full, pres, f_total + b_total,
                                       t0, run_id)

            f_frontier = self._select_frontier(
                f_cands, self.beam_width, self.max_per_class)

            # --- Expand backward frontier ---
            b_cands, b_total, b_result = self._expand_frontier(
                b_frontier, b_visited, depth, False,
                False, b_total, t0, pres_store=b_pres,
            )
            if b_result is not None:
                typ, b_path, b_state = b_result
                rev = [(f"INV:{d}", s) for d, s in reversed(b_path)]
                return self._found(rev, b_state, f_total + b_total,
                                   t0, run_id)

            # Track parents for backward expansion
            for _, _, _, _, path, pres in b_cands:
                if len(path) > 0:
                    child_key = self._key(pres)
                    if child_key in b_parent:
                        continue
                    if len(path) >= 2:
                        parent_key = self._key(path[-2][1])
                    else:
                        parent_key = t_key
                    b_parent[child_key] = (parent_key, path[-1][0])

            # Check intersection: backward × forward
            for _, _, _, _, path, pres in b_cands:
                key = self._key(pres)
                if key in f_visited:
                    full = self._reconstruct_backward(
                        path, pres, key, f_parent, b_parent,
                        start, target, f_pres, b_pres)
                    return self._found(full, pres, f_total + b_total,
                                       t0, run_id)

            b_frontier = self._select_frontier(
                b_cands, self.beam_width, self.max_per_class)

            if f_total + b_total >= self.max_states:
                break

            # Update best
            if f_frontier:
                bl = f_frontier[0][1].total_length()
                bc = f_frontier[0][1].total_cyclic_length()
                if bl < best_len:
                    best_len = bl
                if bc < best_cyclic:
                    best_cyclic = bc
                    depths_since_improvement = 0

            depths_since_improvement += 1

            # Restart forward when stalled
            if (depths_since_improvement >= self.restart_patience
                    and len(f_frontier) >= 4):
                inject_n = max(1, int(len(f_frontier) * self.restart_fraction))
                seen_structures: set = set()
                diverse: list = []
                for _, _, _, _, path, pres in f_cands:
                    sig = tuple(sorted(len(r) for r in pres.relators))
                    if sig not in seen_structures:
                        seen_structures.add(sig)
                        diverse.append((path, pres))
                        if len(diverse) >= inject_n:
                            break
                if diverse:
                    f_frontier = f_frontier[:len(f_frontier) - inject_n] + diverse
                depths_since_improvement = 0
                if self.verbose:
                    print(f"  ⚡ F-RESTART: injected {len(diverse)} states "
                          f"at depth {depth+1}", flush=True)

            # Reporting
            now = time.monotonic()
            if self.verbose and now - last_report > 15.0:
                f_size = len(f_visited)
                b_size = len(b_visited)
                inter = len(set(f_visited.keys()) & set(b_visited.keys()))
                print(
                    f"  [depth={depth+1:3d}  f_beam={len(f_frontier):4d}  "
                    f"b_beam={len(b_frontier):4d}  "
                    f"f_states={f_size:6d}  b_states={b_size:6d}  "
                    f"inter={inter:3d}  "
                    f"best_len={best_len:2d}  "
                    f"elapsed={now-t0:.0f}s]",
                    flush=True,
                )
                last_report = now

            # Checkpoint
            if now - last_checkpoint > 120.0:
                self._save_checkpoint(run_id, {
                    "found": False, "depth": depth + 1,
                    "f_states": len(f_visited),
                    "b_states": len(b_visited),
                    "intersection": inter,
                    "best_len": best_len,
                    "best_cyclic": best_cyclic,
                    "elapsed": now - t0,
                })
                last_checkpoint = now

        elapsed = time.monotonic() - t0
        return False, None, {
            "states_explored": f_total + b_total,
            "f_states": len(f_visited),
            "b_states": len(b_visited),
            "depth": depth + 1 if depth < self.max_depth else self.max_depth,
            "best_length": best_len,
            "best_cyclic": best_cyclic,
            "reason": "bidirectional search exhausted",
            "elapsed": elapsed,
        }

    def _reconstruct(self, f_path, f_state, meeting_key,
                     f_parent, b_parent, start, target,
                     f_pres, b_pres):
        """Reconstruct full path: start → meeting → trivial."""
        fwd = list(f_path)

        bwd = []
        cur_key = meeting_key
        while cur_key != self._key(target):
            if cur_key not in b_parent:
                break
            pk, desc = b_parent[cur_key]
            next_state = b_pres.get(pk, target)
            bwd.append((f"INV:{desc}", next_state))
            cur_key = pk

        return fwd + bwd

    def _reconstruct_backward(self, b_path, b_state, meeting_key,
                               f_parent, b_parent, start, target,
                               f_pres, b_pres):
        """Reconstruct full path: start → meeting → trivial."""
        fwd = []
        cur_key = meeting_key
        chain = []
        while cur_key != self._key(start):
            if cur_key not in f_parent:
                break
            pk, desc = f_parent[cur_key]
            state_at_cur = f_pres.get(cur_key, b_state)
            chain.append((desc, state_at_cur))
            cur_key = pk
        chain.reverse()
        fwd = chain

        bwd = []
        for i in range(len(b_path) - 1, -1, -1):
            desc, state = b_path[i]
            if i > 0:
                prev_state = b_path[i-1][1]
            else:
                prev_state = target
            bwd.append((f"INV:{desc}", prev_state))

        return fwd + bwd

    def _found(self, path, state, total_states, t0, run_id):
        if self.verbose:
            print(f"\n  ✓ FOUND after {total_states} states!", flush=True)
        elapsed = time.monotonic() - t0
        stats = {
            "states_explored": total_states,
            "depth": len(path),
            "reason": "found",
            "elapsed": elapsed,
        }
        result = {"found": True, "path_len": len(path), "stats": stats}
        self._save_checkpoint(run_id + "_FOUND", result)
        return True, path, stats


# ═══════════════════════════════════════════════════════════════════════
# 5.  Forward-only beam search (merged V3 + V4 + zip)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MergedBeamSearcher:
    """Forward-only beam search merging zip scoring + working-dir features."""

    beam_width: int = 2000
    max_depth: int = 500
    max_states: int = 50_000_000
    timeout: float = 7200.0
    use_substitution_moves: bool = True
    use_generator_moves: bool = True
    use_potential_scoring: bool = True
    subst_slots: int = 0
    max_per_class: int = 5
    restart_patience: int = 30
    restart_fraction: float = 0.3
    verbose: bool = True
    results_dir: str = "results"
    seed: int = 0
    weak_nf: bool = False

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        if self.seed:
            random.seed(self.seed)

    def _key(self, pres: Presentation) -> tuple:
        return weak_state_key(pres) if self.weak_nf else pres.state_key()

    def _make_move_set(self, p: Presentation) -> list[MoveRecord]:
        if self.use_generator_moves:
            return enumerate_moves_with_stabilization(p)
        return enumerate_moves_12rule(p)

    def _save_checkpoint(self, run_id: str, data: dict):
        fname = f"{self.results_dir}/{run_id}_checkpoint.json"
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)

    def search(self, start: Presentation,
               run_id: Optional[str] = None) -> SearchResult:
        if run_id is None:
            run_id = f"merged_{int(time.time())}"
        t0 = time.monotonic()

        if start.is_solved():
            return True, [], {
                "states_explored": 0, "depth": 0, "reason": "already trivial"
            }

        visited: dict = {self._key(start): 0}
        total_states = 0
        start_feat = analyze_potential(start)
        best_score = start_feat.composite_score(0, self.use_potential_scoring)
        best_len = start.total_length()
        best_cyclic = start.total_cyclic_length()
        depths_since_improvement = 0
        last_report = 0.0
        last_checkpoint = 0.0

        frontier = [([], start)]

        subst_slots = self.subst_slots or max(self.beam_width // 4, 1)
        regular_slots = self.beam_width - subst_slots

        for depth in range(self.max_depth):
            if not frontier or time.monotonic() - t0 > self.timeout:
                break

            regular_candidates: list[tuple] = []
            subst_candidates: list[tuple] = []

            for path, pres in frontier:
                if time.monotonic() - t0 > self.timeout:
                    break

                for move_desc, move_fn in self._make_move_set(pres):
                    new_pres = move_fn(pres)
                    key = self._key(new_pres)
                    if key in visited and visited[key] <= depth + 1:
                        continue
                    visited[key] = depth + 1
                    total_states += 1
                    new_path = path + [(move_desc, new_pres)]

                    if new_pres.is_solved():
                        if self.verbose:
                            print(f"\n  ✓ FOUND at depth {depth+1} after "
                                  f"{total_states} states!", flush=True)
                        elapsed = time.monotonic() - t0
                        stats = {"states_explored": total_states,
                                 "depth": depth + 1, "reason": "found",
                                 "elapsed": elapsed}
                        self._save_checkpoint(run_id + "_FOUND",
                                              {"found": True, "stats": stats})
                        return True, new_path, stats

                    feat = analyze_potential(new_pres)
                    if self.use_potential_scoring:
                        score_tuple = feat.composite_score(depth + 1, True)
                        score = feat.scalar_score(depth + 1)
                    else:
                        score_tuple = (feat.total_length + feat.cyclic_length,
                                       0, 0, feat.total_length)
                        score = feat.total_length + 0.7 * feat.cyclic_length

                    eclass = (new_pres.total_length(),
                              new_pres.total_cyclic_length(),
                              feat.shared_letters)

                    regular_candidates.append(
                        (score_tuple, eclass, len(new_path),
                         total_states, new_path, new_pres))

                # Cached substitution super-moves
                if self.use_substitution_moves:
                    for sub_desc, sub_res in get_cached_substitution_moves(pres):
                        key = self._key(sub_res)
                        if key in visited and visited[key] <= depth + 1:
                            continue
                        visited[key] = depth + 1
                        total_states += 1
                        new_path = path + [(sub_desc, sub_res)]

                        if sub_res.is_solved():
                            if self.verbose:
                                print(f"\n  ✓ FOUND via super-move at depth "
                                      f"{depth+1} after {total_states}!",
                                      flush=True)
                            elapsed = time.monotonic() - t0
                            stats = {"states_explored": total_states,
                                     "depth": depth + 1,
                                     "reason": "found via super-move",
                                     "elapsed": elapsed}
                            self._save_checkpoint(run_id + "_FOUND",
                                                  {"found": True, "stats": stats})
                            return True, new_path, stats

                        feat = analyze_potential(sub_res)
                        if self.use_potential_scoring:
                            score_tuple = feat.composite_score(depth + 1, True)
                            score = feat.scalar_score(depth + 1)
                        else:
                            score_tuple = (feat.total_length +
                                           feat.cyclic_length, 0, 0,
                                           feat.total_length)
                            score = feat.total_length + 0.7 * feat.cyclic_length

                        eclass = (sub_res.total_length(),
                                  sub_res.total_cyclic_length(),
                                  feat.shared_letters)

                        subst_candidates.append(
                            (score_tuple, eclass, len(new_path),
                             total_states, new_path, sub_res))

            if total_states >= self.max_states:
                break

            # Two-track beam with per-class diversity
            regular_candidates.sort(key=lambda x: x[0])
            subst_candidates.sort(key=lambda x: x[0])

            def select_diverse(cands, n, max_per):
                out = []
                cc: dict = defaultdict(int)
                for item in cands:
                    if len(out) >= n:
                        break
                    eclass = item[2]
                    if max_per > 0 and cc[eclass] >= max_per:
                        continue
                    cc[eclass] += 1
                    out.append((item[4], item[5]))
                return out

            new_frontier = (
                select_diverse(regular_candidates, regular_slots, self.max_per_class)
                +
                select_diverse(subst_candidates, subst_slots, self.max_per_class)
            )
            frontier = new_frontier

            # Update best
            if frontier:
                bp = frontier[0][1]
                bl, bc = bp.total_length(), bp.total_cyclic_length()
                bf = analyze_potential(bp)
                bs = bf.composite_score(depth, self.use_potential_scoring)
                if bs < best_score:
                    best_score = bs
                    depths_since_improvement = 0
                if bl < best_len:
                    best_len = bl
                if bc < best_cyclic:
                    best_cyclic = bc

            # Restart when stalled
            depths_since_improvement += 1
            if (depths_since_improvement >= self.restart_patience
                    and len(frontier) >= 4):
                inject_n = max(1, int(len(frontier) * self.restart_fraction))
                all_cands = regular_candidates + subst_candidates
                seen_structures: set = set()
                diverse: list = []
                for _, _, _, _, path, pres in all_cands:
                    sig = tuple(sorted(len(r) for r in pres.relators))
                    if sig not in seen_structures:
                        seen_structures.add(sig)
                        diverse.append((path, pres))
                        if len(diverse) >= inject_n:
                            break
                if diverse:
                    frontier = frontier[:len(frontier) - inject_n] + diverse
                depths_since_improvement = 0
                if self.verbose:
                    print(f"  ⚡ RESTART: injected {len(diverse)} diverse "
                          f"states at depth {depth+1}", flush=True)

            # Reporting
            now = time.monotonic()
            if self.verbose and now - last_report > 10.0:
                print(
                    f"  [depth={depth+1:3d}  beam={len(frontier):4d}  "
                    f"states={total_states:8d}  len={best_len:2d}  "
                    f"cyclic={best_cyclic:2d}  "
                    f"elapsed={now-t0:.0f}s]",
                    flush=True,
                )
                last_report = now

            if now - last_checkpoint > 60.0:
                self._save_checkpoint(run_id, {
                    "found": False, "depth": depth + 1,
                    "states_explored": total_states,
                    "best_len": best_len, "best_cyclic": best_cyclic,
                    "elapsed": now - t0,
                })
                last_checkpoint = now

        elapsed = time.monotonic() - t0
        return False, None, {
            "states_explored": total_states,
            "depth": depth + 1 if frontier else depth,
            "best_length": best_len,
            "best_cyclic": best_cyclic,
            "reason": "search exhausted",
            "elapsed": elapsed,
        }


# ═══════════════════════════════════════════════════════════════════════
# 6.  Unit tests (from zip)
# ═══════════════════════════════════════════════════════════════════════

def run_tests():
    failures = 0
    # Test 1: standard trivial detected
    p = Presentation.trivial(2)
    assert p.is_standard_trivial(), "standard trivial not recognised"
    assert p.is_solved()
    print("✓ standard trivial detection")

    # Test 2: cyclic canonical prefers positive
    assert cyclic_canonical([1]) == [1]
    assert cyclic_canonical([-1]) == [1]
    assert cyclic_canonical([2]) == [2]
    assert cyclic_canonical([-2]) == [2]
    print("✓ cyclic_canonical total order")

    # Test 3: AK(3) structure
    ak3 = v3.example_ak3()
    assert ak3.total_length() == 13
    assert not ak3.is_solved()
    print("✓ AK(3) presentation correct")

    # Test 4: Myasnikov solves quickly
    pres = v3.example_myasnikov()
    searcher = MergedBeamSearcher(
        beam_width=300, max_depth=40, max_states=200_000,
        timeout=90.0, verbose=False,
        results_dir="/tmp/ac_test_results",
    )
    found, path, stats = searcher.search(pres, run_id="test_myasnikov")
    assert found, f"Myasnikov not solved: {stats}"
    assert path is not None and len(path) > 0
    print(f"✓ Myasnikov solved in {stats['depth']} steps "
          f"({stats['states_explored']} states)")

    # Test 5: cached substitution works
    clear_subst_cache()
    m1 = get_cached_substitution_moves(ak3)
    m2 = get_cached_substitution_moves(ak3)
    assert len(m1) == len(m2)
    assert id(m1) == id(m2)  # same object from cache
    print("✓ substitution caching")

    print("\nAll tests passed.")
    return True


# ═══════════════════════════════════════════════════════════════════════
# 7.  Macro miner (from V4, with caching)
# ═══════════════════════════════════════════════════════════════════════

def random_balanced_presentation(n_gens: int = 2,
                                  max_len: int = 8,
                                  seed: Optional[int] = None) -> Presentation:
    if seed is not None:
        random.seed(seed)
    rels = []
    for _ in range(n_gens):
        length = random.randint(1, max_len)
        word = []
        for _ in range(length):
            gen = random.randint(1, n_gens)
            if random.random() < 0.3:
                gen = -gen
            word.append(gen)
        rels.append(free_reduce(word))
    return Presentation.from_lists(n_gens, rels)


def is_trivial_by_search(pres: Presentation,
                         beam_width: int = 100,
                         max_depth: int = 25,
                         max_states: int = 100_000,
                         timeout: float = 30.0) -> tuple[bool, Optional[list]]:
    searcher = MergedBeamSearcher(
        beam_width=beam_width, max_depth=max_depth,
        max_states=max_states, timeout=timeout,
        use_substitution_moves=True, use_generator_moves=True,
        verbose=False,
    )
    found, path, _ = searcher.search(pres,
                                      run_id=f"trivcheck_{int(time.time())}")
    return found, path


def mine_macros(num_presentations: int = 5000,
                max_len: int = 8,
                output_file: str = "results/macros.json") -> dict:
    t0 = time.monotonic()
    paths: list[list[str]] = []
    tested = 0
    found = 0

    print(f"Mining macros from {num_presentations} random presentations...",
          flush=True)

    for i in range(num_presentations):
        pres = random_balanced_presentation(2, max_len)
        tested += 1
        trivial, path = is_trivial_by_search(pres)
        if trivial and path:
            found += 1
            paths.append([d for d, _ in path])

        if (i + 1) % 500 == 0:
            elapsed = time.monotonic() - t0
            print(f"  [{i+1}/{num_presentations}] found={found} "
                  f"rate={found/max(tested,1)*100:.1f}% "
                  f"elapsed={elapsed:.0f}s", flush=True)

    ngram_counts: dict = defaultdict(int)
    for path in paths:
        for n in range(2, min(6, len(path) + 1)):
            for j in range(len(path) - n + 1):
                ngram_counts[tuple(path[j:j + n])] += 1

    top_macros = sorted(ngram_counts.items(), key=lambda x: -x[1])[:50]

    result = {
        "num_presentations": num_presentations,
        "max_len": max_len,
        "tested": tested,
        "trivial_found": found,
        "trivial_rate": found / max(tested, 1),
        "elapsed": time.monotonic() - t0,
        "top_macros": [
            {"moves": list(m), "count": c} for m, c in top_macros
        ],
        "example_paths": paths[:10],
    }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nMacros saved to {output_file}")
    print("Top macros:")
    for m, c in top_macros[:10]:
        print(f"  [{c:4d}x] {' → '.join(m[:4])}{'...' if len(m) > 4 else ''}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# 8.  Command line
# ═══════════════════════════════════════════════════════════════════════

EXAMPLES = [
    ("trivial", v3.example_trivial(), "Standard trivial"),
    ("myasnikov", v3.example_myasnikov(), "Known AC-trivializable"),
    ("ak3", v3.example_ak3(), "AK(3) — OPEN primary target"),
    ("ak4", v3.example_ak_general(4), "AK(4) — OPEN"),
    ("ak5", v3.example_ak_general(5), "AK(5) — OPEN"),
    ("presentation_p", v3.example_presentation_p(), "Bridge P (Lisitsa)"),
    ("ms_ak3", v3.example_ms_equivalent_ak3(), "MS(3, y⁻¹x⁻¹yxy) ≅ AK(3)"),
]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="AC Conjecture — V5 Merged Solver")
    parser.add_argument("--example", "-e", type=int, default=2)
    parser.add_argument("--beam", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=500)
    parser.add_argument("--states", type=int, default=20_000_000)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-sub", action="store_true")
    parser.add_argument("--no-gen", action="store_true")
    parser.add_argument("--scalar", action="store_true",
                        help="Disable potential-guided scoring")
    parser.add_argument("--bidir", action="store_true",
                        help="Use bidirectional search")
    parser.add_argument("--max-per-class", type=int, default=5,
                        help="Max states per equivalence class (0=unlimited)")
    parser.add_argument("--weak-nf", action="store_true",
                        help="Use weaker normal form (no cyclic canonical)")
    parser.add_argument("--mine", type=int, default=0,
                        help="Run macro miner with N presentations")
    parser.add_argument("--test", action="store_true",
                        help="Run unit tests")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    if args.mine > 0:
        mine_macros(args.mine)
        return

    name, pres, desc = EXAMPLES[args.example]
    print(f"Target: {name} — {desc}")
    print(f"Start:  {pres.show()}", flush=True)
    print(f"Mode:   {'bidirectional' if args.bidir else 'forward'}  "
          f"scoring={'potential' if not args.scalar else 'scalar'}  "
          f"nf={'weak' if args.weak_nf else 'cyclic'}", flush=True)

    if args.bidir:
        searcher = BidirectionalBeamSearcher(
            beam_width=args.beam, max_depth=args.depth,
            max_states=args.states, timeout=args.timeout,
            use_substitution_moves=not args.no_sub,
            use_generator_moves=not args.no_gen,
            use_potential_scoring=not args.scalar,
            max_per_class=args.max_per_class,
            weak_nf=args.weak_nf,
            verbose=True, seed=args.seed,
        )
    else:
        searcher = MergedBeamSearcher(
            beam_width=args.beam, max_depth=args.depth,
            max_states=args.states, timeout=args.timeout,
            use_substitution_moves=not args.no_sub,
            use_generator_moves=not args.no_gen,
            use_potential_scoring=not args.scalar,
            max_per_class=args.max_per_class,
            weak_nf=args.weak_nf,
            verbose=True, seed=args.seed,
        )

    t0 = time.monotonic()
    found, path, stats = searcher.search(
        pres, run_id=f"{name}_v5_{int(t0)}")
    elapsed = time.monotonic() - t0

    print()
    print("─" * 70)
    print("  RESULTS")
    print("─" * 70)
    if found:
        print(f"  ✓  TRIVIALIZATION FOUND!  Depth = {stats['depth']}")
        if path:
            print(f"\n  Move sequence ({len(path)} steps):")
            ltrs = ["a", "b"] if pres.n_gens <= 2 else None
            for step, (desc, state) in enumerate(path):
                tl = state.total_length()
                print(f"    [{step+1:3d}] {desc:40s}  "
                      f"{state.show(ltrs)}  (len={tl})")
    else:
        print(f"  ✗  NOT FOUND")
        print(f"     Reason: {stats.get('reason', 'unknown')}")

    print(f"\n  Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")

    result_data = {
        "run_id": f"{name}_v5_{int(t0)}",
        "found": found,
        "presentation": name,
        "description": desc,
        "params": {
            "beam_width": args.beam,
            "max_depth": args.depth,
            "max_states": args.states,
            "use_potential": not args.scalar,
            "use_substitution": not args.no_sub,
            "use_generator_moves": not args.no_gen,
            "bidirectional": args.bidir,
            "timeout": args.timeout,
            "seed": args.seed,
        },
        "stats": {k: (v if isinstance(v, (int, float, str)) else str(v))
                  for k, v in stats.items()},
        "elapsed": elapsed,
    }
    if path:
        ltrs = ["a", "b"] if pres.n_gens <= 2 else None
        result_data["path"] = [
            {"move": d, "state": s.show(ltrs), "len": s.total_length(),
             "cyclic": s.total_cyclic_length()}
            for d, s in path
        ]

    os.makedirs("results", exist_ok=True)
    fname = (f"results/{name}_v5_{int(t0)}_"
             f"{'FOUND' if found else 'NOT_FOUND'}.json")
    with open(fname, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  Full results saved to: {fname}")
    print()


if __name__ == "__main__":
    main()
