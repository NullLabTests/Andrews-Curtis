#!/usr/bin/env python3
"""
Andrews–Curtis Conjecture — Corrected Solver V3
================================================

Uses the Caltech substitution super-move definition (cyclic rotation),
Booth's canonical cyclic form for 1600x state-space reduction,
and the 12-rule action space from Shehper et al.

The Andrews–Curtis conjecture (1965) states that every balanced
presentation of the trivial group can be transformed into the standard
trivial presentation via a finite sequence of AC-moves.

Key algorithmic innovations from the literature integrated here:
  [Fagan+ 2026] Substitution super-moves via cyclic rotation
  [Ivanov 2018] Cyclic AC conjecture equivalence
  [Lisitsa 2025] 12-rule Prover9 encoding
  [Shehper+ 2024] GS-Sub with 1600x efficiency

Author : NullLabTests
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Callable, Optional

# ═══════════════════════════════════════════════════════════════════════
# 1.  Word utilities
# ═══════════════════════════════════════════════════════════════════════

Letter = int  # positive = generator; negative = inverse


def free_reduce(word: list[Letter]) -> list[Letter]:
    stack: list[Letter] = []
    for a in word:
        if stack and stack[-1] == -a:
            stack.pop()
        else:
            stack.append(a)
    return stack


def multiply(w1: list[Letter], w2: list[Letter]) -> list[Letter]:
    return free_reduce(w1 + w2)


def invert(word: list[Letter]) -> list[Letter]:
    return [-a for a in reversed(word)]


def conjugate(word: list[Letter], gen: Letter) -> list[Letter]:
    return free_reduce([gen] + word + [-gen])


def word_len(word: list[Letter]) -> int:
    return len(word)


def cyclic_rotate(word: list[Letter], k: int) -> list[Letter]:
    """Rotate a word cyclically by k positions (k can be negative)."""
    if not word:
        return []
    n = len(word)
    k = k % n
    return word[k:] + word[:k]


# -- Booth's algorithm for minimal cyclic rotation --
def _least_rotation(s: list[int]) -> int:
    """Booth's O(n) algorithm: return start index of lexicographically
    minimal rotation of the list s."""
    n = len(s)
    if n <= 1:
        return 0
    s2 = s + s
    f = [-1] * (2 * n)
    k = 0
    for j in range(1, 2 * n):
        i = f[j - k - 1]
        while i != -1 and s2[j] != s2[k + i + 1]:
            if s2[j] < s2[k + i + 1]:
                k = j - i - 1
            i = f[i]
        if s2[j] != s2[k + i + 1]:
            if s2[j] < s2[k]:
                k = j
            f[j - k] = -1
        else:
            f[j - k] = i + 1
    return k


def minimal_cyclic_rotation(word: list[Letter]) -> list[Letter]:
    """Return the lexicographically minimal cyclic rotation of word
    (Booth's algorithm)."""
    if not word:
        return []
    k = _least_rotation(word)
    return word[k:] + word[:k]


def _word_key(w: list[Letter]) -> list[tuple]:
    """Sort key for words: (abs_value, sign) to ensure 1<2<...<-2<-1."""
    return [(abs(a), 0 if a > 0 else 1) for a in w]


def cyclic_canonical(word: list[Letter]) -> list[Letter]:
    """Canonical cyclic form: minimum of minimal rotations of
    the word and its inverse.

    Uses a word-invariant ordering: 1 < 2 < ... < n < -n < ... < -2 < -1.
    """
    if not word:
        return []
    c1 = minimal_cyclic_rotation(word)
    c2 = minimal_cyclic_rotation(invert(word))
    # Compare using the canonical word ordering
    k1 = _word_key(c1)
    k2 = _word_key(c2)
    return c1 if k1 <= k2 else c2


def cyclic_length(word: list[Letter]) -> int:
    """Length of the cyclic-reduced form (cancels prefix-suffix pairs)."""
    if not word:
        return 0
    w = list(word)
    while len(w) >= 2 and w[0] == -w[-1]:
        w = w[1:-1]
    return len(w)


def word_str(word: list[Letter], letters: Optional[list[str]] = None) -> str:
    if not word:
        return "ε"
    parts: list[str] = []
    for a in word:
        if a > 0:
            name = letters[a - 1] if letters and a - 1 < len(letters) else f"x{a}"
        else:
            base = letters[(-a) - 1] if letters and (-a) - 1 < len(letters) else f"x{-a}"
            name = base + "⁻¹"
        parts.append(name)
    sep = "·" if len(parts) <= 6 else ""
    return sep.join(parts)


# ═══════════════════════════════════════════════════════════════════════
# 2.  Presentation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Presentation:
    n_gens: int
    relators: list[list[Letter]]

    def __post_init__(self):
        assert len(self.relators) == self.n_gens

    @staticmethod
    def trivial(n: int = 2) -> Presentation:
        return Presentation(n, [[i] for i in range(1, n + 1)])

    @staticmethod
    def from_lists(n_gens: int, rels: list[list[int]]) -> Presentation:
        return Presentation(n_gens, [free_reduce(r) for r in rels])

    def copy(self) -> Presentation:
        return Presentation(self.n_gens, [list(r) for r in self.relators])

    def total_length(self) -> int:
        return sum(len(r) for r in self.relators)

    def total_cyclic_length(self) -> int:
        return sum(cyclic_length(r) for r in self.relators)

    def max_length(self) -> int:
        return max((len(r) for r in self.relators), default=0)

    # -- Core AC moves (standard) --
    def inv(self, i: int) -> Presentation:
        q = self.copy()
        q.relators[i] = invert(q.relators[i])
        return q

    def mul(self, i: int, j: int, inv_j: bool = False) -> Presentation:
        q = self.copy()
        rhs = invert(q.relators[j]) if inv_j else q.relators[j]
        q.relators[i] = multiply(q.relators[i], rhs)
        return q

    def mul_left(self, i: int, j: int, inv_j: bool = False) -> Presentation:
        q = self.copy()
        lhs = invert(q.relators[j]) if inv_j else q.relators[j]
        q.relators[i] = multiply(lhs, q.relators[i])
        return q

    def conj(self, i: int, gen: Letter) -> Presentation:
        q = self.copy()
        q.relators[i] = conjugate(q.relators[i], gen)
        return q

    def permute(self, perm: list[int]) -> Presentation:
        q = self.copy()
        q.relators = [q.relators[p] for p in perm]
        return q

    # -- Generator Nielsen moves --
    def map_gens(self, gen_map: dict[int, list[Letter]]) -> Presentation:
        q = self.copy()
        new_rels = []
        for r in q.relators:
            new_word: list[Letter] = []
            for a in r:
                if a > 0:
                    new_word.extend(gen_map.get(a, [a]))
                else:
                    mapped = gen_map.get(-a, [-a])
                    new_word.extend(invert(mapped))
            new_rels.append(free_reduce(new_word))
        q.relators = new_rels
        return q

    # -- Stabilization --
    def stab_add(self) -> Presentation:
        q = self.copy()
        q.n_gens += 1
        q.relators.append([q.n_gens])
        return q

    def stab_remove(self) -> Optional[Presentation]:
        """Remove generator if any relator is a single generator letter.

        When a relator is just [g], generator g is free.  Remove all
        occurrences of ±g from all relators, drop that relator, and
        renumber generators > g downwards.
        """
        if self.n_gens <= 1:
            return None
        # find first relator that is a single generator
        for idx, rel in enumerate(self.relators):
            if len(rel) == 1:
                g = abs(rel[0])
                if 1 <= g <= self.n_gens:
                    q = self.copy()
                    # remove all instances of ±g from every relator
                    new_rels: list[list[Letter]] = []
                    for j, r in enumerate(self.relators):
                        if j == idx:
                            continue
                        cleaned = [a for a in r if abs(a) != g]
                        # renumber generators > g
                        cleaned = [
                            (a - 1 if a > g else a) if a > 0
                            else (a + 1 if a < -g else a)
                            for a in cleaned
                        ]
                        new_rels.append(cleaned)
                    q.relators = new_rels
                    q.n_gens -= 1
                    return q
        return None

    # -- SUBSTITUTION SUPER-MOVE (correct cyclic definition from Caltech) --
    def substitution_move(
        self, i: int, j_sign: int, k1: int, k2: int
    ) -> Optional[Presentation]:
        """Apply a substitution super-move.

        Parameters (from Fagan+ 2026):
          i      : 0 or 1 — which relator to replace
          j_sign : +1 or -1 — whether to invert the *other* relator
          k1     : cyclic rotation of r_i
          k2     : cyclic rotation of r_{3-i}^j

        Validity condition:
          last(rot^{k1}(r_i)) = inverse(first(rot^{k2}(r_{other}^{j_sign})))

        The move replaces r_i with:
          rot^{k1}(r_i) · rot^{k2}(r_{3-i}^{j_sign})
        """
        n = self.n_gens
        if n != 2:
            return None

        other = 1 - i
        r_target = self.relators[i][:]
        r_source = self.relators[other][:]

        if j_sign == -1:
            r_source = invert(r_source)

        rot_target = cyclic_rotate(r_target, k1)
        rot_source = cyclic_rotate(r_source, k2)

        if not rot_target or not rot_source:
            return None

        # Validity: last of rot_target cancels with first of rot_source
        if rot_target[-1] != -rot_source[0]:
            return None

        new_ri = free_reduce(rot_target + rot_source)
        q = self.copy()
        q.relators[i] = new_ri
        return q

    def all_substitution_moves(self) -> list[tuple[str, Presentation]]:
        """Enumerate all valid substitution super-moves."""
        results: list[tuple[str, Presentation]] = []
        if self.n_gens != 2:
            return results

        for i in (0, 1):
            for j_sign in (1, -1):
                max_k1 = max(len(self.relators[i]), 1)
                max_k2 = max(len(self.relators[1 - i]), 1)
                for k1 in range(max_k1):
                    for k2 in range(max_k2):
                        result = self.substitution_move(i, j_sign, k1, k2)
                        if result is not None:
                            desc = (f"SUBST r{i}←r{1-i}"
                                    f"{'⁻¹' if j_sign==-1 else ''}"
                                    f" rot({k1},{k2})")
                            results.append((desc, result))
        return results

    # -- Normalization (cyclic canonical form) --
    def normalize(self) -> Presentation:
        """Canonical form using Booth's cyclic rotation + sorting.

        This gives a 1600x state-space reduction (Fagan+ 2026).
        """
        reduced = []
        for r in self.relators:
            cr = cyclic_canonical(free_reduce(r))
            reduced.append(cr)

        # Deduplicate and remove empty
        seen: set = set()
        unique: list[list[Letter]] = []
        for r in reduced:
            t = tuple(r)
            if t not in seen:
                seen.add(t)
                if t:
                    unique.append(list(r))
        unique.sort(key=tuple)
        while len(unique) < self.n_gens:
            unique.append([])
        return Presentation(self.n_gens, unique[:self.n_gens])

    def state_key(self) -> tuple:
        """Hashable key using canonical cyclic form."""
        reduced = tuple(
            sorted(tuple(cyclic_canonical(free_reduce(r))) for r in self.relators)
        )
        return (self.n_gens, reduced)

    def is_trivial(self) -> bool:
        return all(len(r) == 0 for r in self.relators)

    def is_standard_trivial(self) -> bool:
        if self.n_gens == 0:
            return True
        norm = self.normalize().relators
        expected = {tuple([i]) for i in range(1, self.n_gens + 1)}
        actual = {tuple(r) for r in norm}
        return actual == expected

    def is_solved(self) -> bool:
        return self.is_trivial() or self.is_standard_trivial()

    def show(self, letters: Optional[list[str]] = None) -> str:
        gens_str = " ".join(
            letters[i] if letters and i < len(letters) else f"x{i+1}"
            for i in range(self.n_gens)
        )
        rels_str = ", ".join(word_str(r, letters) for r in self.relators)
        return f"⟨ {gens_str}  |  {rels_str} ⟩"

    def __str__(self) -> str:
        return self.show()


# ═══════════════════════════════════════════════════════════════════════
# 3.  Move enumeration (12-rule variant + generator moves)
# ═══════════════════════════════════════════════════════════════════════

MoveRecord = tuple[str, Callable[[Presentation], Presentation]]


def enumerate_moves_12rule(p: Presentation) -> list[MoveRecord]:
    """The 12-rule action space from Shehper et al. / Lisitsa 2025.

    For 2-generator presentations, this gives exactly 12 moves:
    - AC'1: 4 multiply rules (r_i ← r_i · r_j^{±1} for i≠j)
    - AC'2: 8 conjugate rules (r_i ← g·r_i·g^{-1} for g ∈ {a,b,a⁻¹,b⁻¹})
    """
    n = p.n_gens
    moves: list[MoveRecord] = []

    # AC'1 — multiply (4 rules for 2 gens)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            moves.append((f"MUL r{i}·r{j}", lambda q, a=i, b=j: q.mul(a, b)))
            moves.append((f"MUL r{i}·r{j}⁻¹", lambda q, a=i, b=j: q.mul(a, b, True)))

    # AC'2 — conjugate by generators and inverses (8 rules for 2 gens)
    for i in range(n):
        for g in range(1, n + 1):
            moves.append((f"CONJ r{i} by x{g}", lambda q, idx=i, gen=g: q.conj(idx, gen)))
            moves.append((f"CONJ r{i} by x{g}⁻¹", lambda q, idx=i, gen=-g: q.conj(idx, -g)))

    return moves


def enumerate_moves_full(p: Presentation) -> list[MoveRecord]:
    """Full move set: 12-rule + permute + generator Nielsen + stabilization."""
    moves = enumerate_moves_12rule(p)
    n = p.n_gens

    # Permute
    for i in range(n - 1):
        perm = list(range(n))
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        moves.append((f"SWAP r{i}↔r{i+1}", lambda q, p_=perm: q.permute(p_)))

    # Generator Nielsen: invert generator
    for g in range(1, n + 1):
        gm = {a: [a] for a in range(1, n + 1)}
        gm[g] = [-g]
        moves.append((f"GENINV x{g}", lambda q, m=gm: q.map_gens(m)))

    # Generator Nielsen: multiply generators
    for g in range(1, n + 1):
        for h in range(1, n + 1):
            if g == h:
                continue
            gm = {a: [a] for a in range(1, n + 1)}
            gm[g] = [g, h]
            moves.append((f"GENMUL x{g}→x{g}x{h}", lambda q, m=gm: q.map_gens(m)))

    # Stabilization: add generator (AC3)
    moves.append((f"STAB_ADD", lambda q: q.stab_add()))

    # Stabilization: remove generator (AC3 inverse)
    sr = p.stab_remove()
    if sr is not None:
        moves.append((f"STAB_REMOVE", lambda q: q.stab_remove()))

    return moves


# ═══════════════════════════════════════════════════════════════════════
# 4.  Beam search with correct substitution super-moves
# ═══════════════════════════════════════════════════════════════════════

SearchResult = tuple[bool, Optional[list[tuple[str, Presentation]]], dict]


@dataclass
class ACBeamSearcherV3:
    """Beam search with correct substitution super-moves and cyclic
    canonical form (Booth)."""

    beam_width: int = 500
    max_depth: int = 200
    max_states: int = 10_000_000
    timeout: float = 7200.0
    use_substitution_moves: bool = True
    use_generator_moves: bool = True
    subst_slots: int = 0              # reserved slots for substitution children (0 = auto: beam_width//4)
    restart_patience: int = 20        # levels without progress before injecting diversity
    restart_fraction: float = 0.25    # fraction of beam replaced on restart
    verbose: bool = True
    results_dir: str = "results"

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)

    def _make_move_set(self, p: Presentation) -> list[MoveRecord]:
        """Build the move set for a presentation."""
        if self.use_generator_moves:
            return enumerate_moves_full(p)
        return enumerate_moves_12rule(p)

    def _save_checkpoint(self, run_id: str, data: dict):
        fname = f"{self.results_dir}/{run_id}_checkpoint.json"
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)

    def search(self, start: Presentation,
               run_id: Optional[str] = None) -> SearchResult:
        if run_id is None:
            run_id = f"run_{int(time.time())}"

        t0 = time.monotonic()

        if start.is_solved():
            return True, [], {
                "states_explored": 0, "depth": 0, "reason": "already trivial"
            }

        visited: dict = {start.state_key(): 0}
        total_states = 0
        best_len = start.total_length()
        best_cyclic = start.total_cyclic_length()
        depths_since_improvement = 0
        last_report = 0.0
        last_checkpoint = 0.0

        # frontier: list of (path, presentation) at current depth
        frontier: list[tuple[list[tuple[str, Presentation]], Presentation]] = [
            ([], start)
        ]

        subst_slots = self.subst_slots or max(self.beam_width // 4, 1)
        regular_slots = self.beam_width - subst_slots

        for depth in range(self.max_depth):
            if not frontier:
                break
            if time.monotonic() - t0 > self.timeout:
                break

            regular_candidates: list[tuple] = []
            subst_candidates: list[tuple] = []

            for path, pres in frontier:
                if time.monotonic() - t0 > self.timeout:
                    break

                # Standard moves
                moves = self._make_move_set(pres)

                for move_desc, move_fn in moves:
                    new_pres = move_fn(pres)
                    key = new_pres.state_key()

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
                        result = {
                            "found": True, "path_len": len(new_path),
                            "stats": {
                                "states_explored": total_states,
                                "depth": depth + 1,
                                "reason": "found",
                                "elapsed": elapsed,
                            }
                        }
                        self._save_checkpoint(run_id + "_FOUND", result)
                        return True, new_path, result["stats"]

                    tl = new_pres.total_length()
                    cl = new_pres.total_cyclic_length()
                    # score lower for fewer/lower generators in play
                    gen_span = 0
                    if self.use_generator_moves:
                        all_letters = set()
                        for r in new_pres.relators:
                            all_letters.update(abs(a) for a in r)
                        gen_span = len(all_letters) * 2
                    score = tl + cl + gen_span

                    regular_candidates.append(
                        (score, len(new_path), total_states,
                         new_path, new_pres))

                # Substitution super-moves (correct cyclic version)
                if self.use_substitution_moves:
                    for sub_desc, sub_res in pres.all_substitution_moves():
                        key = sub_res.state_key()
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
                            result = {
                                "found": True, "path_len": len(new_path),
                                "stats": {
                                    "states_explored": total_states,
                                    "depth": depth + 1,
                                    "reason": "found via super-move",
                                    "elapsed": elapsed,
                                }
                            }
                            self._save_checkpoint(run_id + "_FOUND", result)
                            return True, new_path, result["stats"]

                        tl = sub_res.total_length()
                        cl = sub_res.total_cyclic_length()
                        gen_span = 0
                        if self.use_generator_moves:
                            all_letters = set()
                            for r in sub_res.relators:
                                all_letters.update(abs(a) for a in r)
                            gen_span = len(all_letters) * 2
                        # substitution moves get no length penalty beyond
                        # the FIRST application, since they need room to grow
                        applied_sub = sum(1 for d, _ in new_path
                                          if d.startswith("SUBST"))
                        length_penalty = tl + cl if applied_sub <= 1 else 0
                        score = length_penalty + gen_span

                        subst_candidates.append(
                            (score, len(new_path), total_states,
                             new_path, sub_res))

            if total_states >= self.max_states:
                break

            # Two-track beam: keep best regular + best substitution
            regular_candidates.sort(key=lambda x: x[0])
            subst_candidates.sort(key=lambda x: x[0])

            frontier = (
                [(path, pres)
                 for _, _, _, path, pres
                 in regular_candidates[:regular_slots]]
                +
                [(path, pres)
                 for _, _, _, path, pres
                 in subst_candidates[:subst_slots]]
            )

            # Update best stats
            if frontier:
                best = frontier[0][1]
                bl = best.total_length()
                bc = best.total_cyclic_length()
                if bl < best_len:
                    best_len = bl
                    depths_since_improvement = 0
                if bc < best_cyclic:
                    best_cyclic = bc

            # Restart: inject diverse states when stalled
            depths_since_improvement += 1
            if (depths_since_improvement >= self.restart_patience
                    and len(frontier) >= 4):
                inject_n = max(1, int(len(frontier) * self.restart_fraction))
                # keep best inject_n states based on diversity of state_key
                all_cands = regular_candidates + subst_candidates
                # score by structural diversity (different norm lengths)
                seen_structures: set = set()
                diverse: list = []
                for _, _, _, path, pres in all_cands:
                    sig = tuple(sorted(len(r) for r in pres.relators))
                    if sig not in seen_structures:
                        seen_structures.add(sig)
                        diverse.append((path, pres))
                        if len(diverse) >= inject_n:
                            break
                # replace worst frontier entries with diverse states
                if diverse:
                    frontier = frontier[:len(frontier) - inject_n] + diverse
                depths_since_improvement = 0
                if self.verbose:
                    print(f"  ⚡ RESTART: injected {len(diverse)} diverse "
                          f"states", flush=True)

            # Reporting
            now = time.monotonic()
            if self.verbose and now - last_report > 5.0:
                print(
                    f"  [depth={depth+1:3d}  beam={len(frontier):4d}  "
                    f"explored={total_states:8d}  best_len={best_len:2d}  "
                    f"best_cyclic={best_cyclic:2d}  "
                    f"elapsed={now-t0:.0f}s]",
                    flush=True,
                )
                last_report = now

            # Checkpoint every 60s
            if now - last_checkpoint > 60.0:
                self._save_checkpoint(run_id, {
                    "found": False,
                    "depth": depth + 1,
                    "states_explored": total_states,
                    "best_len": best_len,
                    "best_cyclic": best_cyclic,
                    "elapsed": now - t0,
                })
                last_checkpoint = now

        elapsed = time.monotonic() - t0
        return False, None, {
            "states_explored": total_states,
            "depth": self.max_depth,
            "best_length": best_len,
            "best_cyclic": best_cyclic,
            "reason": "beam search exhausted",
            "elapsed": elapsed,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5.  Example presentations
# ═══════════════════════════════════════════════════════════════════════

def example_trivial() -> Presentation:
    return Presentation.trivial(2)

def example_myasnikov() -> Presentation:
    return Presentation.from_lists(2, [
        [1, 1, -2, -2, -2],
        [1, 2, 1, -2, -1, -2],
    ])

def example_ak3() -> Presentation:
    """AK(3) = ⟨ a,b | a³b⁴, abab⁻¹a⁻¹b⁻¹ ⟩  — OPEN"""
    return Presentation.from_lists(2, [
        [1, 1, 1, 2, 2, 2, 2],
        [1, 2, 1, -2, -1, -2],
    ])

def example_ak_general(n: int) -> Presentation:
    """AK(n) = ⟨ a,b | a^n b^{n+1}, abab⁻¹a⁻¹b⁻¹ ⟩"""
    return Presentation.from_lists(2, [
        [1] * n + [2] * (n + 1),
        [1, 2, 1, -2, -1, -2],
    ])

def example_presentation_p() -> Presentation:
    """Presentation P from Shehper et al. / Lisitsa 2025.

    ⟨a,b | a⁻¹b⁻¹ab⁻¹a⁻¹bab⁻²aba⁻¹b, b⁻¹a⁻¹b²a⁻¹b⁻¹abab⁻²a⟩
    This is the bridge presentation used in the stable AK(3) approach.
    Total length ≈ 46.
    """
    return Presentation.from_lists(2, [
        [-1, -2, 1, -2, -1, 2, 1, -2, -2, 1, 2, 1, -1, 2],
        [-2, -1, 2, 2, -1, -2, -1, 2, 1, 2, -2, -2, 1],
    ])

def example_ms_equivalent_ak3() -> Presentation:
    """MS(n, y⁻¹x⁻¹yxy) for n=3 — AC-equivalent to AK(3).

    From Myasnikov et al. 2002: AK(n) ≅ MS(n, y^{-1}x^{-1}yxy).
    MS(n,w) = ⟨ x,y | x^{-1}y^{n}x = y^{n+1}, x = w ⟩
    """
    return Presentation.from_lists(2, [
        [-1, 2, 2, 2, 1, -2, -2, -2, -2],
        [-1, -2, -1, 2, 1, 2],
    ])


# ═══════════════════════════════════════════════════════════════════════
# 6.  Command line
# ═══════════════════════════════════════════════════════════════════════

EXAMPLES: dict[str, tuple[Presentation, str]] = {
    "trivial": (example_trivial(), "Standard trivial"),
    "myasnikov": (example_myasnikov(), "Known AC-trivializable"),
    "ak3": (example_ak3(), "AK(3) — OPEN (primary target)"),
    "ak4": (example_ak_general(4), "AK(4) — OPEN"),
    "ak5": (example_ak_general(5), "AK(5) — OPEN"),
    "presentation_p": (example_presentation_p(), "Bridge presentation P (Lisitsa)"),
    "ms_ak3": (example_ms_equivalent_ak3(), "MS(3, y⁻¹x⁻¹yxy) ≅ AK(3)"),
}

ALL_NAMES = list(EXAMPLES)


def safe_input(prompt: str, default: str = "") -> str:
    try:
        r = input(prompt).strip()
        return r if r else default
    except EOFError:
        return default


def main() -> None:
    print("═" * 70)
    print("  Andrews–Curtis Conjecture — V3 Corrected Solver")
    print("  Cyclic super-moves + Booth canonical form + 12-rule")
    print("═" * 70)
    print()

    for i, name in enumerate(ALL_NAMES):
        pres, desc = EXAMPLES[name]
        ltrs = ["a", "b"] if pres.n_gens <= 2 else None
        print(f"  [{i:1d}] {name:18s}  — {desc}")
        print(f"      {pres.show(ltrs)}  (len={pres.total_length()})")
        print()

    raw = safe_input("  Select example [0-6, default=2 (ak3)]: ", "2")
    try:
        name = ALL_NAMES[int(raw)]
    except (ValueError, IndexError):
        name = "ak3"

    pres, desc = EXAMPLES[name]
    print(f"\n  Target: {name} — {desc}")
    print(f"  Start:  {pres}")
    print()

    beam_w = int(safe_input("  Beam width [500]: ", "500"))
    max_d = int(safe_input("  Max depth [150]: ", "150"))
    max_s = int(safe_input("  Max states [8000000]: ", "8000000"))
    use_sub = safe_input("  Use substitution super-moves? [Y/n]: ", "y").lower()
    use_sub_bool = use_sub not in ("n", "no")
    use_gen = safe_input("  Use generator Nielsen moves? [Y/n]: ", "y").lower()
    use_gen_bool = use_gen not in ("n", "no")
    to = float(safe_input("  Timeout (s) [3600]: ", "3600"))

    searcher = ACBeamSearcherV3(
        beam_width=beam_w,
        max_depth=max_d,
        max_states=max_s,
        timeout=to,
        use_substitution_moves=use_sub_bool,
        use_generator_moves=use_gen_bool,
        verbose=True,
    )

    print(f"\n  Running search (sub={use_sub_bool}, gen={use_gen_bool})...\n")
    t0 = time.monotonic()
    found, path, stats = searcher.search(pres, run_id=f"{name}_v3_{int(t0)}")
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
                print(f"    [{step+1:3d}] {desc:35s}  {state.show(ltrs)}  (len={tl})")
    else:
        print(f"  ✗  NOT FOUND within search limits.")
        print(f"     Reason: {stats.get('reason', 'unknown')}")

    print(f"\n  Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")

    # Save final result
    result_data = {
        "run_id": f"{name}_v3_{int(t0)}",
        "found": found,
        "presentation": name,
        "description": desc,
        "params": {
            "beam_width": beam_w,
            "max_depth": max_d,
            "max_states": max_s,
            "use_substitution": use_sub_bool,
            "use_generator_moves": use_gen_bool,
            "timeout": to,
        },
        "stats": {k: (v if isinstance(v, (int, float, str)) else str(v))
                  for k, v in stats.items()},
        "elapsed": elapsed,
    }
    if path:
        ltrs = ["a", "b"] if pres.n_gens <= 2 else None
        result_data["path"] = [
            {"move": d, "state": s.show(ltrs), "len": s.total_length()}
            for d, s in path
        ]

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    fname = f"{results_dir}/{name}_v3_{int(t0)}_{'FOUND' if found else 'NOT_FOUND'}.json"
    with open(fname, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  Full results saved to: {fname}")
    print()


if __name__ == "__main__":
    main()
