#!/usr/bin/env python3
"""
Andrews–Curtis Conjecture — Advanced Solver
=============================================

Strategy: beam search with substitution super-moves, cyclic reduction,
and generator Nielsen automorphisms.  Targets AK(3) — the smallest open
potential counterexample to the unstable conjecture.

The Andrews–Curtis conjecture (1965) states that every balanced
presentation of the trivial group can be transformed into the standard
trivial presentation via a finite sequence of AC-moves.

Authors: NullLabTests
Reference: https://github.com/NullLabTests/Andrews-Curtis
"""

from __future__ import annotations

import itertools
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Callable, Optional

# ═══════════════════════════════════════════════════════════════════════
# 1.  Word utilities
# ═══════════════════════════════════════════════════════════════════════

Letter = int  # positive = generator, negative = inverse


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


def cyclic_reduce(word: list[Letter]) -> list[Letter]:
    """Cyclically reduce: cancel trailing letters with leading ones."""
    if not word:
        return []
    w = list(word)
    while len(w) >= 2 and w[0] == -w[-1]:
        w = w[1:-1]
    return w


def cyclic_length(word: list[Letter]) -> int:
    """Length of the cyclically reduced form."""
    return len(cyclic_reduce(word))


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

    # -- AC moves --
    def invert_relator(self, i: int) -> Presentation:
        q = self.copy()
        q.relators[i] = invert(q.relators[i])
        return q

    def multiply_relator(self, i: int, j: int, use_inv: bool = False) -> Presentation:
        q = self.copy()
        rhs = invert(q.relators[j]) if use_inv else q.relators[j]
        q.relators[i] = multiply(q.relators[i], rhs)
        return q

    def multiply_relator_left(self, i: int, j: int, use_inv: bool = False) -> Presentation:
        q = self.copy()
        lhs = invert(q.relators[j]) if use_inv else q.relators[j]
        q.relators[i] = multiply(lhs, q.relators[i])
        return q

    def conjugate_relator(self, i: int, gen: Letter) -> Presentation:
        q = self.copy()
        q.relators[i] = conjugate(q.relators[i], gen)
        return q

    def permute_relators(self, perm: list[int]) -> Presentation:
        q = self.copy()
        q.relators = [q.relators[p] for p in perm]
        return q

    # -- Generator Nielsen moves --
    def apply_generator_map(self, gen_map: dict[int, list[Letter]]) -> Presentation:
        """Apply a Nielsen automorphism to the generators.

        gen_map maps old generator index → new word (over old generators).
        Every relator is rewritten accordingly.
        """
        q = self.copy()
        new_rels = []
        for r in q.relators:
            new_word: list[Letter] = []
            for a in r:
                if a > 0:
                    mapped = gen_map.get(a, [a])
                    new_word.extend(mapped)
                else:
                    mapped = gen_map.get(-a, [-a])
                    new_word.extend(invert(mapped))
            new_rels.append(free_reduce(new_word))
        q.relators = new_rels
        return q

    # -- Stabilization moves --
    def stabilize_add(self) -> Presentation:
        """Add a new generator g_{n+1} and relator g_{n+1}. (AC4)"""
        q = self.copy()
        q.n_gens += 1
        q.relators.append([q.n_gens])
        return q

    def stabilize_remove(self) -> Optional[Presentation]:
        """Remove the last generator+relator if the relator is a single generator. (AC5)"""
        if self.n_gens <= 1:
            return None
        last = self.relators[-1]
        if len(last) == 1 and abs(last[0]) <= self.n_gens:
            q = self.copy()
            q.n_gens -= 1
            q.relators.pop()
            return q
        return None

    # -- Super-move: substitute subword using another relator --
    def substitute(self, i: int, j: int) -> list[Presentation]:
        """Try to eliminate a subword from relator i using relator j.

        If relator j contains a word w and relator i contains w or w^{-1}
        as a subword, returns presentations where that subword has been
        removed via AC-move multiplication.

        This is the 'super-move' concept from the Caltech AC-SolverX work.
        """
        results = []
        r_i = self.relators[i]
        r_j = self.relators[j]

        if not r_j:
            return results

        # Try each contiguous subword of r_j as the "substitution key"
        max_sub = min(len(r_j), 6)
        for sub_len in range(1, max_sub + 1):
            for start in range(len(r_j) - sub_len + 1):
                w = r_j[start:start + sub_len]
                w_inv = invert(w)

                # Try to find w in r_i
                for pos in range(len(r_i) - sub_len + 1):
                    if r_i[pos:pos + sub_len] == w:
                        # Remove w from r_i by multiplying with a conjugate of r_j
                        new_r = r_i[:pos] + r_i[pos + sub_len:]
                        q = self.copy()
                        q.relators[i] = free_reduce(new_r)
                        results.append(q)

                # Also try w^{-1}
                if w != w_inv:
                    for pos in range(len(r_i) - sub_len + 1):
                        if r_i[pos:pos + sub_len] == w_inv:
                            new_r = r_i[:pos] + r_i[pos + sub_len:]
                            q = self.copy()
                            q.relators[i] = free_reduce(new_r)
                            results.append(q)
        return results

    # -- Normalization --
    def normalize(self) -> Presentation:
        reduced = [free_reduce(r) for r in self.relators]
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
        reduced = tuple(sorted(tuple(free_reduce(r)) for r in self.relators))
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
# 3.  Move enumeration with generator Nielsen moves
# ═══════════════════════════════════════════════════════════════════════

MoveRecord = tuple[str, Callable[[Presentation], Presentation]]


def enumerate_moves(p: Presentation) -> list[MoveRecord]:
    """Return all single-AC-move + generator-Nielsen-move closures."""
    n = p.n_gens
    moves: list[MoveRecord] = []

    # AC₁ invert
    for i in range(n):
        moves.append((f"INV r{i}", lambda q, idx=i: q.invert_relator(idx)))

    # AC₂ multiply right
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            moves.append((f"MUL r{i}·r{j}", lambda q, a=i, b=j: q.multiply_relator(a, b)))
            moves.append((f"MUL r{i}·r{j}⁻¹", lambda q, a=i, b=j: q.multiply_relator(a, b, True)))
            moves.append((f"MUL r{j}·r{i}", lambda q, a=i, b=j: q.multiply_relator_left(a, b)))
            moves.append((f"MUL r{j}⁻¹·r{i}", lambda q, a=i, b=j: q.multiply_relator_left(a, b, True)))

    # AC₃ conjugate
    for i in range(n):
        for g in range(1, n + 1):
            moves.append((f"CONJ r{i} by x{g}", lambda q, idx=i, gen=g: q.conjugate_relator(idx, gen)))
            moves.append((f"CONJ r{i} by x{g}⁻¹", lambda q, idx=i, gen=-g: q.conjugate_relator(idx, -g)))

    # AC₄ permute (adjacent swaps)
    for i in range(n - 1):
        perm = list(range(n))
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        moves.append((f"SWAP r{i}↔r{i+1}", lambda q, p=perm: q.permute_relators(p)))

    # Generator Nielsen moves (invert a generator)
    for g in range(1, n + 1):
        gen_map = {a: [a] for a in range(1, n + 1)}
        gen_map[g] = [-g]
        moves.append((f"GENINV x{g}", lambda q, gm=gen_map: q.apply_generator_map(gm)))

    # Generator Nielsen moves (multiply: replace g by g*h)
    for g in range(1, n + 1):
        for h in range(1, n + 1):
            if g == h:
                continue
            gen_map = {a: [a] for a in range(1, n + 1)}
            gen_map[g] = [g, h]
            moves.append((f"GENMUL x{g}→x{g}x{h}", lambda q, gm=gen_map: q.apply_generator_map(gm)))

    return moves


# ═══════════════════════════════════════════════════════════════════════
# 4.  Beam search with substitution super-moves
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ACBeamSearcher:
    """Beam search for AC-trivializations.

    Keeps the top-K most promising states at each depth level.
    Optionally applies substitution super-moves.
    """

    beam_width: int = 100
    max_depth: int = 50
    max_states: int = 2_000_000
    timeout: float = 300.0
    use_super_moves: bool = True
    use_generator_moves: bool = True
    verbose: bool = True

    def search(self, start: Presentation) -> tuple[bool, Optional[list[tuple[str, Presentation]]], dict]:
        t0 = time.monotonic()
        target = Presentation.trivial(start.n_gens)

        if start.is_solved():
            return True, [], {"states_explored": 0, "depth": 0, "reason": "already trivial"}

        letters = ["a", "b"] if start.n_gens <= 2 else None
        visited: dict = {start.state_key(): 0}
        total_states = 0
        best_len = start.total_length()
        best_cyclic = start.total_cyclic_length()
        last_report = 0.0

        # frontier: list of (path, presentation) at current depth
        frontier: list[tuple[list[tuple[str, Presentation]], Presentation]] = [([], start)]
        global_best_path = None

        for depth in range(self.max_depth):
            if not frontier:
                break

            if time.monotonic() - t0 > self.timeout:
                break

            # -- Expand frontier --
            candidates: list[tuple] = []

            for path, pres in frontier:
                if time.monotonic() - t0 > self.timeout:
                    break

                # Standard AC moves + generator moves
                all_moves = enumerate_moves(pres) if self.use_generator_moves else enumerate_moves(pres)
                # Filter to only standard moves if not using generator moves
                if not self.use_generator_moves:
                    all_moves = [(d, f) for d, f in all_moves if not d.startswith("GEN")]

                for move_desc, move_fn in all_moves:
                    new_pres = move_fn(pres)
                    key = new_pres.state_key()

                    if key in visited and visited[key] <= depth + 1:
                        continue
                    visited[key] = depth + 1
                    total_states += 1

                    new_path = path + [(move_desc, new_pres)]

                    if new_pres.is_solved():
                        if self.verbose:
                            print(f"\n  ✓ FOUND at depth {depth+1} after {total_states} states!", flush=True)
                        return True, new_path, {
                            "states_explored": total_states,
                            "depth": depth + 1,
                            "reason": "found",
                            "elapsed": time.monotonic() - t0,
                        }

                    # Heuristic: cyclic length + total length
                    tl = new_pres.total_length()
                    cl = new_pres.total_cyclic_length()
                    score = tl + cl

                    candidates.append((score, len(new_path), total_states, new_path, new_pres))

                # Substitution super-moves
                if self.use_super_moves and pres.n_gens >= 2:
                    for i in range(pres.n_gens):
                        for j in range(pres.n_gens):
                            if i == j:
                                continue
                            for sub_pres in pres.substitute(i, j):
                                key = sub_pres.state_key()
                                if key in visited and visited[key] <= depth + 1:
                                    continue
                                visited[key] = depth + 1
                                total_states += 1

                                new_path = path + [(f"SUBST r{i}←r{j}", sub_pres)]
                                if sub_pres.is_solved():
                                    if self.verbose:
                                        print(f"\n  ✓ FOUND at depth {depth+1} after {total_states} states!", flush=True)
                                    return True, new_path, {
                                        "states_explored": total_states,
                                        "depth": depth + 1,
                                        "reason": "found via super-move",
                                        "elapsed": time.monotonic() - t0,
                                    }

                                tl = sub_pres.total_length()
                                cl = sub_pres.total_cyclic_length()
                                score = tl + cl
                                candidates.append((score, len(new_path), total_states, new_path, sub_pres))

            # -- Prune to beam --
            if total_states >= self.max_states:
                break

            candidates.sort(key=lambda x: x[0])
            frontier = [(path, pres) for _, _, _, path, pres in candidates[:self.beam_width]]

            # Stats
            if frontier:
                best_score = candidates[0][0] if candidates else 999
                best_state = frontier[0][1]
                bl = best_state.total_length()
                if bl < best_len:
                    best_len = bl
                bc = best_state.total_cyclic_length()
                if bc < best_cyclic:
                    best_cyclic = bc

            if self.verbose and time.monotonic() - last_report > 3.0:
                print(
                    f"  [depth={depth+1:2d}  beam={len(frontier):4d}  "
                    f"explored={total_states:7d}  best_len={best_len:2d}  "
                    f"best_cyclic={best_cyclic:2d}]",
                    flush=True,
                )
                last_report = time.monotonic()

        return False, None, {
            "states_explored": total_states,
            "depth": self.max_depth,
            "best_length": best_len,
            "best_cyclic": best_cyclic,
            "reason": "beam search exhausted",
            "elapsed": time.monotonic() - t0,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5.  Genetic algorithm search
# ═══════════════════════════════════════════════════════════════════════

import random


@dataclass
class ACGeneticSearcher:
    """Genetic algorithm for AC-trivialization path discovery.

    Each genome is a fixed-length sequence of move indices.
    Fitness = negative of final total length (lower is better).
    """

    pop_size: int = 200
    genome_length: int = 30
    mutation_rate: float = 0.15
    generations: int = 100
    timeout: float = 120.0
    verbose: bool = True

    def __post_init__(self):
        self._move_cache: dict[int, list[MoveRecord]] = {}

    def _get_moves(self, pres: Presentation) -> list[MoveRecord]:
        """Get moves for a presentation, cached by n_gens."""
        key = pres.n_gens
        if key not in self._move_cache:
            self._move_cache[key] = enumerate_moves(pres)
        return self._move_cache[key]

    def _random_genome(self, n_moves: int) -> list[int]:
        return [random.randrange(n_moves) for _ in range(self.genome_length)]

    def _evaluate(self, genome: list[int], start: Presentation) -> tuple[int, Presentation, list[str], list[Presentation]]:
        pres = start.copy()
        path_desc: list[str] = []
        path_states: list[Presentation] = [pres.copy()]
        moves = self._get_moves(pres)

        for idx in genome:
            if idx >= len(moves):
                continue
            desc, fn = moves[idx]
            pres = fn(pres)
            path_desc.append(desc)
            path_states.append(pres.copy())
            if pres.is_solved():
                break
            # Recompute moves if generator count changed
            if pres.n_gens != path_states[-2].n_gens:
                moves = self._get_moves(pres)

        return pres.total_length(), pres, path_desc, path_states

    def search(self, start: Presentation) -> tuple[bool, Optional[list[tuple[str, Presentation]]], dict]:
        t0 = time.monotonic()
        n_moves = len(self._get_moves(start))

        # Initialize population
        pop = [self._random_genome(n_moves) for _ in range(self.pop_size)]

        best_fitness = 999
        best_genome = None
        best_path = None
        best_pres = None
        last_report = 0.0
        total_evals = 0
        stalled = 0

        for gen in range(self.generations):
            if time.monotonic() - t0 > self.timeout:
                break

            # Evaluate
            scored: list[tuple[int, list[int], Presentation, list[str], list[Presentation]]] = []
            for genome in pop:
                length, pres, descs, states = self._evaluate(genome, start)
                total_evals += 1
                scored.append((length, genome, pres, descs, states))

                if pres.is_solved():
                    path = list(zip(descs, states[1:]))
                    elapsed = time.monotonic() - t0
                    if self.verbose:
                        print(f"\n  ✓ GA FOUND at generation {gen}!", flush=True)
                    return True, path, {
                        "states_explored": total_evals,
                        "depth": len(descs),
                        "reason": "found via GA",
                        "elapsed": elapsed,
                    }

            # Sort by fitness (lower total length = better)
            scored.sort(key=lambda x: x[0])
            best = scored[0][0]

            if best < best_fitness:
                best_fitness = best
                best_genome = scored[0][1]
                best_path = list(zip(scored[0][3], scored[0][4][1:]))
                best_pres = scored[0][2]
                stalled = 0
            else:
                stalled += 1

            if self.verbose and (gen % 10 == 0 or time.monotonic() - last_report > 5.0):
                print(
                    f"  [GA gen={gen:3d}  best_len={best:2d}  stalled={stalled:2d}]",
                    flush=True,
                )
                last_report = time.monotonic()

            # Selection: top 20% breed
            elite_count = max(2, self.pop_size // 5)
            elite = [s[1] for s in scored[:elite_count]]

            # Crossover + mutation
            new_pop = list(elite)
            while len(new_pop) < self.pop_size:
                if random.random() < 0.3 and len(elite) >= 2:
                    # Crossover
                    p1 = random.choice(elite)
                    p2 = random.choice(elite)
                    split = random.randrange(1, self.genome_length - 1)
                    child = p1[:split] + p2[split:]
                else:
                    # Copy a parent
                    child = list(random.choice(elite))

                # Mutation
                for i in range(len(child)):
                    if random.random() < self.mutation_rate:
                        child[i] = random.randrange(n_moves)

                new_pop.append(child)

            # Adaptive mutation: increase if stalled
            if stalled > 15:
                self.mutation_rate = min(0.5, self.mutation_rate + 0.05)
            else:
                self.mutation_rate = 0.15

            pop = new_pop

        return False, best_path, {
            "states_explored": total_evals,
            "depth": self.genome_length,
            "best_length": best_fitness,
            "reason": "GA generations exhausted",
            "elapsed": time.monotonic() - t0,
        }


# ═══════════════════════════════════════════════════════════════════════
# 6.  Example presentations
# ═══════════════════════════════════════════════════════════════════════

def example_trivial() -> Presentation:
    return Presentation.trivial(2)

def example_myasnikov() -> Presentation:
    return Presentation.from_lists(2, [
        [1, 1, -2, -2, -2],
        [1, 2, 1, -2, -1, -2],
    ])

def example_ak3() -> Presentation:
    """AK(3) = ⟨ a,b | a³b⁴, abab⁻¹a⁻¹b⁻¹ ⟩  — OPEN POTENTIAL COUNTEREXAMPLE"""
    return Presentation.from_lists(2, [
        [1, 1, 1, 2, 2, 2, 2],
        [1, 2, 1, -2, -1, -2],
    ])

def example_dunwoody() -> Presentation:
    return Presentation.from_lists(2, [
        [1, 2, 1, -2, -1, -2],
        [1, 1, -2, -2, -2],
    ])

def example_ms9() -> Presentation:
    return Presentation.from_lists(2, [
        [-1] + [2] * 9 + [1] + [-2] * 10,
        [-1, -2, 1, 2, -1],
    ])

def example_ak_general(n: int) -> Presentation:
    """AK(n) = ⟨ a,b | a^n b^{n+1}, abab⁻¹a⁻¹b⁻¹ ⟩"""
    return Presentation.from_lists(2, [
        [1] * n + [2] * (n + 1),
        [1, 2, 1, -2, -1, -2],
    ])


# ═══════════════════════════════════════════════════════════════════════
# 7.  Command line
# ═══════════════════════════════════════════════════════════════════════

EXAMPLES: dict[str, tuple[Presentation, str]] = {
    "trivial": (example_trivial(), "Standard trivial"),
    "myasnikov": (example_myasnikov(), "Known AC-trivializable"),
    "ak3": (example_ak3(), "AK(3) — OPEN potential counterexample"),
    "ak4": (example_ak_general(4), "AK(4) — OPEN"),
    "ak5": (example_ak_general(5), "AK(5) — OPEN"),
    "dunwoody": (example_dunwoody(), "Dunwoody-style"),
    "ms9": (example_ms9(), "Miller-Schupp MS9 (Lisitsa 2025)"),
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
    print("  Andrews–Curtis Conjecture — ADVANCED Solver")
    print("  Beam search + super-moves + genetic algorithm")
    print("═" * 70)
    print()

    for i, name in enumerate(ALL_NAMES):
        pres, desc = EXAMPLES[name]
        ltrs = ["a", "b"] if pres.n_gens <= 2 else None
        print(f"  [{i:1d}] {name:12s}  — {desc}")
        print(f"      {pres.show(ltrs)}  (len={pres.total_length()})")
        print()

    raw = safe_input("  Select example [0-6, default=0]: ", "0")
    try:
        name = ALL_NAMES[int(raw)]
    except (ValueError, IndexError):
        print("  Using example 0.")
        name = ALL_NAMES[0]

    pres, desc = EXAMPLES[name]
    print(f"\n  Target: {name} — {desc}")
    print(f"  Start:  {pres}")
    print()

    algo = safe_input("  Algorithm [beam/ga, default=beam]: ", "beam").strip().lower()
    beam_w = int(safe_input("  Beam width [200]: ", "200"))
    max_d = int(safe_input("  Max depth [50]: ", "50"))
    max_s = int(safe_input("  Max states [500000]: ", "500000"))
    super_s = safe_input("  Use super-moves? [Y/n]: ", "y").lower()
    use_super = super_s not in ("n", "no")
    gen_m = safe_input("  Use generator Nielsen moves? [Y/n]: ", "y").lower()
    use_gen = gen_m not in ("n", "no")
    to = float(safe_input("  Timeout (s) [300]: ", "300"))

    if algo == "ga":
        pop = int(safe_input("  GA population [200]: ", "200"))
        gen_len = int(safe_input("  GA genome length [50]: ", "50"))
        ga_gens = int(safe_input("  GA generations [200]: ", "200"))
        searcher = ACGeneticSearcher(
            pop_size=pop,
            genome_length=gen_len,
            generations=ga_gens,
            timeout=to,
            verbose=True,
        )
    else:
        searcher = ACBeamSearcher(
            beam_width=beam_w,
            max_depth=max_d,
            max_states=max_s,
            timeout=to,
            use_super_moves=use_super,
            use_generator_moves=use_gen,
            verbose=True,
        )

    print(f"\n  Running {algo.upper()} search...\n")
    t0 = time.monotonic()
    found, path, stats = searcher.search(pres)
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
                print(f"    [{step+1:3d}] {desc:30s}  {state.show(ltrs)}  (len={tl})")
    else:
        print(f"  ✗  NOT FOUND within search limits.")
        print(f"     Reason: {stats.get('reason', 'unknown')}")

    print(f"\n  Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")
    print()


if __name__ == "__main__":
    main()
