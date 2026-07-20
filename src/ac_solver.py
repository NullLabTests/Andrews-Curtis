#!/usr/bin/env python3
"""
Andrews–Curtis Conjecture — Computational Searcher
=====================================================================

The Andrews–Curtis conjecture (1965) states that every balanced
presentation of the trivial group can be transformed into the standard
trivial presentation via a finite sequence of AC-moves.

A presentation is balanced if it has the same number of generators as
relators.

AC-moves (on the relators, with generators fixed):
  1. Invert a relator:           r_i → r_i^{-1}
  2. Multiply one relator by another:  r_i → r_i * r_j  (or r_i * r_j^{-1})
  3. Conjugate a relator by a generator:  r_i → g_k * r_i * g_k^{-1}
     (for any generator g_k, or its inverse)
  4. Permute the relators  (optional but recommended)

The goal is to reduce all relators to the empty word.

References
----------
[AC65]  Andrews & Curtis, Proc. Amer. Math. Soc. 16 (1965) 192-195.
[AC66]  Andrews & Curtis, Amer. Math. Monthly 73 (1966) 21-28.
[HM93]  Hog-Angeloni & Metzler, LMS Lecture Note Ser. 197 (1993) 365-380.
[B15]   Bridson, arXiv:1504.04187 (2015).
[L25]   Lisitsa, Examples and Counterexamples (2025).

Author : NullLabTests
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Callable, Optional


# ═══════════════════════════════════════════════════════════════════════
# 1.  Free-group word utilities
# ═══════════════════════════════════════════════════════════════════════

Letter = int  # positive = generator index; negative = its inverse


def free_reduce(word: list[Letter]) -> list[Letter]:
    """Fully cancel adjacent inverse pairs (free reduction).

    Uses a stack: when the top of stack equals the negation of the
    next letter, the pair cancels.  O(n) time, O(n) space.

    Examples
    --------
    >>> free_reduce([1, -1])
    []
    >>> free_reduce([1, 2, -2, 1])
    [1, 1]
    """
    stack: list[Letter] = []
    for a in word:
        if stack and stack[-1] == -a:
            stack.pop()
        else:
            stack.append(a)
    return stack


def multiply(w1: list[Letter], w2: list[Letter]) -> list[Letter]:
    """Concatenate w1 and w2, then free-reduce."""
    return free_reduce(w1 + w2)


def invert(word: list[Letter]) -> list[Letter]:
    """Reverse and negate every letter.

    For a word  g1 g2 ... gk, returns  gk^{-1} ... g2^{-1} g1^{-1}.
    This satisfies  invert(invert(w)) == w  after free reduction.
    """
    return [-a for a in reversed(word)]


def conjugate(word: list[Letter], gen: Letter) -> list[Letter]:
    """Return  gen · word · gen^{-1}   after free reduction.

    Parameters
    ----------
    word : list[Letter]
        The word to conjugate.
    gen : Letter
        A single generator (positive) or its inverse (negative).
    """
    return free_reduce([gen] + word + [-gen])


def word_len(word: list[Letter]) -> int:
    """Return the length of a word (number of letters)."""
    return len(word)


def word_str(word: list[Letter], letters: Optional[list[str]] = None) -> str:
    """Pretty-print a word.

    Parameters
    ----------
    word : list[Letter]
        Integer representation of a free-group word.
    letters : list[str] or None
        If provided, maps generator index i to letters[i-1].
        Otherwise uses ``x1, x2, …``.

    Returns
    -------
    str
        Human-readable representation like ``a·b·a⁻¹`` (short words)
        or ``aba⁻¹`` (long words).
    """
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
# 2.  Presentation class
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Presentation:
    """A balanced group presentation  ⟨ g₁ … gₙ | r₁ … rₙ ⟩.

    Fields
    ------
    n_gens : int
        Number of generators (and relators).
    relators : list[list[Letter]]
        One freely-reduced list of integers per relator.
        Relators MAY be empty (the identity word).

    The presentation is *balanced* iff n_gens == len(relators).
    """

    n_gens: int
    relators: list[list[Letter]]

    def __post_init__(self) -> None:
        assert len(self.relators) == self.n_gens, (
            f"Balanced required: {self.n_gens} generators, "
            f"{len(self.relators)} relators"
        )

    # -- Convenience constructors --------------------------------------------

    @staticmethod
    def trivial(n: int = 2) -> Presentation:
        """Return the standard trivial presentation  ⟨ x₁…xₙ | x₁…xₙ ⟩."""
        return Presentation(n, [[i] for i in range(1, n + 1)])

    @staticmethod
    def from_lists(n_gens: int, relator_lists: list[list[int]]) -> Presentation:
        """Build from explicit integer lists.  Each list is free-reduced."""
        assert len(relator_lists) == n_gens
        return Presentation(n_gens, [free_reduce(r) for r in relator_lists])

    # -- Accessors ------------------------------------------------------------

    def copy(self) -> Presentation:
        """Return an independent shallow copy of the relator list-of-lists."""
        return Presentation(self.n_gens, [list(r) for r in self.relators])

    def total_length(self) -> int:
        """Sum of all relator word lengths (0 = all relators empty)."""
        return sum(len(r) for r in self.relators)

    def max_length(self) -> int:
        """Length of the longest individual relator."""
        return max((len(r) for r in self.relators), default=0)

    # -- Core AC-move application (each returns a NEW Presentation) ---------

    def invert_relator(self, i: int) -> Presentation:
        """AC₁: replace r_i with r_i^{-1}."""
        q = self.copy()
        q.relators[i] = invert(q.relators[i])
        return q

    def multiply_relator(self, i: int, j: int,
                         use_inverse: bool = False) -> Presentation:
        """AC₂: replace r_i with r_i * r_j  (or r_i * r_j^{-1})."""
        q = self.copy()
        rhs = invert(q.relators[j]) if use_inverse else q.relators[j]
        q.relators[i] = multiply(q.relators[i], rhs)
        return q

    def multiply_relator_left(self, i: int, j: int,
                              use_inverse: bool = False) -> Presentation:
        """AC₂ (left variant): replace r_i with r_j * r_i (or r_j^{-1} * r_i)."""
        q = self.copy()
        lhs = invert(q.relators[j]) if use_inverse else q.relators[j]
        q.relators[i] = multiply(lhs, q.relators[i])
        return q

    def conjugate_relator(self, i: int, gen: Letter) -> Presentation:
        """AC₃: replace r_i with gen · r_i · gen^{-1}."""
        q = self.copy()
        q.relators[i] = conjugate(q.relators[i], gen)
        return q

    def permute_relators(self, perm: list[int]) -> Presentation:
        """AC₄: reorder relators by the given permutation list."""
        q = self.copy()
        q.relators = [q.relators[p] for p in perm]
        return q

    # -- Normalisation / canonical form -------------------------------------

    def normalize(self) -> Presentation:
        """Return a canonical form for equality testing.

        Steps:
          1. Free-reduce every relator.
          2. Remove empty relators and deduplicate identical relators.
          3. Sort remaining relators lexicographically.
          4. Pad with empty relators if needed to keep the presentation
             balanced (same number of relators as generators).

        This reduces the search space by collapsing equivalent states.
        """
        # free-reduce
        reduced = [free_reduce(r) for r in self.relators]
        # deduplicate and remove empty
        seen: set[tuple[Letter, ...]] = set()
        unique: list[list[Letter]] = []
        for r in reduced:
            t = tuple(r)
            if t not in seen:
                seen.add(t)
                if t:
                    unique.append(list(r))
        unique.sort(key=tuple)
        # pad with empty relators to maintain balanced count
        while len(unique) < self.n_gens:
            unique.append([])
        return Presentation(self.n_gens, unique[:self.n_gens])

    def state_key(self) -> tuple:
        """Immutable hashable key for visited-set membership.

        Returns a tuple ``(n_gens, sorted_tuple_of_relator_tuples)``.
        Two presentations that are AC-equivalent up to trivial relabeling
        will have the same state key.
        """
        reduced = tuple(
            sorted(tuple(free_reduce(r)) for r in self.relators)
        )
        return (self.n_gens, reduced)

    def is_trivial(self) -> bool:
        """Check whether all relators reduce to the empty word.

        The goal of the AC conjecture is to reach a presentation where
        every relator is the identity (empty word).  If all relators are
        empty, the group presented is the free group of rank n_gens.
        For a presentation known to define the trivial group, reaching
        all-empty relators indicates AC-trivializability.

        NOTE: Some formulations use  ⟨ x₁…xₙ | x₁…xₙ ⟩  as the target.
        The two formulations are equivalent up to AC-moves, but checking
        for empty relators is the more natural target for the search.
        """
        return all(len(r) == 0 for r in self.relators)

    def is_standard_trivial(self) -> bool:
        """Check if this IS the standard trivial presentation
        ⟨ x₁…xₙ | x₁…xₙ ⟩  where each relator is a single generator."""
        if self.n_gens == 0:
            return True
        norm = self.normalize().relators
        expected = {tuple([i]) for i in range(1, self.n_gens + 1)}
        actual = {tuple(r) for r in norm}
        return actual == expected

    # -- Pretty printing ----------------------------------------------------

    def show(self, letters: Optional[list[str]] = None) -> str:
        """Render the presentation as a human-readable string."""
        gens_str = " ".join(
            letters[i] if letters and i < len(letters) else f"x{i+1}"
            for i in range(self.n_gens)
        )
        rels_str = ", ".join(word_str(r, letters) for r in self.relators)
        return f"⟨ {gens_str}  |  {rels_str} ⟩"

    def __str__(self) -> str:
        return self.show()

    def __repr__(self) -> str:
        return f"Presentation({self.n_gens}, {self.relators})"


# ═══════════════════════════════════════════════════════════════════════
# 3.  Top-level move functions
# ═══════════════════════════════════════════════════════════════════════

# Supported move types for use with apply_move()
MOVE_INVERT = "invert"
MOVE_MULTIPLY = "multiply"
MOVE_MULTIPLY_INV = "multiply_inv"
MOVE_MULTIPLY_LEFT = "multiply_left"
MOVE_MULTIPLY_LEFT_INV = "multiply_left_inv"
MOVE_CONJUGATE = "conjugate"
MOVE_PERMUTE = "permute"


def apply_move(pres: Presentation, move_type: str, i: int,
               j: Optional[int] = None,
               gen: Optional[Letter] = None,
               perm: Optional[list[int]] = None) -> Presentation:
    """Unified entry point for applying any AC move.

    Parameters
    ----------
    pres : Presentation
        The presentation to transform.
    move_type : str
        One of ``"invert"``, ``"multiply"``, ``"multiply_inv"``,
        ``"multiply_left"``, ``"multiply_left_inv"``, ``"conjugate"``,
        ``"permute"``.
    i : int
        Index of the relator to change (0-based).
    j : int or None
        Index of the second relator for multiply moves.
    gen : Letter or None
        Generator (positive) or its inverse (negative) for conjugate.
    perm : list[int] or None
        Permutation list for permute.

    Returns
    -------
    Presentation
        A new presentation after applying the requested move.

    Raises
    ------
    ValueError
        If the move type is unknown or required parameters are missing.
    """
    if move_type == MOVE_INVERT:
        return pres.invert_relator(i)
    elif move_type == MOVE_MULTIPLY:
        if j is None:
            raise ValueError("multiply requires j")
        return pres.multiply_relator(i, j, use_inverse=False)
    elif move_type == MOVE_MULTIPLY_INV:
        if j is None:
            raise ValueError("multiply_inv requires j")
        return pres.multiply_relator(i, j, use_inverse=True)
    elif move_type == MOVE_MULTIPLY_LEFT:
        if j is None:
            raise ValueError("multiply_left requires j")
        return pres.multiply_relator_left(i, j, use_inverse=False)
    elif move_type == MOVE_MULTIPLY_LEFT_INV:
        if j is None:
            raise ValueError("multiply_left_inv requires j")
        return pres.multiply_relator_left(i, j, use_inverse=True)
    elif move_type == MOVE_CONJUGATE:
        if gen is None:
            raise ValueError("conjugate requires gen")
        return pres.conjugate_relator(i, gen)
    elif move_type == MOVE_PERMUTE:
        if perm is None:
            raise ValueError("permute requires perm")
        return pres.permute_relators(perm)
    else:
        raise ValueError(f"Unknown move_type: {move_type}")


# ═══════════════════════════════════════════════════════════════════════
# 4.  Move enumeration
# ═══════════════════════════════════════════════════════════════════════

MoveRecord = tuple[str, Callable[[Presentation], Presentation]]


def enumerate_moves(p: Presentation) -> list[MoveRecord]:
    """Return all single-AC-move closures applicable to presentation *p*.

    Each entry is ``(description_string, callable)`` where the callable
    takes a ``Presentation`` and returns a new ``Presentation``.

    The branching factor grows as O(n^2) where n = n_gens.
    For n=2 there are ~20 distinct move records.
    """
    n = p.n_gens
    moves: list[MoveRecord] = []

    for i in range(n):
        moves.append((f"INV r{i}", lambda q, idx=i: q.invert_relator(idx)))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            moves.append((f"MUL r{i}·r{j}", lambda q, a=i, b=j: q.multiply_relator(a, b)))
            moves.append((f"MUL r{i}·r{j}⁻¹", lambda q, a=i, b=j: q.multiply_relator(a, b, use_inverse=True)))
            moves.append((f"MUL r{j}·r{i}", lambda q, a=i, b=j: q.multiply_relator_left(a, b)))
            moves.append((f"MUL r{j}⁻¹·r{i}", lambda q, a=i, b=j: q.multiply_relator_left(a, b, use_inverse=True)))

    for i in range(n):
        for g in range(1, n + 1):
            moves.append((f"CONJ r{i} by x{g}", lambda q, idx=i, gen=g: q.conjugate_relator(idx, gen)))
            moves.append((f"CONJ r{i} by x{g}⁻¹", lambda q, idx=i, gen=-g: q.conjugate_relator(idx, -g)))

    for i in range(n - 1):
        perm = list(range(n))
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        moves.append((f"SWAP r{i}↔r{i+1}", lambda q, p=perm: q.permute_relators(p)))

    return moves


# ═══════════════════════════════════════════════════════════════════════
# 5.  Search algorithms
# ═══════════════════════════════════════════════════════════════════════

SearchResult = tuple[bool, Optional[list[tuple[str, Presentation]]], dict]


@dataclass
class ACSearcher:
    """BFS / heuristic search for AC-trivializations.

    Parameters
    ----------
    max_states : int
        Maximum number of distinct states to explore before giving up.
    max_depth : int
        Maximum BFS depth (moves applied) before giving up.
    heuristic : bool
        If True, use a priority queue ordered by total relator length
        (A*-like) instead of strict BFS.  The heuristic is *admissible*
        because length only decreases when progress is made.
    verbose : bool
        Print progress during search.
    early_prune_factor : int
        If a state's total length exceeds this multiple of the best
        length seen so far (plus a constant), the branch is skipped.
        Helps control search-space explosion.
    """

    max_states: int = 200_000
    max_depth: int = 30
    heuristic: bool = True
    verbose: bool = True
    early_prune_factor: int = 10

    # ------------------------------------------------------------------
    def search(
        self,
        start: Presentation,
        target: Optional[Presentation] = None,
        timeout: float = 120.0,
    ) -> SearchResult:
        """Run the search.

        Parameters
        ----------
        start : Presentation
            Balanced presentation to start from.
        target : Presentation or None
            Target presentation (default: all-relators-empty).
        timeout : float
            Wall-clock time limit in seconds.

        Returns
        -------
        (found, path, stats)
            found  : True iff a trivialization was found.
            path   : list of (move_description, presentation_after_move)
                     if found, else None.
            stats  : dict with search statistics (states explored, depth,
                     best length, reason for termination, elapsed time).
        """
        if target is None:
            target = Presentation.trivial(start.n_gens)

        start_key = start.state_key()
        target_key = target.state_key()

        if start_key == target_key:
            return True, [], {"states_explored": 0, "depth": 0,
                              "reason": "already trivial"}

        start_priority = start.total_length() if self.heuristic else 0

        # Priority queue entries: (priority, depth, tiebreaker, pres, path)
        queue: list[tuple] = []
        tiebreaker = 0
        heappush(queue, (start_priority, 0, tiebreaker, start, []))
        tiebreaker += 1

        visited: dict = {start_key: 0}
        states_explored = 0
        best_length = start.total_length()
        last_report = 0.0
        t0 = time.monotonic()

        while queue and states_explored < self.max_states:
            priority, depth, _, pres, path = heappop(queue)

            if time.monotonic() - t0 > timeout:
                return False, None, {
                    "states_explored": states_explored,
                    "depth": depth,
                    "best_length": best_length,
                    "reason": "timeout",
                    "elapsed": time.monotonic() - t0,
                }

            # -- progress reporting (every 2 seconds) --
            if self.verbose and time.monotonic() - last_report > 2.0:
                tl = pres.total_length()
                if tl < best_length:
                    best_length = tl
                print(
                    f"  [depth={depth:2d}  explored={states_explored:7d}  "
                    f"queue={len(queue):7d}  best_len={best_length:3d}]",
                    flush=True,
                )
                last_report = time.monotonic()

            if depth >= self.max_depth:
                continue

            # Generate all successor states
            for move_desc, move_fn in enumerate_moves(pres):
                new_pres = move_fn(pres)
                key = new_pres.state_key()

                if key in visited and visited[key] <= depth + 1:
                    continue
                visited[key] = depth + 1
                states_explored += 1

                new_path = path + [(move_desc, new_pres)]

                # Success check: all empty, standard trivial, or matched target
                if new_pres.is_trivial() or new_pres.is_standard_trivial() or key == target_key:
                    if self.verbose:
                        print(
                            f"\n  ✓ FOUND trivialization at depth {depth+1} "
                            f"after {states_explored} states!",
                            flush=True,
                        )
                    return True, new_path, {
                        "states_explored": states_explored,
                        "depth": depth + 1,
                        "best_length": 0,
                        "reason": "found",
                        "elapsed": time.monotonic() - t0,
                    }

                # Pruning: skip states whose total length has exploded
                tl = new_pres.total_length()
                if tl > self.early_prune_factor * best_length + 100:
                    continue
                if tl < best_length:
                    best_length = tl

                priority = tl if self.heuristic else depth + 1
                heappush(queue, (priority, depth + 1, tiebreaker,
                                 new_pres, new_path))
                tiebreaker += 1

        # Exhausted search space
        return False, None, {
            "states_explored": states_explored,
            "depth": self.max_depth,
            "best_length": best_length,
            "reason": "max_states or max_depth reached",
            "elapsed": time.monotonic() - t0,
        }


# ═══════════════════════════════════════════════════════════════════════
# 6.  Example presentations
# ═══════════════════════════════════════════════════════════════════════

def example_trivial() -> Presentation:
    """⟨ a, b  |  a, b ⟩    — the standard trivial presentation.

    Already trivial — serves as a sanity check.
    """
    return Presentation.trivial(2)


def example_myasnikov() -> Presentation:
    """⟨ a, b  |  a² b⁻³,  a b a b⁻¹ a⁻¹ b⁻¹ ⟩

    Known to be AC-trivializable (Myasnikov, reported in
    Myasnikov–Myasnikov–Shpilrain 2002).  Total length = 11.
    """
    return Presentation.from_lists(2, [
        [1, 1, -2, -2, -2],
        [1, 2, 1, -2, -1, -2],
    ])


def example_ak3() -> Presentation:
    """⟨ a, b  |  a³ b⁴,  a b a b⁻¹ a⁻¹ b⁻¹ ⟩   —  AK(3)

    Akbulut–Kirby series:  AK(n) = ⟨ a,b | a^{n} b^{n+1}, abab^{-1}a^{-1}b^{-1} ⟩.
    AK(3) has total length 13.  Its AC-status is OPEN (as of 2026).
    This is one of the smallest open potential counterexamples.
    """
    return Presentation.from_lists(2, [
        [1, 1, 1, 2, 2, 2, 2],
        [1, 2, 1, -2, -1, -2],
    ])


def example_dunwoody() -> Presentation:
    """⟨ a, b  |  a b a b⁻¹ a⁻¹ b⁻¹,  a² b⁻³ ⟩

    Dunwoody-style potential counterexample.  Total length = 11.
    """
    return Presentation.from_lists(2, [
        [1, 2, 1, -2, -1, -2],
        [1, 1, -2, -2, -2],
    ])


def example_miller_schupp() -> Presentation:
    """⟨ a, b  |  a⁻¹ b⁹ a b⁻¹⁰,  a⁻¹ b⁻¹ a b a⁻¹ ⟩   (Miller–Schupp MS9)

    Trivializes in 8,634 moves (Lisitsa, 2025, using Prover9 ATP).
    Total length = 26.  This is a stress test — will NOT succeed at
    shallow depth without advanced search techniques.
    """
    return Presentation.from_lists(2, [
        [-1] + [2] * 9 + [1] + [-2] * 10,
        [-1, -2, 1, 2, -1],
    ])


# ═══════════════════════════════════════════════════════════════════════
# 7.  Command-line interface
# ═══════════════════════════════════════════════════════════════════════

EXAMPLES: dict[str, tuple[Presentation, str]] = {
    "trivial": (example_trivial(),
                "Standard trivial — sanity check (2 gens, len=2)"),
    "myasnikov": (example_myasnikov(),
                  "Myasnikov — known AC-trivializable (2 gens, len=11)"),
    "ak3": (example_ak3(),
            "Akbulut–Kirby AK(3) — OPEN POTENTIAL COUNTEREXAMPLE (2 gens, len=13)"),
    "dunwoody": (example_dunwoody(),
                 "Dunwoody-style — OPEN (2 gens, len=11)"),
    "ms9": (example_miller_schupp(),
            "Miller–Schupp MS9 — 8634-move trivialization (Lisitsa 2025) (2 gens, len=26)"),
}

ALL_NAMES = list(EXAMPLES)


def safe_input(prompt: str, default: str = "") -> str:
    """Wrapper around ``input()`` that returns *default* on EOF."""
    try:
        r = input(prompt).strip()
        return r if r else default
    except EOFError:
        return default


def main() -> None:
    print("═" * 66)
    print("  Andrews–Curtis Conjecture — Computational Searcher")
    print("═" * 66)
    print()
    print("  Available examples:\n")

    for i, name in enumerate(ALL_NAMES):
        pres, desc = EXAMPLES[name]
        ltrs = ["a", "b"] if pres.n_gens <= 2 else None
        print(f"  [{i}]  {name:12s}  — {desc}")
        print(f"       {pres.show(ltrs)}")
        print()

    raw = safe_input("  Select example [0-4, default=0]: ", "0")
    try:
        idx = int(raw)
        name = ALL_NAMES[idx]
    except (ValueError, IndexError):
        print("  Invalid choice, using example 0.")
        name = ALL_NAMES[0]

    pres, desc = EXAMPLES[name]
    print(f"\n  Searching: {name} — {desc}")
    print(f"  Starting:  {pres}")
    print()

    max_states = int(safe_input("  Max states to explore [200000]: ", "200000"))
    max_depth = int(safe_input("  Max search depth [20]: ", "20"))
    heuristic_str = safe_input("  Use heuristic (A*)? [Y/n]: ", "y").lower()
    heuristic = heuristic_str not in ("n", "no", "false", "0")
    verbose_str = safe_input("  Verbose progress? [Y/n]: ", "y").lower()
    verbose = verbose_str not in ("n", "no", "false", "0")
    timeout = float(safe_input("  Timeout (seconds) [60]: ", "60"))

    searcher = ACSearcher(
        max_states=max_states,
        max_depth=max_depth,
        heuristic=heuristic,
        verbose=verbose,
    )

    print("\n  Starting search...\n")
    t0 = time.monotonic()
    found, path, stats = searcher.search(pres, timeout=timeout)
    elapsed = time.monotonic() - t0

    print()
    print("─" * 66)
    print("  RESULTS")
    print("─" * 66)
    if found:
        print(f"  ✓  TRIVIALIZATION FOUND!  Depth = {stats['depth']}")
        if path and verbose:
            print("\n  Move sequence:")
            ltrs = ["a", "b"] if pres.n_gens <= 2 else None
            for step, (desc, state) in enumerate(path):
                tl = state.total_length()
                print(f"    [{step+1:3d}] {desc:20s}  "
                      f"{state.show(ltrs)}  (len={tl})")
    else:
        print(f"  ✗  Trivialization NOT FOUND within search limits.")
        print(f"     Reason: {stats.get('reason', 'unknown')}")

    print(f"\n  Statistics:")
    print(f"    States explored:  {stats['states_explored']}")
    print(f"    Max depth:        {stats.get('depth', 'N/A')}")
    print(f"    Best length:      {stats.get('best_length', 'N/A')}")
    print(f"    Elapsed:          {elapsed:.2f} s")
    print()


# ═══════════════════════════════════════════════════════════════════════
# 8.  Extension notes (README-style)
# ═══════════════════════════════════════════════════════════════════════
#
# How to extend this solver
# -------------------------
#
# 1.  Additional AC-moves
#     -------------------
#     Stabilisation moves (AC4/AC5) allow adding/removing a generator
#     and corresponding relator.  To add: modify ``enumerate_moves()``
#     to yield ``stabilize_add`` / ``stabilize_remove`` closures.
#     These let the searcher change the number of generators, which is
#     necessary for the *weak* Andrews–Curtis conjecture.
#
# 2.  Parallel / distributed search
#     ------------------------------
#     The search is embarrassingly parallel at the frontier level.
#     Replace the single priority queue with a Redis-backed queue or
#     Python ``multiprocessing`` workers that each pop a batch of
#     frontier states, expand them, and push successors.  A shared
#     Bloom filter (``pybloom_live``) can reduce visited-set memory.
#
# 3.  Better heuristics
#     ------------------
#     The current heuristic (total relator length) is admissible but
#     weak.  Better candidates:
#       * Length of the longest relator.
#       * Number of distinct generator symbols appearing.
#       * Cyclic reduced length (ignore cyclic permutations).
#       * Smallest length achievable by a single Nielsen move
#         (look-ahead heuristic).
#       * ML regression model trained on known simplification paths
#         to predict "distance to trivial".
#
# 4.  Integration with SAT / SMT solvers
#     -----------------------------------
#     Following Lisitsa (2013, 2025), the AC-equivalence problem can
#     be encoded as a first-order logic theory  ACT_n  and given to
#     automated theorem provers (Prover9, Vampire, E).  The search
#     then becomes a proof search rather than a state-space search.
#     This approach has found trivialisations thousands of moves long
#     that BFS cannot reach.
#
# 5.  Genetic algorithms
#     -------------------
#     Myasnikov (1999) used a genetic algorithm to find simplification
#     sequences.  Each genome is a sequence of AC-moves; fitness is
#     the final total relator length (or negative depth).  Crossover
#     splices move sequences; mutation inserts/deletes/changes moves.
#     This can find solutions at much greater depth than BFS.
#
# 6.  Symmetry reduction
#     -------------------
#     The search graph has many symmetries: renaming generators,
#     permuting relators, etc.  Better canonicalisation (e.g. using
#     a canonical form under the full automorphism group of the free
#     group) would reduce visited-set size by orders of magnitude.
#
# 7.  Bounded-width / beam search
#     ----------------------------
#     Keep only the top-k most promising states at each depth level.
#     This trades completeness for memory efficiency.  Can be combined
#     with iterative deepening.
#
# 8.  Domain-specific pruning
#     ------------------------
#     The relator length frequently explodes before it contracts.
#     Heuristics that detect "runaway" growth and deprioritise those
#     branches are critical.  The current ``early_prune_factor`` is a
#     simple version; more sophisticated approaches include not
#     applying multiplications that increase length beyond a moving
#     window average.
#
# --------------------------------------------------------------------
# Theoretical limitations
# -----------------------
# Bridson (2015) proved that the length of AC-simplification sequences
# is bounded below by a superexponential function of the input length.
# This means that even if the conjecture holds, exhaustive search will
# fail for all but the shortest presentations.  The known 8,634-move
# trivialization for MS9 required ~74 days and 56 GB of RAM using an
# automated theorem prover (Lisitsa 2025).
#
# These computational approaches are therefore best understood as
# *experimental mathematics* tools: they discover new simplifications
# and eliminate potential counterexamples, but they cannot disprove
# the conjecture (which would require a single non-trivializable
# presentation), and they cannot prove it (which would require showing
# that *every* presentation trivializes).
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
