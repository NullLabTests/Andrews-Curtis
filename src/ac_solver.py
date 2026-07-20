#!/usr/bin/env python3
"""
Andrews–Curtis Conjecture — Computational Searcher
===================================================

The Andrews–Curtis conjecture (1965) states that every balanced
presentation of the trivial group can be transformed into the standard
trivial presentation via a finite sequence of AC-moves.

A presentation is *balanced* iff it has the same number of generators as
relators.

AC-moves (on the relators, with generators fixed):
  1. Invert a relator:           r_i → r_i⁻¹
  2. Multiply one relator by another:  r_i → r_i · r_j   (or r_i · r_j⁻¹)
  3. Conjugate a relator by a generator:  r_i → g_k · r_i · g_k⁻¹
  4. Permute the relators

The *standard trivial presentation* is  ⟨ x₁ … xₙ | x₁ … xₙ ⟩.

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

import itertools
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Callable, Iterator, Optional


# ═══════════════════════════════════════════════════════════════════════
# 1.  Free-group word utilities
# ═══════════════════════════════════════════════════════════════════════

Letter = int  # positive = generator, negative = inverse


def free_reduce(word: list[Letter]) -> list[Letter]:
    """Fully cancel adjacent inverse pairs (free reduction).

    Stack-based, O(len(word)).  Returns a new list.
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
    """Reverse and negate every letter."""
    return [-a for a in reversed(word)]


def conjugate(word: list[Letter], gen: Letter) -> list[Letter]:
    """Return  gen · word · gen⁻¹   after free reduction."""
    return free_reduce([gen] + word + [-gen])


def word_len(word: list[Letter]) -> int:
    return len(word)


def word_str(word: list[Letter], letters: Optional[list[str]] = None) -> str:
    """Pretty-print a word.

    If *letters* is given (e.g. ``['a','b']``) the integers are mapped to
    those characters; otherwise we use ``x1, x2, …``.
    """
    if not word:
        return "ε"
    parts: list[str] = []
    for a in word:
        if a > 0:
            if letters:
                name = letters[a - 1] if a - 1 < len(letters) else f"x{a}"
                parts.append(name)
            else:
                parts.append(f"x{a}")
        else:
            if letters:
                name = letters[(-a) - 1] if (-a) - 1 < len(letters) else f"x{-a}"
                parts.append(name + "⁻¹")
            else:
                parts.append(f"x{-a}⁻¹")
    return "·".join(parts) if len(parts) <= 6 else "".join(parts)


def parse_word(s: str, n_gens: int) -> list[Letter]:
    """Parse a simple string into a word.

    Recognises ``a, b, c`` or ``x1, x2`` with optional ``^-1`` or ``⁻¹``.
    """
    import re
    tokens = re.findall(r"[a-zA-Z]\d*(?:⁻¹|⁻?¹|⁻?)?", s)
    result: list[Letter] = []
    for tok in tokens:
        m = re.match(r"([a-zA-Z])(\d*)(.*)", tok)
        if not m:
            continue
        prefix = m.group(1).lower()
        num = int(m.group(2)) if m.group(2) else 1
        idx = ord(prefix) - ord("a") + 1 if prefix.isalpha() else num
        if idx > n_gens:
            raise ValueError(f"Generator index {idx} exceeds n_gens={n_gens}")
        exp = -1 if "⁻" in m.group(3) or m.group(3) in ("⁻¹", "⁻1", "-1") else 1
        result.append(exp * idx)
    return free_reduce(result)


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
        One list of integers per relator.  Each is kept freely reduced.
    """

    n_gens: int
    relators: list[list[Letter]]

    def __post_init__(self) -> None:
        assert len(self.relators) == self.n_gens, (
            f"Balanced required: {self.n_gens} gens vs {len(self.relators)} rels"
        )

    # -- Convenience constructors ------------------------------------------

    @staticmethod
    def trivial(n: int = 2) -> Presentation:
        """Return the standard trivial presentation  ⟨ x₁…xₙ | x₁…xₙ ⟩."""
        return Presentation(n, [[i] for i in range(1, n + 1)])

    @staticmethod
    def from_strings(n_gens: int, relator_strs: list[str]) -> Presentation:
        """Build from strings like ``["a a b", "b a"]``.

        Letters ``a..z`` map to generators 1..26.
        """
        assert len(relator_strs) == n_gens
        relators = [parse_word(s, n_gens) for s in relator_strs]
        return Presentation(n_gens, relators)

    # -- Accessors ---------------------------------------------------------

    def copy(self) -> Presentation:
        return Presentation(self.n_gens, [list(r) for r in self.relators])

    def total_length(self) -> int:
        return sum(len(r) for r in self.relators)

    def max_length(self) -> int:
        return max((len(r) for r in self.relators), default=0)

    # -- AC-move application (returns NEW Presentation) --------------------

    def invert(self, i: int) -> Presentation:
        """AC₁: r_i ← r_i⁻¹."""
        q = self.copy()
        q.relators[i] = invert(q.relators[i])
        return q

    def multiply(self, i: int, j: int, inv: bool = False) -> Presentation:
        """AC₂: r_i ← r_i · r_j   (if *inv*,  r_i ← r_i · r_j⁻¹)."""
        q = self.copy()
        rhs = invert(q.relators[j]) if inv else q.relators[j]
        q.relators[i] = multiply(q.relators[i], rhs)
        return q

    def multiply_left(self, i: int, j: int, inv: bool = False) -> Presentation:
        """r_i ← r_j · r_i   (left-multiply variant)."""
        q = self.copy()
        lhs = invert(q.relators[j]) if inv else q.relators[j]
        q.relators[i] = multiply(lhs, q.relators[i])
        return q

    def conjugate(self, i: int, gen: Letter) -> Presentation:
        """AC₃: r_i ← gen · r_i · gen⁻¹."""
        q = self.copy()
        q.relators[i] = conjugate(q.relators[i], gen)
        return q

    def permute(self, perm: list[int]) -> Presentation:
        """AC₄: reorder relators by given permutation."""
        q = self.copy()
        q.relators = [q.relators[p] for p in perm]
        return q

    # -- Norm / equality helpers -------------------------------------------

    def normalize(self) -> Presentation:
        """Return a canonical representative for equality testing.

        Steps:
          1. Free-reduce every relator.
          2. Remove trailing empty relators (or keep them balanced).
          3. Sort relators lexicographically.

        NOTE: For visited-state tracking we use a *sorted frozenset of
        tuples* on the normalized relators, which is more robust.
        """
        q = self.copy()
        q.relators = [free_reduce(r) for r in q.relators]
        q.relators.sort(key=tuple)
        return q

    def state_key(self) -> tuple:
        """Immutable hashable key for visited-set membership.

        Sorted tuple of tuples, ignoring empty relators (they carry no
        information).  The number of generators is also part of the key.
        """
        reduced = tuple(sorted(tuple(free_reduce(r)) for r in self.relators))
        return (self.n_gens, reduced)

    def is_trivial(self) -> bool:
        """Check if this IS the standard trivial presentation.

        The standard trivial presentation has each relator equal to a
        single generator (up to order).
        """
        if self.n_gens == 0:
            return True
        norm = self.normalize().relators
        if len(norm) != self.n_gens:
            return False
        # Check that for each generator i there is exactly one relator [i]
        expected = {tuple([i]) for i in range(1, self.n_gens + 1)}
        actual = {tuple(r) for r in norm}
        return actual == expected

    # -- Pretty printing ---------------------------------------------------

    def show(self, letters: Optional[list[str]] = None) -> str:
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
# 3.  Move enumeration
# ═══════════════════════════════════════════════════════════════════════

MoveFunction = Callable[[Presentation], Presentation]


def enumerate_moves(p: Presentation) -> list[tuple[str, MoveFunction]]:
    """Return all single-AC-move closures for presentation *p*.

    Returns a list of ``(description_string, callable)`` pairs.
    The callable takes a Presentation and returns a new Presentation.
    """
    n = p.n_gens
    moves: list[tuple[str, MoveFunction]] = []

    # AC₁  invert
    for i in range(n):
        moves.append((f"INV r{i}", lambda q, idx=i: q.invert(idx)))

    # AC₂  multiply  r_i ← r_i · r_j^{±1}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            moves.append((f"MUL r{i}·r{j}", lambda q, a=i, b=j: q.multiply(a, b)))
            moves.append((f"MUL r{i}·r{j}⁻¹", lambda q, a=i, b=j: q.multiply(a, b, inv=True)))
            moves.append((f"MUL r{j}·r{i}", lambda q, a=i, b=j: q.multiply_left(a, b)))
            moves.append((f"MUL r{j}⁻¹·r{i}", lambda q, a=i, b=j: q.multiply_left(a, b, inv=True)))

    # AC₃  conjugate by each generator (and its inverse)
    for i in range(n):
        for g in range(1, n + 1):
            moves.append((f"CONJ r{i} by x{g}", lambda q, idx=i, gen=g: q.conjugate(idx, gen)))
            # conj by inverse adds nothing new formally (gen^{-1}·r·gen), but
            # helps reach different parts of the search space
            moves.append((f"CONJ r{i} by x{g}⁻¹", lambda q, idx=i, gen=-g: q.conjugate(idx, gen)))

    # AC₄  adjacent swaps — enough to generate all permutations
    for i in range(n - 1):
        perm = list(range(n))
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        moves.append((f"SWAP r{i}↔r{i+1}", lambda q, p=perm: q.permute(p)))

    return moves


# ═══════════════════════════════════════════════════════════════════════
# 4.  Search algorithms
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
        (A*-like) instead of strict BFS.
    verbose : bool
        Print progress during search.
    """

    max_states: int = 200_000
    max_depth: int = 30
    heuristic: bool = True
    verbose: bool = True

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
            Target presentation (default: standard trivial).
        timeout : float
            Wall-clock time limit in seconds.

        Returns
        -------
        (found, path, stats)
            found  : True iff a trivialization was found.
            path   : list of (move_desc, pres_after_move) if found, else None.
            stats  : dict with search statistics.
        """
        if target is None:
            target = Presentation.trivial(start.n_gens)

        start_key = start.state_key()
        target_key = target.state_key()

        if start_key == target_key:
            return True, [], {"states_explored": 0, "depth": 0, "reason": "already trivial"}

        letters = [chr(ord("a") + i) for i in range(26)] if start.n_gens <= 26 else None

        # Priority queue entries:  (priority, depth, state, path)
        # priority = total_relator_length  (lower is better)
        # With heuristic=False, we use (depth,) so it's BFS.
        start_priority = start.total_length()
        if not self.heuristic:
            start_priority = 0

        queue: list[tuple] = []
        counter = 0
        heappush(queue, (start_priority, 0, counter, start, []))
        counter += 1

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

            # -- progress reporting --
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

                # Check for success
                if key == target_key or new_pres.is_trivial():
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

                # Prune: if total length has exploded, deprioritise
                tl = new_pres.total_length()
                if tl > 10 * best_length + 100:
                    continue  # skip this branch entirely
                if tl < best_length:
                    best_length = tl

                priority = tl if self.heuristic else depth + 1
                heappush(queue, (priority, depth + 1, counter, new_pres, new_path))
                counter += 1

        # Exhausted
        return False, None, {
            "states_explored": states_explored,
            "depth": self.max_depth,
            "best_length": best_length,
            "reason": "max_states or max_depth reached",
            "elapsed": time.monotonic() - t0,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5.  Example presentations
# ═══════════════════════════════════════════════════════════════════════

def example_trivial() -> Presentation:
    """⟨ a, b  |  a, b ⟩    — the standard trivial presentation.

    Already trivial — serves as a sanity check.
    """
    return Presentation.trivial(2)


def example_myasnikov() -> Presentation:
    """⟨ a, b  |  aab⁻¹b⁻¹b⁻¹,  abab⁻¹a⁻¹b⁻¹ ⟩

    Known to be AC-trivializable (Myasnikov).  Total length ≈ 12.
    """
    return Presentation(2, [
        [1, 1, -2, -2, -2],
        [1, 2, 1, -2, -1, -2],
    ])

def example_miller_schupp() -> Presentation:
    """⟨ a, b  |  a⁻¹b⁹ab⁻¹⁰,  a⁻¹b⁻¹aba⁻¹ ⟩   (Miller-Schupp MS9)

    Trivializes in 8,634 moves (Lisitsa 2025).  Total length = 23.
    Included as a stress test (will NOT succeed at shallow depth).
    """
    return Presentation(2, [
        [-1] + [2]*9 + [1] + [-2]*10,
        [-1, -2, 1, 2, -1],
    ])

def example_ak3() -> Presentation:
    """⟨ a, b  |  a³b⁴,  abab⁻¹a⁻¹b⁻¹ ⟩   —  AK(3) potential counterexample

    Akbulut–Kirby series AK(n) =  ⟨ a,b | a^{n} b^{n+1},  abab^{-1}a^{-1}b^{-1} ⟩.
    AK(3) total length = 13.  Its AC-status is OPEN (as of 2026).
    This is one of the smallest open cases.
    """
    return Presentation(2, [
        [1, 1, 1, 2, 2, 2, 2],
        [1, 2, 1, -2, -1, -2],
    ])


def example_dunwoody() -> Presentation:
    """⟨ a, b  |  abab⁻¹a⁻¹b⁻¹,  a²b⁻³ ⟩

    From Dunwoody / early potential counterexample literature.
    Total length = 12.
    """
    return Presentation(2, [
        [1, 2, 1, -2, -1, -2],
        [1, 1, -2, -2, -2],
    ])


# ═══════════════════════════════════════════════════════════════════════
# 6.  Command-line interface
# ═══════════════════════════════════════════════════════════════════════

EXAMPLES: dict[str, tuple[Presentation, str]] = {
    "trivial": (example_trivial(), "Standard trivial — sanity check"),
    "myasnikov": (example_myasnikov(), "Myasnikov — known AC-trivializable"),
    "ak3": (example_ak3(), "Akbulut–Kirby AK(3) — OPEN POTENTIAL COUNTEREXAMPLE"),
    "dunwoody": (example_dunwoody(), "Dunwoody-style — OPEN"),
    "ms9": (example_miller_schupp(), "Miller–Schupp MS9 — trivializes in 8634 moves (Lisitsa 2025)"),
}

ALL_NAMES = list(EXAMPLES)


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
        print(f"       {pres.show(ltrs)}   (len={pres.total_length()})")
        print()

    try:
        raw = input("  Select example [0-4, default=0]: ").strip()
    except EOFError:
        raw = ""
    choice = raw if raw else "0"
    try:
        idx = int(choice)
        name = ALL_NAMES[idx]
    except (ValueError, IndexError):
        print(f"  Invalid choice, using example 0.")
        name = ALL_NAMES[0]

    pres, desc = EXAMPLES[name]
    print(f"\n  Searching: {name} — {desc}")
    print(f"  Starting presentation: {pres}")
    print()

    # -- Configure search parameters --
    def safe_input(prompt: str, default: str = "") -> str:
        try:
            r = input(prompt).strip()
            return r if r else default
        except EOFError:
            return default

    max_states_str = safe_input("  Max states to explore [200000]: ", "200000")
    max_states = int(max_states_str)

    max_depth_str = safe_input("  Max search depth [20]: ", "20")
    max_depth = int(max_depth_str)

    heuristic_str = safe_input("  Use heuristic (A*)? [Y/n]: ", "y").lower()
    heuristic = heuristic_str not in ("n", "no", "false", "0")

    verbose_str = safe_input("  Verbose progress? [Y/n]: ", "y").lower()
    verbose = verbose_str not in ("n", "no", "false", "0")

    timeout_str = safe_input("  Timeout (seconds) [60]: ", "60")
    timeout = float(timeout_str)

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
                mark = " ◀ TRIVIAL" if state.is_trivial() else ""
                print(f"    [{step+1:3d}] {desc:20s}  {state.show(ltrs)}  (len={tl}){mark}")
    else:
        print(f"  ✗  Trivialization NOT FOUND within search limits.")
        print(f"     Reason: {stats.get('reason', 'unknown')}")

    print(f"\n  Statistics:")
    print(f"    States explored:  {stats['states_explored']}")
    print(f"    Max depth:        {stats.get('depth', 'N/A')}")
    print(f"    Best length:      {stats.get('best_length', 'N/A')}")
    print(f"    Elapsed:          {elapsed:.2f} s")
    print()


if __name__ == "__main__":
    main()
