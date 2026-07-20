"""
Andrews–Curtis move engine.

Provides the core data structures and transformations for exploring
the Andrews–Curtis conjecture on balanced group presentations.

A balanced presentation is ⟨x₁ … xₙ | r₁ … rₙ⟩ with equal numbers of
generators and relators.  The AC moves are:

  AC₁  invert a relator
  AC₂  permute relators
  AC₃  multiply one relator by another (Nielsen)
  AC₃′ conjugate a relator by a generator (or its inverse)

References
----------
AC65 : J. J. Andrews & M. L. Curtis, Proc. Amer. Math. Soc. 16 (1965)
AC66 : J. J. Andrews & M. L. Curtis, Amer. Math. Monthly 73 (1966)
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Words in a free group
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Letter:
    """A single generator or its inverse."""
    generator: str
    exponent: int = 1  # +1 or -1

    def __post_init__(self):
        assert self.exponent in (1, -1), f"exponent must be ±1, got {self.exponent}"

    def inverse(self) -> Letter:
        return Letter(self.generator, -self.exponent)

    def __str__(self) -> str:
        if self.exponent == 1:
            return self.generator
        return self.generator + "⁻¹"

    def __repr__(self) -> str:
        return str(self)


def reduce_word(word: list[Letter]) -> list[Letter]:
    """Cancel adjacent inverse pairs (free reduction)."""
    stack: list[Letter] = []
    for a in word:
        if stack and stack[-1].generator == a.generator and stack[-1].exponent == -a.exponent:
            stack.pop()
        else:
            stack.append(a)
    return stack


def letter_str(letter: Letter) -> str:
    return str(letter)


def word_str(word: list[Letter]) -> str:
    return "".join(letter_str(a) for a in word) or "ε"


def parse_word(s: str, generator_set: Optional[set[str]] = None) -> list[Letter]:
    """Parse a string like ``"a b⁻¹ a b"`` into a reduced word."""
    tokens = re.findall(r"[a-zA-Z](?:⁻¹|⁻?¹|⁻?)?" , s)
    result = []
    for tok in tokens:
        gen = tok[0]
        if generator_set is not None and gen not in generator_set:
            raise ValueError(f"Unknown generator '{gen}'")
        exp = 1
        if "⁻" in tok or "⁻¹" in tok or "⁻1" in tok:
            exp = -1
        result.append(Letter(gen, exp))
    return reduce_word(result)


# ---------------------------------------------------------------------------
# Presentations
# ---------------------------------------------------------------------------

@dataclass
class Presentation:
    """A balanced group presentation ⟨X | R⟩."""
    generators: list[str]
    relators: list[list[Letter]]

    def __post_init__(self):
        assert len(self.generators) == len(self.relators), (
            f"Presentation must be balanced: {len(self.generators)} generators vs "
            f"{len(self.relators)} relators"
        )

    def copy(self) -> Presentation:
        return deepcopy(self)

    def __str__(self) -> str:
        gens = " ".join(self.generators)
        rels = ", ".join(word_str(r) for r in self.relators)
        return f"⟨{gens} | {rels}⟩"

    def __repr__(self) -> str:
        return str(self)

    def total_length(self) -> int:
        return sum(len(r) for r in self.relators)


# ---------------------------------------------------------------------------
# AC moves
# ---------------------------------------------------------------------------

def ac1_invert(p: Presentation, i: int) -> Presentation:
    """AC₁: replace relator rᵢ by rᵢ⁻¹."""
    assert 0 <= i < len(p.relators)
    q = p.copy()
    q.relators[i] = reduce_word([a.inverse() for a in reversed(q.relators[i])])
    return q


def ac2_permute(p: Presentation, perm: list[int]) -> Presentation:
    """AC₂: permute relators according to *perm* (a permutation of
    ``range(n)``)."""
    assert sorted(perm) == list(range(len(p.relators)))
    q = p.copy()
    q.relators = [q.relators[j] for j in perm]
    return q


def ac3_multiply(p: Presentation, i: int, j: int, side: str = "right") -> Presentation:
    """AC₃: replace rᵢ by rᵢ·rⱼ (left or right)."""
    assert i != j
    assert side in ("left", "right")
    q = p.copy()
    left = q.relators[j] + q.relators[i] if side == "left" else q.relators[i] + q.relators[j]
    q.relators[i] = reduce_word(left)
    return q


def ac3p_conjugate(p: Presentation, i: int, gen: str) -> Presentation:
    """AC₃′: conjugate relator rᵢ by generator *gen*."""
    assert gen in p.generators or gen.endswith("⁻¹") and gen.rstrip("⁻¹") in p.generators
    g = gen.rstrip("⁻¹")
    inv = "-1" if gen.endswith("⁻¹") else ""
    q = p.copy()
    conj = parse_word(f"{gen} {word_str(q.relators[i])} {g}{inv}", set(p.generators))
    q.relators[i] = conj
    return q


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

ACMove = Callable[[Presentation], Presentation]


def all_ac_moves(p: Presentation) -> list[ACMove]:
    """Generate a list of all single-step AC-move closures applicable to *p*.

    Returns a list of callables that each accept a Presentation and return
    a new Presentation with one AC move applied.
    """
    moves: list[ACMove] = []
    n = len(p.relators)

    for i in range(n):
        moves.append(lambda q, idx=i: ac1_invert(q, idx))

    # AC₂ — all non-identity permutations (minimal set: adjacent swaps)
    for i in range(n - 1):
        perm = list(range(n))
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
        moves.append(lambda q, p_=perm: ac2_permute(q, p_))

    # AC₃ — rᵢ ← rᵢ·rⱼ (right)
    for i in range(n):
        for j in range(n):
            if i != j:
                moves.append(lambda q, a=i, b=j: ac3_multiply(q, a, b, "right"))

    # AC₃′ — conjugate by each generator
    for i in range(n):
        for g in p.generators:
            moves.append(lambda q, a=i, gen=g: ac3p_conjugate(q, a, gen))
            moves.append(lambda q, a=i, gen=g + "⁻¹": ac3p_conjugate(q, a, gen + "⁻¹"))

    return moves


# ---------------------------------------------------------------------------
# Known presentations (potential counterexamples)
# ---------------------------------------------------------------------------

def akbulut_kirby(n: int) -> Presentation:
    """Akbulut–Kirby series (n ≥ 2)."""
    g = ["a", "b"]
    a, b = Letter("a"), Letter("b")
    r1 = [a, b, a, b, a, b.inverse(), a, b.inverse(), a, b.inverse(), a.inverse(), b, a.inverse(), b, a.inverse(), b.inverse(), a.inverse(), b.inverse()]
    r2 = []
    return Presentation(g, [reduce_word(r1), reduce_word(r2)])


def trivial_presentation(n: int) -> Presentation:
    """The trivial presentation ⟨x₁…xₙ | x₁…xₙ⟩."""
    gens = [f"x{i}" for i in range(1, n + 1)]
    rels = [[Letter(g)] for g in gens]
    return Presentation(gens, rels)


# ---------------------------------------------------------------------------
# Simple cli
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = trivial_presentation(2)
    print("Trivial:", p)
    p2 = ac1_invert(p, 0)
    print("AC₁ on r₀:", p2)
    print("Length:", p.total_length())
