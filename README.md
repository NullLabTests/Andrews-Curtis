# Andrews–Curtis Conjecture

Investigations into the Andrews–Curtis conjecture (1965) in combinatorial group theory.

## Overview

The **Andrews–Curtis conjecture** (J. J. Andrews and M. L. Curtis, 1965) states that every **balanced presentation** of the trivial group can be transformed into the trivial presentation by a finite sequence of **Andrews–Curtis moves**:

| Move | Description |
|------|-------------|
| (AC1) | Replace a relator by its inverse |
| (AC2) | Permute the relators |
| (AC3) | Replace a relator $r_i$ by $r_i r_j$ ($j \neq i$) |
| (AC3') | Conjugate a relator by any generator |

A **balanced presentation** is one with the same number of generators and relators: $\langle x_1, \dots, x_n \mid r_1, \dots, r_n \rangle$.

The conjecture remains **open** (2026). It is widely believed to be **false**, but no counterexamples are known. It is related to the Zeeman conjecture, the Poincaré conjecture, and the 3-deformation problem for contractible 2-complexes.

## Status

- **1965** — Conjecture proposed by Andrews and Curtis
- **2003** — Perelman proves Poincaré conjecture (implies stable AC for thickenable presentations)
- **2025** — Longest AC-simplification found: 8,634 moves for a Miller–Schupp presentation (Lisitsa)
- **2026** — Computational search: AK(3) = ⟨a,b | a³b⁴, abab⁻¹a⁻¹b⁻¹⟩ resists ALL trivialization attempts (15M+ states, 9 configurations, up to depth 500). See [RESULTS.md](RESULTS.md).

## Repository Structure

```
├── README.md           # This file
├── src/                # Computational tools for AC exploration
├── notes/              # Research notes and observations
└── references/         # Papers and external resources
```

## Key Concepts

- **Nielsen transformations** on relators + conjugation = AC moves
- **Stabilization** (AC4/AC5): adding/removing a generator+relator pair
- **Weak AC conjecture**: allows stabilizations
- **Thickenable presentations**: those whose 2-complex embeds in a 3-manifold (known to satisfy stable AC)
- **Potential counterexamples**: Akbulut–Kirby series, Miller–Schupp series, AK(3)
- **Solver** (this repo): V5 merged solver with forward beam search, bidirectional meet-in-the-middle, substitution super-moves, weak/cyclic normal forms, novelty search, and macro mining

## References

- Andrews, J. J. and Curtis, M. L. *Free groups and handlebodies*. Proc. Amer. Math. Soc. 16 (1965), 192–195.
- Andrews, J. J. and Curtis, M. L. *Extended Nielsen Operations in Free Groups*. Amer. Math. Monthly 73 (1966), 21–28.
- Bridson, M. R. *The complexity of balanced presentations and the Andrews-Curtis conjecture*. arXiv:1504.04187 (2015).
- Lisitsa, A. *Automated theorem proving reveals a lengthy Andrews–Curtis trivialization*. Examples and Counterexamples (2025).
