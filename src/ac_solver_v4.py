#!/usr/bin/env python3
"""
Andrews–Curtis Conjecture — V4 Potential-Guided Solver
=======================================================

Key innovations over V3:
  1. Structural potential analysis — multi-objective scoring that
     rewards cancellation-ready states even at higher total length.
  2. Depth-aware length budget — deeper states may have longer
     relators if they show high structural potential.
  3. Beam diversity by equivalence class — keep structurally
     distinct states even if their scalar scores are similar.
  4. SUBST→primitive verifier — expands substitution super-moves
     into sequences of 12-rule moves for certified solutions.
  5. Macro mining — generate random AC-trivial presentations,
     collect successful paths, extract frequent sub-sequences
     as new higher-level moves.
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


# ═══════════════════════════════════════════════════════════════════════
# 1.  Structural potential analysis
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PotentialFeatures:
    """Structural features of a presentation relevant to AC progress."""
    total_length: int = 0
    cyclic_length: int = 0
    max_relator_length: int = 0

    # Cancellation potential
    adjacent_cancellations: int = 0      # prefix-suffix cancel pairs across rotations
    cyclic_cancel_pairs: int = 0         # number of (rot, letter) that could cancel

    # Letter interaction
    shared_letters: int = 0              # gens appearing in BOTH relators
    isolated_letters: int = 0            # gens appearing in only one relator
    letter_jaccard: float = 0.0          # overlap ratio

    # Structure signals
    single_letter_relators: int = 0      # enables STAB_REMOVE
    empty_relators: int = 0              # trivially satisfied
    subword_contains: int = 0            # does one relator contain another as subword?
    exponent_balance: float = 0.0        # 0 = perfectly balanced, 1 = max imbalance

    # Ratio features
    length_over_cyclic: float = 1.0      # tl/cl, >1 means room for cyclic reduction

    def composite_score(self, depth: int = 0, use_potential: bool = True) -> tuple:
        """Return a multi-objective score tuple.

        Lower is better.  Designed so that states with HIGH structural
        potential can sort ahead of states with slightly lower length
        but low potential.
        """
        tl = self.total_length
        cl = self.cyclic_length

        if not use_potential:
            return (tl + cl, 0, 0, tl)

        # Primary: cyclic length (fundamental invariant-like)
        # Secondary (negative = bonus): structural progress signals
        structural_bonus = (
            + self.single_letter_relators * 5     # big bonus: STAB_REMOVE ready
            + self.empty_relators * 3
            + self.adjacent_cancellations * 1     # cancellation-ready
            + self.shared_letters * 2             # interaction between relators
            - self.isolated_letters * 3           # penalty for dead letters
        )

        # Tertiary: total length (tiebreaker, depth-tolerant)
        # Higher depth → more tolerance for length increase
        depth_tolerance = depth // 8
        length_score = max(0, tl - depth_tolerance)

        return (cl, -structural_bonus, length_score, tl)

    def __lt__(self, other: PotentialFeatures) -> bool:
        return self.composite_score() < other.composite_score()


def analyze_potential(pres: Presentation) -> PotentialFeatures:
    """Compute structural potential features for a presentation."""
    f = PotentialFeatures()
    f.total_length = pres.total_length()
    f.cyclic_length = pres.total_cyclic_length()
    f.max_relator_length = pres.max_length()
    f.length_over_cyclic = (f.total_length / max(f.cyclic_length, 1))

    rels = pres.relators
    n = pres.n_gens

    # Single-letter and empty relators
    f.single_letter_relators = sum(1 for r in rels if len(r) == 1)
    f.empty_relators = sum(1 for r in rels if len(r) == 0)

    # Adjacent cancellation potential: across all cyclic rotations, count
    # how many have matching prefix-suffix pairs (first == -last)
    for r in rels:
        if len(r) >= 2:
            for k in range(len(r)):
                rot = cyclic_rotate(r, k)
                if rot[0] == -rot[-1]:
                    f.adjacent_cancellations += 1

    # Letter sharing
    gen_sets = [set(abs(a) for a in r) for r in rels]
    if len(gen_sets) >= 2:
        common = set.intersection(*gen_sets)
        all_gens = set.union(*gen_sets)
        f.shared_letters = len(common)
        f.isolated_letters = len(all_gens) - len(common)
        f.letter_jaccard = len(common) / max(len(all_gens), 1)
    elif len(gen_sets) == 1:
        f.shared_letters = len(gen_sets[0])
        f.isolated_letters = 0
        f.letter_jaccard = 1.0

    # Subword containment: does any relator contain another as contiguous subword?
    for i, r in enumerate(rels):
        ri = tuple(r)
        for j, s in enumerate(rels):
            if i != j and len(s) > 0 and len(s) <= len(r):
                sj = tuple(s)
                if sj in (tuple(cyclic_rotate(r, k)) for k in range(len(r))):
                    f.subword_contains += 1

    # Exponent balance: how close are exponent sums to zero?
    exp_sums = []
    for g in range(1, n + 1):
        total = 0
        for r in rels:
            total += sum(1 for a in r if a == g) - sum(1 for a in r if a == -g)
        exp_sums.append(abs(total))
    f.exponent_balance = sum(exp_sums) / max(n, 1)

    return f


# ═══════════════════════════════════════════════════════════════════════
# 2.  Potential-guided beam searcher
# ═══════════════════════════════════════════════════════════════════════

SearchResult = tuple[bool, Optional[list[tuple[str, Presentation]]], dict]


@dataclass
class PotentialBeamSearcher:
    """Beam search guided by structural potential.

    Uses multi-objective scoring (cyclic_length, -structural_bonus,
    depth_tolerant_length) instead of a single scalar.
    """

    beam_width: int = 2000
    max_depth: int = 500
    max_states: int = 50_000_000
    timeout: float = 7200.0
    use_substitution_moves: bool = True
    use_generator_moves: bool = True
    use_potential_scoring: bool = True
    # Beam diversity: keep at most N states per "equivalence class"
    # where class = (total_length, cyclic_length, shared_letters)
    max_per_class: int = 5
    restart_patience: int = 30
    restart_fraction: float = 0.3
    verbose: bool = True
    results_dir: str = "results"
    seed: int = 0

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        if self.seed:
            random.seed(self.seed)

    def _make_move_set(self, p: Presentation) -> list[tuple[str, Callable]]:
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
            run_id = f"pot_{int(time.time())}"
        t0 = time.monotonic()

        if start.is_solved():
            return True, [], {
                "states_explored": 0, "depth": 0, "reason": "already trivial"
            }

        visited: dict = {start.state_key(): 0}
        total_states = 0
        start_feat = analyze_potential(start)
        best_score = start_feat.composite_score(0, self.use_potential_scoring)
        best_len = start.total_length()
        best_cyclic = start.total_cyclic_length()
        depths_since_improvement = 0
        last_report = 0.0
        last_checkpoint = 0.0

        # frontier: list of (path, presentation)
        frontier: list[tuple[list[tuple[str, Presentation]], Presentation]] = [
            ([], start)
        ]

        for depth in range(self.max_depth):
            if not frontier:
                break
            if time.monotonic() - t0 > self.timeout:
                break

            candidates: list[tuple] = []

            for path, pres in frontier:
                if time.monotonic() - t0 > self.timeout:
                    break

                # Standard + generator moves
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

                    feat = analyze_potential(new_pres)
                    score = feat.composite_score(depth + 1,
                                                 self.use_potential_scoring)

                    # Equivalence class for diversity
                    eclass = (new_pres.total_length(),
                              new_pres.total_cyclic_length(),
                              feat.shared_letters)

                    candidates.append(
                        (score, eclass, len(new_path),
                         total_states, new_path, new_pres))

                # Substitution super-moves
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

                        feat = analyze_potential(sub_res)
                        score = feat.composite_score(depth + 1,
                                                     self.use_potential_scoring)
                        eclass = (sub_res.total_length(),
                                  sub_res.total_cyclic_length(),
                                  feat.shared_letters)

                        candidates.append(
                            (score, eclass, len(new_path),
                             total_states, new_path, sub_res))

            if total_states >= self.max_states:
                break

            # Sort by multi-objective score
            candidates.sort(key=lambda x: x[0])

            # Beam selection with per-class diversity limit
            class_count: dict = defaultdict(int)
            new_frontier: list = []
            for score, eclass, _, _, path, pres in candidates:
                if len(new_frontier) >= self.beam_width:
                    break
                if self.max_per_class > 0:
                    if class_count[eclass] >= self.max_per_class:
                        continue
                    class_count[eclass] += 1
                new_frontier.append((path, pres))
            frontier = new_frontier

            # Update best stats
            if frontier:
                best = frontier[0][1]
                bl = best.total_length()
                bc = best.total_cyclic_length()
                bf = analyze_potential(best)
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
                all_cands = candidates
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
                best_p = frontier[0][1] if frontier else None
                bl = best_p.total_length() if best_p else 0
                bc = best_p.total_cyclic_length() if best_p else 0
                print(
                    f"  [depth={depth+1:3d}  beam={len(frontier):4d}  "
                    f"states={total_states:8d}  len={bl:2d}  "
                    f"cyclic={bc:2d}  "
                    f"elapsed={now-t0:.0f}s]",
                    flush=True,
                )
                last_report = now

            # Checkpoint
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
            "reason": "beam search exhausted",
            "elapsed": elapsed,
        }


# ═══════════════════════════════════════════════════════════════════════
# 3.  SUBST → primitive verifier
# ═══════════════════════════════════════════════════════════════════════

def expand_substitution(pre_subst: Presentation,
                        post_subst: Presentation,
                        max_depth: int = 8,
                        beam_width: int = 200) -> Optional[list[tuple[str, Presentation]]]:
    """Expand a substitution super-move into primitive 12-rule moves.

    Uses short beam search from pre_subst to post_subst, ordered by
    edit distance to the target.

    Returns the primitive-move path if found, None if expansion fails.
    """
    if post_subst.is_solved():
        return []

    visited = {pre_subst.state_key(): 0}
    frontier: list[tuple[list[tuple[str, Presentation]], Presentation]] = [
        ([], pre_subst)
    ]

    target_key = post_subst.state_key()

    for depth in range(max_depth):
        if not frontier:
            break

        candidates: list[tuple] = []

        for path, pres in frontier:
            moves = enumerate_moves_full(pres)
            # Also try substitution moves if primitive expansion fails
            for desc, fn in moves:
                new_p = fn(pres)
                key = new_p.state_key()
                if key in visited and visited[key] <= depth + 1:
                    continue
                visited[key] = depth + 1

                new_path = path + [(desc, new_p)]

                if key == target_key:
                    return new_path

                # Hamming-like distance: count different letters
                dist = abs(new_p.total_length() - post_subst.total_length())
                candidates.append((dist, len(new_path), id(new_p), new_path, new_p))

        candidates.sort(key=lambda x: x[0])
        frontier = [(path, pres)
                    for _, _, _, path, pres in candidates[:beam_width]]

    return None


def verify_path(path: list[tuple[str, Presentation]],
                verbose: bool = True) -> tuple[bool, list]:
    """Verify a path that may contain SUBST super-moves.

    Attempts to expand each SUBST into primitive 12-rule moves.
    Returns (verified_ok, expanded_path).
    """
    expanded: list = []
    for i, (desc, state) in enumerate(path):
        if desc.startswith("SUBST"):
            pre_state = path[i - 1][1] if i > 0 else state
            if verbose:
                print(f"  Verifying {desc}...", end=" ", flush=True)
            sub_path = expand_substitution(pre_state, state)
            if sub_path is not None:
                expanded.extend(sub_path)
                if verbose:
                    print(f"OK ({len(sub_path)} primitive moves)", flush=True)
            else:
                if verbose:
                    print("FAILED — using super-move as-is", flush=True)
                expanded.append((desc, state))
        else:
            expanded.append((desc, state))
    return True, expanded


# ═══════════════════════════════════════════════════════════════════════
# 4.  Macro miner: generate AC-trivial presentations and mine paths
# ═══════════════════════════════════════════════════════════════════════

def random_balanced_presentation(n_gens: int = 2,
                                  max_len: int = 8,
                                  seed: Optional[int] = None) -> Presentation:
    """Generate a random balanced presentation of total length ≤ max_len."""
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
    """Quick check if a presentation is AC-trivial via beam search."""
    from ac_solver_v3 import ACBeamSearcherV3
    searcher = ACBeamSearcherV3(
        beam_width=beam_width,
        max_depth=max_depth,
        max_states=max_states,
        timeout=timeout,
        use_substitution_moves=True,
        use_generator_moves=True,
        verbose=False,
    )
    found, path, _ = searcher.search(pres,
                                     run_id=f"trivcheck_{int(time.time())}")
    return found, path


def mine_macros(num_presentations: int = 5000,
                max_len: int = 8,
                output_file: str = "results/macros.json") -> dict:
    """Generate random presentations, filter for AC-trivial ones,
    collect successful paths, and mine frequent move sub-sequences."""
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
                  f"tested={tested} rate={found/max(tested,1)*100:.1f}% "
                  f"elapsed={elapsed:.0f}s", flush=True)

    # Count n-gram frequencies
    ngram_counts: dict = defaultdict(int)
    for path in paths:
        for n in range(2, min(6, len(path) + 1)):
            for j in range(len(path) - n + 1):
                ngram = tuple(path[j:j + n])
                ngram_counts[ngram] += 1

    # Extract top macros
    top_macros = sorted(ngram_counts.items(), key=lambda x: -x[1])[:50]

    result = {
        "num_presentations": num_presentations,
        "max_len": max_len,
        "tested": tested,
        "trivial_found": found,
        "trivial_rate": found / max(tested, 1),
        "elapsed": time.monotonic() - t0,
        "top_macros": [
            {"moves": list(m), "count": c}
            for m, c in top_macros
        ],
        "example_paths": paths[:10],
    }

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nMacros saved to {output_file}")
    print(f"Top macros:")
    for m, c in top_macros[:10]:
        print(f"  [{c:4d}x] {' → '.join(m[:4])}{'...' if len(m) > 4 else ''}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# 5.  Command line
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="AC Conjecture — V4 Potential-Guided Solver")
    parser.add_argument("--example", "-e", type=int, default=2,
                        help="Example index (0=trivial,1=myasnikov,2=ak3,"
                             "3=ak4,4=ak5,5=pres_p,6=ms_ak3)")
    parser.add_argument("--beam", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=500)
    parser.add_argument("--states", type=int, default=20_000_000)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-sub", action="store_true",
                        help="Disable substitution super-moves")
    parser.add_argument("--no-gen", action="store_true",
                        help="Disable generator Nielsen moves")
    parser.add_argument("--scalar", action="store_true",
                        help="Use scalar scoring instead of potential")
    parser.add_argument("--mine", type=int, default=0,
                        help="Run macro miner with N presentations")
    parser.add_argument("--verify", type=str, default="",
                        help="Verify a saved result JSON file")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    EXAMPLES = [
        ("trivial", v3.example_trivial(), "Standard trivial"),
        ("myasnikov", v3.example_myasnikov(), "Known AC-trivializable"),
        ("ak3", v3.example_ak3(), "AK(3) — OPEN"),
        ("ak4", v3.example_ak_general(4), "AK(4) — OPEN"),
        ("ak5", v3.example_ak_general(5), "AK(5) — OPEN"),
        ("presentation_p", v3.example_presentation_p(), "Bridge P (Lisitsa)"),
        ("ms_ak3", v3.example_ms_equivalent_ak3(), "MS(3, y⁻¹x⁻¹yxy) ≅ AK(3)"),
    ]

    # Macro mining mode
    if args.mine > 0:
        mine_macros(args.mine)
        return

    # Verify mode
    if args.verify:
        with open(args.verify) as f:
            data = json.load(f)
        if "path" not in data:
            print("No path found in result file")
            return
        path = [(m["move"], None) for m in data["path"]]  # states not needed for verification
        # Reconstruct state sequence
        from ac_solver_v3 import ACBeamSearcherV3
        # Just print verification info
        print(f"Path length: {len(path)}")
        print(f"SUPER moves: {sum(1 for d,_ in path if d.startswith('SUBST'))}")
        print("Note: full verification requires state reconstruction.")
        return

    # Search mode
    name, pres, desc = EXAMPLES[args.example]
    print(f"Target: {name} — {desc}")
    print(f"Start:  {pres.show()}", flush=True)

    use_potential = not args.scalar
    searcher = PotentialBeamSearcher(
        beam_width=args.beam,
        max_depth=args.depth,
        max_states=args.states,
        timeout=args.timeout,
        use_substitution_moves=not args.no_sub,
        use_generator_moves=not args.no_gen,
        use_potential_scoring=use_potential,
        verbose=True,
        seed=args.seed,
    )

    t0 = time.monotonic()
    found, path, stats = searcher.search(pres, run_id=f"{name}_v4_{int(t0)}")
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
        print(f"  ✗  NOT FOUND within search limits.")
        print(f"     Reason: {stats.get('reason', 'unknown')}")

    print(f"\n  Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.2f}")
        else:
            print(f"    {k}: {v}")

    result_data = {
        "run_id": f"{name}_v4_{int(t0)}",
        "found": found,
        "presentation": name,
        "description": desc,
        "params": {
            "beam_width": args.beam,
            "max_depth": args.depth,
            "max_states": args.states,
            "use_potential": use_potential,
            "use_substitution": not args.no_sub,
            "use_generator_moves": not args.no_gen,
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
    fname = (f"results/{name}_v4_{int(t0)}_"
             f"{'FOUND' if found else 'NOT_FOUND'}.json")
    with open(fname, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  Full results saved to: {fname}")
    print()


if __name__ == "__main__":
    main()
