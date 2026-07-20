#!/usr/bin/env python3
"""GA / random-walk solver for the Andrews–Curtis conjecture.

Uses a population of random AC paths evolved by mutation (extending paths
with random valid moves) and evaluated by a fitness function that rewards
progress toward trivialization.
"""

import sys, os, time, json, random
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))
from ac_solver_v3 import (
    Presentation, enumerate_moves_12rule, enumerate_moves_full,
    free_reduce, invert, cyclic_rotate, cyclic_canonical,
)


@dataclass
class GAState:
    path: list[tuple[str, Presentation]] = field(default_factory=list)
    current: Optional[Presentation] = None

    def copy(self):
        return GAState(path=self.path[:], current=self.current)

    def __len__(self):
        return len(self.path)


@dataclass
class GASolver:
    pop_size: int = 1000
    max_depth: int = 200
    max_states: int = 10_000_000
    timeout: float = 3600.0
    mutation_rate: float = 0.5
    elite_fraction: float = 0.1
    verbose: bool = True
    use_substitution: bool = True
    use_generator_moves: bool = True
    results_dir: str = "results"

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)

    def _move_set(self, p: Presentation):
        if self.use_generator_moves:
            return enumerate_moves_full(p)
        return enumerate_moves_12rule(p)

    def _fitness(self, pres: Presentation) -> float:
        """Lower is better. Combines total length, cyclic length, and
        structural features."""
        if pres.is_solved():
            return -1e9
        tl = pres.total_length()
        cl = pres.total_cyclic_length()
        # bonus for having a single-letter relator (enables STAB_REMOVE)
        single_letter_bonus = 0
        for r in pres.relators:
            if len(r) == 1:
                single_letter_bonus -= 3
        # bonus for fewer distinct generators in play
        all_letters = set()
        for r in pres.relators:
            all_letters.update(abs(a) for a in r)
        gen_penalty = len(all_letters) * 2
        return tl + cl + gen_penalty + single_letter_bonus

    def _random_move(self, pres: Presentation):
        """Apply a random valid move to a presentation."""
        moves = self._move_set(pres)
        if self.use_substitution and pres.n_gens == 2:
            sub_moves = list(pres.all_substitution_moves())
            for d, r in sub_moves:
                moves.append(("SUBST_" + d, lambda p, res=r: res))
        if not moves:
            return None, None
        desc, fn = random.choice(moves)
        new_pres = fn(pres)
        if new_pres.state_key() == pres.state_key():
            return None, None
        return desc, new_pres

    def _random_extend(self, state: GAState, steps: int = 1):
        """Extend a random walk by `steps` random moves."""
        for _ in range(steps):
            if state.current is None:
                break
            desc, new_p = self._random_move(state.current)
            if desc is None:
                break
            state.path.append((desc, new_p))
            state.current = new_p
        return state

    def search(self, start: Presentation,
               run_id: Optional[str] = None) -> tuple[bool, Optional[list], dict]:
        if run_id is None:
            run_id = f"ga_{int(time.time())}"
        t0 = time.monotonic()

        # Initialize population with random walks of varying lengths
        pop: list[GAState] = []
        visited = {start.state_key(): True}

        # seed population with short random walks
        for i in range(self.pop_size):
            walk_len = random.randint(1, min(5, self.max_depth))
            s = GAState(current=start.copy())
            s = self._random_extend(s, walk_len)
            if s.current:
                sk = s.current.state_key()
                if sk not in visited:
                    visited[sk] = True
                    pop.append(s)

        best_fitness = self._fitness(start) if start else 999
        best_state = GAState(current=start.copy())
        total_states = 1
        last_report = 0.0
        no_improve = 0

        for gen in range(self.max_depth):
            now = time.monotonic()
            if now - t0 > self.timeout:
                break
            if total_states >= self.max_states:
                break

            # Evaluate fitness
            scored = [(self._fitness(s.current), s) for s in pop if s.current]
            if not scored:
                break
            scored.sort(key=lambda x: x[0])

            current_best = scored[0][0]
            if current_best < best_fitness:
                best_fitness = current_best
                best_state = scored[0][1]
                no_improve = 0
            else:
                no_improve += 1

            # Reporting
            if self.verbose and now - last_report > 10.0:
                best_p = best_state.current
                bl = best_p.total_length() if best_p else 0
                print(
                    f"  [gen={gen+1:3d}  pop={len(pop):4d}  "
                    f"states={total_states:8d}  best_len={bl:2d}  "
                    f"best_fit={best_fitness:8.1f}  "
                    f"stall={no_improve:2d}  "
                    f"elapsed={now-t0:.0f}s]",
                    flush=True,
                )
                last_report = now

            # Found?
            if best_state.current and best_state.current.is_solved():
                elapsed = time.monotonic() - t0
                return True, best_state.path, {
                    "states_explored": total_states,
                    "depth": len(best_state.path),
                    "reason": "found",
                    "elapsed": elapsed,
                }

            # Selection: keep elite + tournament
            elite_n = max(1, int(self.pop_size * self.elite_fraction))
            new_pop = [s for _, s in scored[:elite_n]]

            # Fill rest with mutated copies
            while len(new_pop) < self.pop_size:
                parent = random.choice(scored[:max(10, len(scored)//2)])[1]
                child = parent.copy()
                child = self._random_extend(child,
                                            random.randint(1, min(5, self.max_depth - len(child))))
                if child.current and child.current.state_key() not in visited:
                    visited[child.current.state_key()] = True
                    total_states += 1
                    new_pop.append(child)

            # Fresh blood: inject completely new random walks
            blood_n = max(1, self.pop_size // 20)
            for _ in range(blood_n):
                walk_len = random.randint(1, 10)
                s = GAState(current=start.copy())
                s = self._random_extend(s, walk_len)
                if s.current and s.current.state_key() not in visited:
                    visited[s.current.state_key()] = True
                    total_states += 1
                    new_pop.append(s)

            pop = new_pop[:self.pop_size]

            # Restart if stalled
            if no_improve >= 30:
                if self.verbose:
                    print(f"  ⚡ GA RESTART (stalled {no_improve} gens)", flush=True)
                # keep elite, replace rest with fresh random walks
                pop = [s for _, s in scored[:elite_n]]
                while len(pop) < self.pop_size:
                    walk_len = random.randint(1, 15)
                    s = GAState(current=start.copy())
                    s = self._random_extend(s, walk_len)
                    if s.current:
                        pop.append(s)
                no_improve = 0

        elapsed = time.monotonic() - t0
        best_p = best_state.current
        return False, best_state.path if best_state.current else None, {
            "states_explored": total_states,
            "depth": len(best_state.path) if best_state.current else 0,
            "best_length": best_p.total_length() if best_p else 0,
            "reason": "ga exhausted",
            "elapsed": elapsed,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=200)
    parser.add_argument("--states", type=int, default=5_000_000)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--no-sub", action="store_true")
    parser.add_argument("--no-gen", action="store_true")
    parser.add_argument("--example", type=int, default=2)
    args = parser.parse_args()

    examples = {
        0: ("trivial", Presentation(2, [[1], [2]])),
        1: ("myasnikov", Presentation(2, [[1,1,-2,-2,-2], [1,2,1,-2,-1,-2]])),
        2: ("ak3", Presentation(2, [[1,1,1,2,2,2,2], [1,2,1,-2,-1,-2]])),
        5: ("presentation_p", Presentation(2,
            [[-1,-2,1,-2,-1,2,1,-2,-2,1,2,2],
             [-2,-1,2,2,-1,-2,-1,2,1,-2,1]])),
        6: ("ms_ak3", Presentation(2, [[-1,2,2,2,1,-2,-2,-2,-2],
                                        [-1,-2,-1,2,1,2]])),
    }
    name, start = examples.get(args.example, examples[2])
    print(f"Target: {name}")
    print(f"Start:  {start.show()}", flush=True)

    solver = GASolver(
        pop_size=args.beam,
        max_depth=args.depth,
        max_states=args.states,
        timeout=args.timeout,
        use_substitution=not args.no_sub,
        use_generator_moves=not args.no_gen,
    )
    found, path, stats = solver.search(start, run_id=f"ga_{name}")
    print(f"\n{'✓ FOUND' if found else '✗ NOT FOUND'}")
    print(stats)
    if found and path:
        for i, (desc, pres) in enumerate(path):
            print(f"  [{i+1:3d}] {desc:40s} -> {pres.show()}")
    elif path:
        print("Best path found:")
        for i, (desc, pres) in enumerate(path):
            print(f"  [{i+1:3d}] {desc:40s} -> {pres.show()}")
