#!/usr/bin/env python3
"""Non-interactive AK(3) run — maximum search, saves results."""

import sys, time, json, os
sys.path.insert(0, os.path.dirname(__file__))

from ac_solver_pro import (
    Presentation, example_ak3, example_ak_general, ACBeamSearcher, ACGeneticSearcher
)

OUTPUT_DIR = "/workspaces/Andrews-Curtis/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_ID = f"ak3_beam_{int(time.time())}"

def save_result(found, path, stats, run_id):
    ltrs = ["a", "b"]
    data = {
        "run_id": run_id,
        "found": found,
        "path_len": len(path) if path else 0,
        "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float, str))},
        "path": [
            {"move": desc, "state": pres.show(ltrs), "len": pres.total_length()}
            for desc, pres in (path or [])
        ],
    }
    status = "FOUND" if found else "NOT_FOUND"
    fname = f"{OUTPUT_DIR}/{run_id}_{status}.json"
    with open(fname, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {fname}")
    return fname

# ══ Phase 1: Beam search with super-moves ══
print("=" * 70)
print("  PHASE 1: Beam search on AK(3)")
print("  Max search — super-moves + generator Nielsen moves")
print("=" * 70)

pres = example_ak3()
print(f"  Target: AK(3)  len={pres.total_length()}")
print(f"  Start:  {pres.show(['a','b'])}")

searcher = ACBeamSearcher(
    beam_width=500,
    max_depth=200,
    max_states=10_000_000,
    timeout=7200,
    use_super_moves=True,
    use_generator_moves=True,
    verbose=True,
)

print("\n" + "─" * 70)
found, path, stats = searcher.search(pres)
save_result(found, path, stats, RUN_ID)

if not found:
    print("\n  Phase 1 did not find trivialization. Starting Phase 2 (expanded beam)...")
    print("=" * 70)

    # ══ Phase 2: Wider beam, deeper ══
    RUN_ID2 = f"ak3_beam_wide_{int(time.time())}"
    searcher2 = ACBeamSearcher(
        beam_width=1000,
        max_depth=100,
        max_states=5_000_000,
        timeout=7200,
        use_super_moves=True,
        use_generator_moves=True,
        verbose=True,
    )
    found2, path2, stats2 = searcher2.search(pres)
    save_result(found2, path2, stats2, RUN_ID2)

    if not found2:
        print("\n  Phase 2 also failed. Starting Phase 3 (GA)...")
        print("=" * 70)

        # ══ Phase 3: GA ══
        RUN_ID3 = f"ak3_ga_{int(time.time())}"
        searcher3 = ACGeneticSearcher(
            pop_size=500,
            genome_length=100,
            generations=500,
            timeout=7200,
            verbose=True,
        )
        found3, path3, stats3 = searcher3.search(pres)
        save_result(found3, path3, stats3, RUN_ID3)

        if not found3:
            print("\n  All phases exhausted. AK(3) remains unsolved.")
            print("  This is consistent with it being a genuine potential counterexample.")

print("\n  Done.")
