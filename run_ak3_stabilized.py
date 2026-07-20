#!/usr/bin/env python3
"""Targeted AK(3) search forcing stabilization first, then exploring in 3-gen space."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ac_solver_v3 import Presentation, ACBeamSearcherV3

ak3 = Presentation(2, [[1,1,1,2,2,2,2], [1,2,1,-2,-1,-2]])
print("AK(3) start:", ak3.show(), flush=True)

# Stabilize immediately to get into 3-gen space
stable = ak3.stab_add()
print("Stabilized (3-gen):", stable.show(), flush=True)

s = ACBeamSearcherV3(
    beam_width=1000,
    max_depth=300,
    max_states=20_000_000,
    timeout=3600,
    use_substitution_moves=False,  # only works in 2-gen
    use_generator_moves=True,
    restart_patience=20,
    restart_fraction=0.3,
    verbose=True,
)
found, path, stats = s.search(stable, run_id="ak3_stab_v3")
print("FOUND" if found else "NOT FOUND", flush=True)
print(stats, flush=True)
if found and path:
    for i, (desc, pres) in enumerate(path):
        print(f"  [{i+1:3d}] {desc:40s} -> {pres.show()}", flush=True)
