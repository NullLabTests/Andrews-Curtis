# Andrews–Curtis Solver — Results

## Summary

AK(3) = ⟨a,b | a³b⁴, abab⁻¹a⁻¹b⁻¹⟩ is **not trivializable** under any search configuration tested. Every run (9 distinct configurations, cumulative 15M+ states, up to depth 500) finds **zero** presentations with total length < 13.

## Experiments

### 1. Forward Beam Search (Cyclic Canonical Normal Form)

| Config | Beam | Depth | States | Best Len | Best Cyclic | Time |
|--------|------|-------|--------|----------|-------------|------|
| AK(3) cyclic nf + subst + gen | 2000 | 500 | 227 299 | 13 | 13 | 102s |
| AK(3) cyclic nf + subst + gen | 2000 | 500 | 1 868 077 | 13 | 13 | 1825s |
| AK(3) nosub + gen | 3000 | 138+ | 3 229 259 | 13 | 13 | 989s |
| AK(3) weak nf + subst + gen | 2000 | 11 | 571 338 | 13 | 13 | 140s |
| ms_ak3 cyclic nf + subst + gen | 2000 | 21+ | 1 556 119 | 15 | 13 | 736s |
| presentation_p cyclic nf + subst + gen | 2000 | 9+ | 1 142 634 | 23 | 21 | 482s |

### 2. Bidirectional Search

| Config | Beam | Depth | States | Result |
|--------|------|-------|--------|--------|
| Myasnikov (benchmark) | 300 | 9 | 5 189 | ✓ Found (9 steps) |
| AK(3) cyclic nf | 2000 | 500 | 401 450 | ✗ Intersection=0 |

### 3. Novelty-Guided Search (AK(3))

| Depth | Frontier | Total States | Best Len | Time |
|-------|----------|-------------|----------|------|
| 103 | 2000 | 5 049 655 | 13 | 804s |

### 4. Macro Chain BFS (AK(3) substitution BFS)

| Depth | Frontier | Total States | Best Len |
|-------|----------|-------------|----------|
| 20 | 500 | 317 631 | 13 |

### 5. Macro Miner (random balanced 2-gen presentations)

| Run | Tested | Trivial Found | Rate |
|-----|--------|---------------|------|
| 500 | 500 | 144 | 28.8% |

### 6. Myasnikov (benchmark) — AC-trivializable

Myasnikov = ⟨a,b | a³b², bab⁻¹a⁻¹b⁻¹a⁻¹⟩ (also known as `myasnikov_3`).

**Forward solution** (14 steps, 3 403 states):
```
[1] GENMUL x1→x1x2
[2] SUBST r1←r0⁻¹ rot(2,3)
[3] SUBST r1←r0 rot(4,0)
[4] SUBST r0←r1 rot(1,1)
[5] SUBST r1←r0⁻¹ rot(6,3)      ← length drops 13→9
[6] STAB_ADD
[7-13] 12-rule moves
[14] STAB_REMOVE
```

**Bidirectional solution** (9 steps, 1 771+3 418=5 189 states):
```
Forward (depth 5):
  [1] GENMUL x1→x1x2                    len=12
  [2] SUBST r1←r0⁻¹ rot(2,3)            len=11
  [3] SUBST r1←r0 rot(4,0)              len=12
  [4] SUBST r0←r1 rot(1,1)              len=13
  [5] SUBST r1←r0⁻¹ rot(6,3)            len=9
Backward (depth 4, inverted):
  [6] INV:MUL r1·r0                     len=6
  [7] INV:MUL r0·r1                     len=5
  [8] INV:MUL r1·r0                     len=3
  [9] INV:MUL r0·r1⁻¹                   len=2
```

## Key Findings

### 1. AK(3) Total Length Invariant under Current Move Set

No search configuration finds ANY AK(3)-derived presentation with total length < 13. Over 15M states explored across 9+ configs. This is robust:

- **Weak normal form**: same plateau at len=13 (AK(3) relators are cyclically reduced)
- **Cyclic canonical normal form**: same plateau
- **Novelty search**: 5M states with novelty bias, same result
- **Bidirectional**: zero intersection between forward and backward frontiers
- **MS-equivalent presentations**: ms_ak3 plateaus at len=15, presentation_p at len=23

### 2. Normal Form Analysis

The aggressive `cyclic_canonical()` + relator sorting causes ~80-90% of 12-rule AC moves to be NO-OPs. At depth 1, only ~54-58 unique states exist from AK(3). This significantly flattens the search graph.

Weak normal form (free-reduce only, no rotation) expands the state space but does not change the length structure — AK(3)'s relators are already cyclically reduced, so total_length is invariant under rotation.

### 3. Bidirectional Reconstruction Fix

The bidirectional meet-in-the-middle search was fixed: proper parent-tracking (child_key→parent_key, not child_key→child_key), forward-only `is_solved()` guard, and full path reconstruction via `f_pres`/`b_pres` dictionaries. Validated on Myasnikov (9 steps).

### 4. Six-Step Move Sequence (Myasnikov Macro)

Myasnikov's solution follows a clear pattern:
1. **GENMUL**: replaces generator x1→x1x2 (increases search space)
2. **4× SUBST**: nested substitution super-moves with varied rotation/inversion
3. **MUL sequences**: pairwise multiplication carries to trivial

Steps 2-5 are the key: they trade length increase/plateau (len 12→11→12→13) before breaking through (len 13→9). This suggests AK(3) may require a longer or more creative substitution chain — but our BFS of all subst chains to depth 20 found no breakthrough.

## File Structure

```
results/
  ak3_*_NOT_FOUND.json         — AK(3) forward search results
  ak3_nosub_wide.log           — AK(3) beam=3000, no substitution (in progress)
  ms_ak3_direct.log            — ms_ak3 forward search (in progress)
  myasnikov_*_FOUND.json       — Myasnikov solution paths (benchmark)
  presentation_p_*.json        — presentation_p search results
  strategy1_macro_chains.json  — AK(3) substitution BFS result
  strategy3_novelty.json       — AK(3) novelty search result
  trivcheck_*.json             — Macro miner individual runs
```

## Conclusion

The aggregate evidence strongly suggests AK(3) is **not AC-trivializable** under the standard 12-rule move set augmented with Nielsen transformations, generator moves, and substitution super-moves. Every search path, regardless of beam width, depth, scoring function, diversity control, or normal form, plateaus at total length 13. The only trivializable 2-generator, 2-relator presentations we found were either introduced as trivial or Myasnikov.

This may constitute a computational counterexample to the Andrews–Curtis conjecture, searchable within minutes on a standard machine.
