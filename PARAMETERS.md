# Parameters of the V0 solvers

Reference for `MHKP-V0-BS.py` (beam search), `MHKP-V0-SA.py` (simulated
annealing) and `MHKP-V0-HC.py` (hill climbing), all solving submodel V0 of
Deplano et al. (2019). Measured values are from 3 s runs on the `MHKP/t8_n*`
corpus; the objective throughout is formula (1), the priority-weighted wasted
space, where **lower is better**.

---

## 1. Beam search — `MHKP-V0-BS.py`

### Search parameters

| symbol | code | default | meaning |
|---|---|---|---|
| `w` | `--width` / `BEAM_WIDTH` | 12 | **Beam width.** How many nodes survive each level. After every level all candidate children are sorted by rollout score and the best `w` are kept; the rest are discarded permanently. |
| `m` | `--actions` / `MAX_ACTIONS` | 60 | **Maximum actions per node.** Upper bound on the children one node may generate. Caps the branching factor so a single node cannot crowd out the level. |
| `T^max` | `--time-limit` | 3.0 s | **Wall-clock budget.** Checked at the top of each level and inside both expansion loops, so the search can stop mid-level and still return a complete packing. |
| — | `--seed` | 1 | Seed for the volume-weighted roulette in `Expand`. The only stochastic element; everything else is deterministic. |
| — | `--stability-rule` | `corners` | Which reading of constraints (10l)–(10o) to enforce. See §4. |

### Where the defaults come from

`w = 12` and `m = 60` are the values Xu et al. (2026) select by grid search in
their Section 5.2. They are carried over because the two parameters trade the
same quantities here as there, not because their instances match ours.

Measured sensitivity (`t8_n50`, 5 instances, 3 s):

| w | m | objective | levels reached | rollouts |
|---|---|---|---|---|
| 1 | 60 | 0.1023 | 13.0 | 372 |
| 4 | 60 | **0.0859** | 14.4 | 1398 |
| 12 | 60 | 0.0896 | 4.8 | 1918 |
| 24 | 60 | 0.0906 | 3.0 | 1823 |
| 12 | 12 | 0.1005 | 13.6 | 1032 |
| 12 | 30 | 0.0915 | 10.6 | 2145 |
| 12 | 120 | **0.0886** | 3.4 | 1545 |

Two things this shows:

* **`w = 1` is clearly bad** (0.1023): with a single surviving node the method
  degenerates to a greedy construction with a rollout, and loses the diversity
  that makes beam search worth the cost.
* **Beyond that the curve is flat.** Across sizes the ordering of `w = 4`,
  `12` and `24` changes with `n` (`w = 4` best at 50 and 90, `w = 12` best at
  30, `w = 24` best at 18), so those differences are within noise at five
  instances per point. `w = 12` is a defensible default rather than a tuned
  optimum; a proper grid search on this corpus would need many more instances
  per cell.

### The width/depth trade-off

`w` and `m` both consume the same fixed budget of rollouts, and each rollout is
the expensive operation. Raising either buys breadth at the cost of **depth**:
at `w = 1` the search reaches 13 levels, at `w = 24` only 3. Since each level
commits exactly one more item, depth is how many items the search places
deliberately rather than leaving to the greedy rollout.

### Derived quantity

| symbol | value | meaning |
|---|---|---|
| `⌊m / \|O\|⌋` | `max(1, m // 2)` = 30 | Items drawn per space by the roulette. `\|O\| = 2` here because V0 allows only two rotations; Xu et al. divide by 6 because they have six orientations. |

---

## 2. Instance parameters (not tuned — properties of the problem)

| symbol | source | meaning |
|---|---|---|
| `p_j` | instance file | **Bin priority**, formula (2). Set to `1 / V_j`, so filling a small bin pays as much as filling a large one. Determines the bin order `κ`. |
| `β` | derived | **Grid size**, Definition 4: the gcd of all item sides. Item positions must be multiples of `β`. `β = 1` means free positioning, which is the usual case on our instances. |
| `O` | model | **Rotation set.** Two orthogonal rotations about the vertical axis (Deplano Table 2): length and width may swap, height is fixed. |
| `L_j, W_j, H_j` | instance file | Bin dimensions: length (Z), width (X), height (Y). |
| `V_j` | derived | Bin volume `L_j · W_j · H_j`. |

---

## 3. Objective

Formula (1) is minimised:

```
f_(1) = Σ_j p_j · (V_j − packed_j)
```

The solvers internally maximise the normalised complement

```
f = Σ_j p_j · packed_j / Σ_j p_j · V_j   ∈ [0, 1]
```

which is a strictly decreasing affine function of `f_(1)`, so it induces the
same ranking and the same optimum. With one bin and `p_j = 1/V_j`, `f_(1)` is
exactly the **wasted fraction**, so `f_(1) = 0.3090` means the bin is 69.10 %
full. This is the unit the paper's Tables 8 and 11 report.

---

## 4. Stability rule — shared by all three solvers

`STABILITY_RULE` in `MHKP-V0-SA.py`, exposed as `--stability-rule`:

| value | rule | source |
|---|---|---|
| `corners` (default) | at least **two of the four** base corners lie within a supporting top face | the paper's **prose**: "at least two bottom corners of k should be on the top surface of i" |
| `peraxis` | the footprints **overlap along x and along y** | what constraints **(10d)–(10k) literally encode**; one corner can satisfy both axis pairs |

The paper does not disambiguate these, and they differ: a box overhanging so
that only one corner is covered passes `peraxis` but fails `corners`.
`mhkp_v0_milp.py` implements the constraints as written and so realises
`peraxis`. **Any MILP-vs-heuristic comparison must set both sides to the same
rule**, or it compares two different problems.

---

## 5. Simulated annealing — `MHKP-V0-SA.py`, for contrast

| symbol | code | default | meaning |
|---|---|---|---|
| `T_0` | `T_INIT` | 0.02 | Initial temperature. Meaningful as a fixed number because the objective is a ratio in [0, 1] on every instance. |
| `α` | `ALPHA` | 0.9995 | Geometric cooling factor, applied every iteration. |
| `T_min` | `T_MIN` | 1e-6 | Temperature floor. |
| `β` | `REHEAT_FRACTION` | 0.25 | Reheat target as a fraction of `T_0`. A reheat resets `T`, it does not scale the annealed value. |
| `σ_max` | `REHEAT_THRESHOLD` | `max(500, 0.75 × sweep)` | Iterations without a new best before reheating plus diversification. |

## 6. Hill climbing — `MHKP-V0-HC.py`, for contrast

| symbol | code | default | meaning |
|---|---|---|---|
| — | `ACCEPT_EQUAL` | `True` | Accept sideways (equal-objective) moves, to traverse the decoder's broad plateaus. Measured: 0.0950 → 0.0846 at n = 50. |
| — | `STALL_SCALE` / `STALL_FLOOR` | 80000 / 600 | Stall limit `max(600, 80000 / n)`, scaled by `1/n` because decode cost grows with `n`. |
| — | `KICK_FROM_BEST` | `True` | On a stall, perturb the incumbent best rather than the current solution, making the method an iterated local search. |

Hill climbing has **no temperature and no schedule**, which removes the
parameter that is most error-prone when porting SA between problems.
