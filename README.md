<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/yboulaamane/sorbent/main/docs/brand/logo-dark.svg">
    <img alt="Sorbent — compound triage" src="https://raw.githubusercontent.com/yboulaamane/sorbent/main/docs/brand/logo-light.svg" width="260">
  </picture>
</p>

<p align="center">
  <em>A sorbent is the phase that holds on to what you are after<br>
  while the rest of the mixture flows past.</em>
</p>

---

**A compound triage service.** Submit a SMILES library, get back a ranked,
deduplicated, liability-flagged shortlist.

Everything Sorbent reports is **deterministic and citable**. Descriptors come
from RDKit, the rule sets are the published literature definitions, the
structural alerts are RDKit's bundled catalogs, and the composite score is a
transparent weighted sum that ships its own breakdown with every molecule.
Nothing here is an affinity, an activity, or a probability of success.

That constraint is the point. A triage tool is useful in proportion to how much
you can trust it, and every number here is reproducible from the input
structure plus a named citation.

One honest caveat, because the distinction is easy to blur: **`clogp` is a
fitted model**, not graph arithmetic. Wildman & Crippen regressed atom-type
contributions against experimental logP, so it is an *estimate* of a physical
property — paracetamol computes 1.35 against an experimental 0.46. It is
deterministic, reproducible and citable, and it is still a prediction. Every
other descriptor is exact given the structure.

---

## Status

**Complete and working.** All 221 tests pass; `ruff`, `ruff format` and `mypy`
are clean.

```bash
make install
make test        # 221 passed
make run         # http://localhost:8000/docs
```

A real run, 5,250 records uploaded as a `.smi` file, finished in 9.5 s across
three chunks on the process pool:

```
submitted            5250
parsed               5200
parse_failed           50
duplicates_removed   2112     <- 40% of the library was redundant
retained             3088
clusters             2617
```

and the report that comes back ranks them:

```
 # id                score     MW  cl  rep    rules  alerts
 1 diazepam          0.928  284.7   4  yes       YY  clean
 2 ibuprofen         0.878  206.3   6  yes       YY  clean
 3 salicylic         0.867  138.1   7  yes       YY  clean
 4 caffeine          0.840  194.2   5  yes       YY  clean
 5 aspirin           0.715  180.2   8  yes       YY  phenol_ester
 6 paracetamol       0.692  151.2   1  yes       YY  hydroquinone
 8 catechol          0.607  110.1   3  yes       YY  catechol_A(92), catechol
 9 erythromycin      0.455  733.9   0  yes       NN  clean
10 aspirin_sodium        —      —   —    —        —  duplicate of aspirin
11 broken                —      —   —    —        —  could not parse SMILES 'C((('
```

Every number there is reproducible from the structure. The one caveat is
`clogp`, and the score has a sharp edge worth reading before you trust the
ranking — see [how the score works](#how-the-score-works-and-its-sharp-edge).

---

## Quickstart

```bash
make install
make run          # http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/v1/jobs \
  -H 'content-type: application/json' \
  -d '{"name":"vendor-plate-1","smiles":["CC(=O)Oc1ccccc1C(=O)O","Cn1cnc2c1c(=O)n(C)c(=O)n2C"]}'
```

```bash
curl -F 'file=@library.smi' -F 'name=molport-subset' localhost:8000/v1/jobs/upload
```

---

## The API

| | |
|---|---|
| `GET /health` | Liveness. Never touches RDKit — a health check that does real work fails exactly when you need it honest. |
| `GET /ready` | Readiness. 503 if the chem layer will not import. |
| `POST /v1/jobs` | Submit inline. **202** + `Location`, never 200 — the work has not happened yet. |
| `POST /v1/jobs/upload` | Submit a `.smi` or `.csv`, multipart. |
| `GET /v1/jobs` | List, newest first. |
| `GET /v1/jobs/{id}` | Status, phase, progress, and reconciling counts. Cheap; poll it. |
| `GET /v1/jobs/{id}/results` | Paginated. `next_offset` is `null` at the end, so clients loop rather than compare counts. |
| `GET /v1/jobs/{id}/results.ndjson` | Streamed, one object per line. Neither side holds 200k records in memory. |
| `DELETE /v1/jobs/{id}` | Cancel if running, delete if finished. |
| `POST /v1/molecules/describe` | Synchronous, capped at 100. Over the cap you get a 413 pointing at `/v1/jobs`. |
| `POST /v1/molecules/standardize` | Salt-strip, neutralise, canonicalise. |

---

## Why it is built this way

Three decisions carry the design. Each is commented where it lives.

### 1. A process pool, not `BackgroundTasks`

RDKit is CPU-bound C++ called from Python. An endpoint that parses 50k
molecules inline blocks the event loop for the whole duration — not just for
that caller, for **every connection the process is serving**. The health check
times out; the load balancer pulls the node.

`BackgroundTasks` does not fix this. It runs the work on the same event loop
after the response is sent, so the blocking is deferred, not removed. A thread
pool is a partial fix — RDKit releases the GIL inside its C++ calls — but the
Python glue between them does not.

So: a `ProcessPoolExecutor` driven from an async supervisor
([`jobs/runner.py`](src/sorbent/jobs/runner.py)). The parent stays responsive,
workers do the chemistry, results come back over a pickle boundary — and that
boundary is what shapes the next decision.

The pool uses the **spawn** start method, not Linux's default fork: forking a
process that already has threads (any ASGI server) can deadlock in the child,
and CPython changes this default in 3.14.

### 2. A two-phase pipeline

[`chem/pipeline.py`](src/sorbent/chem/pipeline.py) splits into:

- **`process_chunk`** — embarrassingly parallel. Everything depending on one
  molecule only: parse, standardise, descriptors, rules, alerts, scaffold,
  fingerprint. Runs in a worker.
- **`finalize`** — inherently global. Deduplication, clustering, ranking,
  top-N. Runs once, after all chunks land.

Anything that *looks* per-molecule but needs library context belongs in phase
two — a worker that assigns a cluster id has guessed, because it cannot see the
other chunks. Getting this boundary wrong is the standard way these pipelines
turn subtly non-deterministic under parallelism, which is why
`test_finalize_is_order_independent` exists.

### 3. Storage behind an interface

[`jobs/store.py`](src/sorbent/jobs/store.py) defines an abstract `JobStore`; the
in-memory implementation is enough to develop against but dies with the process
and is not shared between uvicorn workers. Write a `RedisJobStore` against the
same interface and change one line in the lifespan — the routes never learn
which one they are talking to. Every method is `async` even where the in-memory
one never awaits, precisely so that swap costs nothing.

---

## The science layer

Nine modules under [`src/sorbent/chem/`](src/sorbent/chem/), each documenting the
algorithm, the RDKit calls and the traps found while building it:

| Module | What it does |
|---|---|
| [`parse.py`](src/sorbent/chem/parse.py) | SMILES → sanitised, standardised molecule + InChIKey |
| [`descriptors.py`](src/sorbent/chem/descriptors.py) | MW, clogP, TPSA, HBD/HBA, rotatable bonds, Fsp3, stereo |
| [`rules.py`](src/sorbent/chem/rules.py) | Lipinski, Veber, Egan, Ghose, lead-like, Ro3 |
| [`alerts.py`](src/sorbent/chem/alerts.py) | PAINS / BRENK / NIH / ZINC via RDKit `FilterCatalog` |
| [`scaffolds.py`](src/sorbent/chem/scaffolds.py) | Bemis–Murcko, plain and generic |
| [`fingerprints.py`](src/sorbent/chem/fingerprints.py) | ECFP4 and Tanimoto similarity |
| [`cluster.py`](src/sorbent/chem/cluster.py) | Butina clustering, capped and refusing clearly above it |
| [`score.py`](src/sorbent/chem/score.py) | Composite score and its per-component breakdown |
| [`pipeline.py`](src/sorbent/chem/pipeline.py) | Composes the rest across the two phases |

```bash
make test-chem                             # the science layer
.venv/bin/pytest -m chem -k lipinski -x    # one function at a time
```

The contract tests are not tautologies — they encode real properties. Aspirin
must be closer to salicylic acid than to caffeine; standardising twice must
give the same answer as once; the counts must reconcile
(`parsed + parse_failed == submitted`).

### Traps worth knowing about

RDKit's `TautomerEnumerator` **strips defined sp3 stereochemistry by default**.
`CleanupParameters.tautomerRemoveSp3Stereo` is `True` out of the box, and it
removes stereo from any centre adjacent to a tautomerisable system — which is
the alpha carbon of every amino acid:

```
C[C@H](N)C(=O)O                          ->  CC(N)C(=O)O
C[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)O   ->  CC(N)C(=O)NC(Cc1ccccc1)C(=O)O
```

RDKit is not exactly wrong — such a centre is epimerisable in principle — but
silently racemising a peptidomimetic library is data loss, not triage.
`parse.py` sets the flag `False`; `test_standardisation_preserves_defined_stereo`
pins it.

Two smaller ones, both pinned by tests: `MolFromSmiles("")` returns an **empty
`Mol`, not `None`**, and `"*"` / `"[*]"` (R-group placeholders, common in vendor
SMILES columns) parse and sanitise cleanly, then report MW 0.

A third, in `rules.py`: **Ghose's atom count includes hydrogens.** Reading it
as heavy atoms — easy to do, and common in the wild — inverts the filter. The
two published criteria pin each other: a 160 Da molecule with 20 heavy atoms
would need an average heavy-atom mass of 8 Da, lighter than carbon. Measured:

| | heavy atoms | total atoms | Ghose on heavy | Ghose on total |
|---|---|---|---|---|
| aspirin | 13 | 21 | reject | **accept** |
| caffeine | 14 | 24 | reject | **accept** |
| atorvastatin | 41 | 76 | accept | **reject** |
| erythromycin | 51 | 118 | accept | **reject** |

The right-hand column is the drug-like set. This is why `descriptors.py`
carries `total_atoms` alongside `heavy_atoms`.

And of course **Lipinski permits one violation** — a strict four-of-four check
is the classic wrong implementation. `LIPINSKI_ALLOWED_VIOLATIONS` is a named
constant rather than a buried comparison.

Two in `alerts.py`. **PAINS is exactly the union of PAINS_A/B/C** (480 = 16 +
55 + 409), so requesting `[PAINS, PAINS_B]` reports catechol twice for one
liability — and since `n_alerts` feeds `alert_penalty`, the molecule gets
penalised twice. `find_alerts` deduplicates on (filter set, name, atoms), which
collapses that while keeping genuinely independent flags: PAINS calls catechol
`catechol_A(92)` and BRENK calls it `catechol`, and two catalogs agreeing is
worth seeing. Relatedly, an entry's *name* does not name its family —
`catechol_A(92)` belongs to `FilterSet='PAINS_B'`, the `_A` being Baell's own
numbering — so the reported `catalog` is the entry's own family, not the one
requested.

And **`GetFilterMatches` returns one match per entry**, not every occurrence.
Read naively, a dinitro compound reports atoms `[0,1,2]` — one nitro of two —
so a client highlighting it shows the other as clean. `alerts.py` takes the
SMARTS off the match and re-runs it for all occurrences, giving
`[0,1,2,9,10,11]`. Costs ~13%, paid only when an alert actually fires.

The biggest one, in `cluster.py`: **feed `Butina.ClusterData` a square matrix,
not the flat lower triangle.** `ClusterData` is pure Python, and given a 1D
triangle it starts by doing this:

```python
dist_matrix = np.zeros((nPts, nPts))  # full n×n, float64
idx = np.tril_indices(nPts, -1)  # two int64 arrays of n(n-1)/2
```

So a compact float32 triangle is expanded into a float64 square *plus* index
arrays twice its size. Hand it a correctly shaped n×n array and it skips all of
that, keeping your dtype. Measured peak allocation, identical clusterings from
both paths:

| n | 1D triangle | 2D square |
|---|---|---|
| 2,000 | 104 MB | 17 MB |
| 4,000 | 416 MB | 65 MB |
| 6,000 | 936 MB | 147 MB |

At the 20,000 cap that is ~7.2 GB against 1.6 GB. None of this is visible from
the documented API — only from RDKit's source.

Two in `fingerprints.py`, one of them compounding the other. **ECFP4 is radius
2** — the number in the name is the diameter. And `GetMorganGenerator`'s own
default radius is **3**, so omitting the parameter silently gives you ECFP6
rather than the documented ECFP4. Always pass it.

**Bulk similarity is worth 17×** — `BulkTanimotoSimilarity` over 12,000
fingerprints takes 0.9 ms against 16.4 ms for the Python comprehension. On a
clustering pass that is the whole job.

One in `scaffolds.py`: **scaffold SMILES are written without stereochemistry.**
Kept, nicotine's two enantiomers scaffold to `c1cncc([C@@H]2CCCN2)c1` and
`c1cncc([C@H]2CCCN2)c1` and land in different groups — which defeats the point
of a scaffold as a chemotype key. RDKit's own `MurckoScaffoldSmiles` defaults
to `includeChirality=False` for the same reason. Stereo is preserved on
`standard_smiles`, so nothing is lost, only moved to where it belongs.

Worth knowing rather than a trap: **exocyclic double bonds on ring atoms are
retained**, so `O=C1CCCCC1` scaffolds to itself rather than to `C1CCCCC1`, and
a ring ketone is a different chemotype from its parent ring. That is genuine
Bemis–Murcko behaviour. The generic form does not rescue you — it recolours the
exocyclic oxygen to carbon (`CC1CCCCC1`) rather than dropping it. What the
generic form *does* collapse is heteroatom identity and aromaticity: benzene,
cyclohexane and pyridine all become `C1CCCCC1`.

A fifth, in `descriptors.py`: **average mass, not monoisotopic**.
`Descriptors.MolWt` gives 180.159 for aspirin, `ExactMolWt` gives 180.042.
Drug-likeness rules are written against the average. And HBD/HBA use RDKit's
refined SMARTS rather than Lipinski's literal "count all N and O", which for
aspirin is 3 acceptors rather than 4 — the standard choice, but a borderline
compound can disagree with a tool that took the paper literally.

### Throughput

Measured on this machine, per core:

| stage | mol/s/core | 200k library, 8 cores |
|---|---|---|
| parse + standardise, with canonical tautomer | ~780 | ~32 s |
| parse + standardise, `canonical_tautomer=False` | ~2400 | ~11 s |
| descriptors | ~3200 | ~8 s |
| rules (no RDKit needed) | ~106k | <1 s |
| alerts, PAINS + BRENK (catalogs cached) | ~1550 | ~16 s |
| alerts, rebuilding catalogs per molecule | ~57 | ~7 min |
| Murcko scaffolds | ~14800 | ~2 s |
| generic scaffolds | ~6100 | ~4 s |
| ECFP4 fingerprints | ~106k | <1 s |

Butina clustering does not fit that table, because it is O(n²) and does not
scale with cores:

| n | time | peak memory |
|---|---|---|
| 6,000 | 12 s | 0.15 GB |
| 20,000 | ~140 s | 1.6 GB — `MAX_EXACT_CLUSTER_SIZE` |
| 250,000 | — | ~100 GB, i.e. never |

Above the cap `butina_cluster` refuses with an error naming the alternatives:
cluster the Murcko scaffolds instead (10–50× fewer), or use sphere exclusion
(`rdSimDivPickers.LeaderPicker`), which is O(n·k) and streams.

The tautomer pass costs about 3× and buys keto/enol forms of one compound
deduplicating against each other. It is on by default.

---

## How the score works, and its sharp edge

The composite score is a **weighted geometric mean** of four components, each
in [0, 1]:

    score = exp( sum(w_i * ln(c_i)) / sum(w_i) )

Geometric rather than arithmetic, deliberately: a component near zero should
sink the total rather than be averaged away. Under an arithmetic mean water
scored **0.821** — it passes Lipinski and Veber (both upper bounds only), trips
no alert and has no stereocentre or ring, so three of four components read 1.00
and the fourth was outvoted. Geometrically, and with the weight on the
component that actually varies, it scores **0.090**.

The weights follow from which components carry information. Measured over
15,000 real compounds, `alert_penalty` sits at 1.00 for 87.7% of them and
`complexity_penalty` for 59.1%, while `property_centrality` spreads from 0.43
to 0.96 across the deciles. Two pinned components holding most of the weight
contribute nothing but a smaller exponent on the one that varies, which is why
`property_centrality` outweighs the other two together.

**A component of exactly zero is close to a veto — which is why
`rule_compliance` is not weighted by default.** It is 0.0 whenever every
requested rule set fails, and being a fraction its resolution depends on how
many rule sets were asked for: two give `{0, 0.5, 1}`, one gives `{0, 1}` with
no middle ground. Weighted, it buried marketed drugs:

| | weighted | unweighted (default) |
|---|---|---|
| diazepam | 0.8318 | 0.7487 |
| aspirin | 0.7357 | 0.6174 |
| atorvastatin | 0.0022 | **0.1744** |
| erythromycin | 0.0007 | **0.0295** |
| ciclosporin | 0.0000 | **0.0004** |
| benzene | 0.3547 | 0.1962 |
| water | 0.2164 | 0.0902 |

It cuts both ways: dropping the component lifts molecules that fail their
rules *and lowers trivially small ones*, because water and benzene were being
handed a free 1.00 for passing rules they cannot fail. Roughly a third of
marketed oral drugs violate Ro5, every macrolide does, and most
peptidomimetics fail Veber on TPSA or rotatable bonds — none of them belong
below water.

**Rules are still computed and reported on every molecule.** Dropping the
weight drops the ranking influence, not the evidence: the report still shows
which rules a compound broke and by how much. Put `rule_compliance` back into
`score_weights` when compliance genuinely is your ranking criterion, and know
you are accepting the veto when you do.

Note that *trimming* `rule_sets` does not lift a molecule that fails the ones
that remain — `rule_compliance` is a fraction, so 0/2 and 0/1 are both 0.0.
Erythromycin scored 0.0021 under `[lipinski, veber]` and 0.0021 under
`[veber]`. Only removing the component helps.

Lipinski is also **not** in the default `rule_sets` — the default is `[veber]`
alone. Request Ro5 explicitly when it suits the chemotype, and note that a
single rule set makes `rule_compliance` binary `{0, 1}` with no partial credit
if you do weight it.

Flooring the components before the log was tried as a fix and does not work.
Measured across floors of 1e-6, 0.02, 0.05, 0.10 and 0.20, raising the floor
lifts the rule-failing drugs but lifts water *further*, because water passes
everything except `property_centrality` and so benefits from every floor:

| floor | water | erythromycin |
|---|---|---|
| 1e-6 | 0.465 | 0.002 |
| 0.05 | 0.580 | 0.195 |
| 0.20 | 0.746 | 0.416 |

At no floor does a Ro5-failing real drug outrank water. So `EPSILON` is set
just high enough to keep `ln()` finite and nothing more.

### Trivially small molecules are still best filtered, not ranked low

Weighting `property_centrality` above the components that sit at 1.00 brought
water down from 0.465 to **0.090**, below atorvastatin at 0.174 — something
flooring the components could not achieve. It is still cheaper to remove such
molecules before scoring than to rely on the ranking, with a
`descriptor_windows` minimum on `molecular_weight` — 150 to 200 is usual —
which `finalize` applies as a hard filter:

```json
{"config": {"descriptor_windows": {"molecular_weight": {"minimum": 150}}}}
```

```
1 diazepam    0.912  284.7  clean
2 caffeine    0.679  194.2  clean
3 aspirin     0.667  180.2  phenol_ester
4 benzene         —      —  filtered
5 methane         —      —  filtered
6 water           —      —  filtered
```

## Config

Everything is `SORBENT_`-prefixed; see [`.env.example`](.env.example) and
[`config.py`](src/sorbent/config.py).

| Variable | Default | |
|---|---|---|
| `SORBENT_MAX_LIBRARY_SIZE` | 250000 | Hard cap per job |
| `SORBENT_CHUNK_SIZE` | 2000 | Molecules per worker unit. Too small and pickling dominates; too large and progress goes lumpy |
| `SORBENT_WORKER_PROCESSES` | 0 | 0 → `os.cpu_count()` |
| `SORBENT_MAX_SYNC_BATCH` | 100 | Cap on the synchronous endpoints |

---

## What the report contains

The output is an audit trail, not just a shortlist — but not indiscriminate
either:

- **Parse failures, duplicates and filter drops are always reported.** These
  are findings *about your library* — that 40% of it was redundant, that 50
  records would not parse — and they are the most actionable thing the tool
  knows.
- **Retained molecules outside `top_n` / `representatives_only` are not.**
  They are not problems, merely surplus to what you asked for, and
  `counts.retained` says how many there were. Returning 200,000 records when
  you asked for the best 100 serves nobody.

So `score is not None` identifies exactly the shortlist, and everything else
carries a reason.

## Known limits

Stated rather than hidden:

- **The in-memory store dies with the process** and is not shared across
  uvicorn workers. Run one worker, or write the Redis store.
- **Butina clustering is O(n²)** and will not reach the 250k library cap.
  `MAX_EXACT_CLUSTER_SIZE` refuses the job with a clear error instead of
  running out of memory an hour in.
- **A running chunk cannot be cancelled.** `ProcessPoolExecutor` futures are
  not interruptible once started; cancelling stops further dispatch and
  discards in-flight results.
- **The composite score is a weighted sum, not a prediction.** It is for
  ordering a list. It is not evidence about any molecule.

---

## Layout

```
src/sorbent/
  main.py              app factory + lifespan (owns store and pool)
  config.py            pydantic-settings
  api/
    deps.py            dependency providers — nothing reaches for a global
    routes/            health, jobs, molecules
  schemas/             the wire contract (molecule, filters, job)
  jobs/
    store.py           JobStore ABC + in-memory implementation
    runner.py          async supervisor over the process pool
  chem/                the science layer. No FastAPI imports allowed here.
tests/
  test_api_*.py        the HTTP contract
  test_jobs_store.py   the JobStore contract, for swapping in Redis later
  test_chem_*.py       the science, 194 of the 221
```

`chem/` deliberately imports nothing from FastAPI: it must stay usable from a
worker process, a notebook, or a CLI without an app instance.

## License

MIT.
