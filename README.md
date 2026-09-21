# Winnow

**A compound triage service.** Submit a SMILES library, get back a ranked,
deduplicated, liability-flagged shortlist.

Everything Winnow reports is **computed, not predicted**. Descriptors come from
RDKit, the rule sets are the published literature definitions, the structural
alerts are RDKit's bundled catalogs, and the composite score is a transparent
weighted sum that ships its own breakdown with every molecule. There is no
trained model anywhere in this service — so nothing here is an affinity, an
activity, or a probability of success, and the API says so.

That constraint is the point. A triage tool is useful in proportion to how much
you can trust it, and every number here is reproducible from the input
structure plus a named citation.

---

## Status

The **service is complete and runs**. The **science layer is stubs** — every
function in `src/winnow/chem/` raises `NotImplementedError` behind a docstring
that specifies exactly what it must do.

```bash
make install
make test-api     # 27 passed   <- the service
make test-chem    # 34 passed, 34 failed   <- the spec you are implementing
```

`parse.py` is implemented (see [the stereo note](#a-trap-worth-knowing-about));
the remaining eight modules are stubs.

Start the server against the stubs and it behaves correctly: jobs are accepted,
dispatched, and fail with a message naming the exact stub that stopped them.

```
$ curl -s localhost:8000/v1/jobs/$ID | jq -r .error
not implemented yet: pipeline.py:53 in process_chunk()
```

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
([`jobs/runner.py`](src/winnow/jobs/runner.py)). The parent stays responsive,
workers do the chemistry, results come back over a pickle boundary — and that
boundary is what shapes the next decision.

The pool uses the **spawn** start method, not Linux's default fork: forking a
process that already has threads (any ASGI server) can deadlock in the child,
and CPython changes this default in 3.14.

### 2. A two-phase pipeline

[`chem/pipeline.py`](src/winnow/chem/pipeline.py) splits into:

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

[`jobs/store.py`](src/winnow/jobs/store.py) defines an abstract `JobStore`; the
in-memory implementation is enough to develop against but dies with the process
and is not shared between uvicorn workers. Write a `RedisJobStore` against the
same interface and change one line in the lifespan — the routes never learn
which one they are talking to. Every method is `async` even where the in-memory
one never awaits, precisely so that swap costs nothing.

---

## What you implement

Nine modules under [`src/winnow/chem/`](src/winnow/chem/). Each stub carries the
algorithm, the RDKit calls to use, and the traps. Suggested order — each step
makes the next testable:

| # | Module | What it does | Watch out for |
|---|---|---|---|
| 1 | ~~`parse.py`~~ **done** | SMILES → sanitised, standardised mol + InChIKey | — |
| 2 | `descriptors.py` | MW, clogP, TPSA, HBD/HBA, RotB, Fsp3, stereo | Keys must match the `Descriptors` schema exactly |
| 3 | `rules.py` | Lipinski, Veber, Egan, Ghose, lead-like, Ro3 | **Lipinski permits one violation** — the most-mis-implemented rule in cheminformatics |
| 4 | `alerts.py` | PAINS / BRENK / NIH via RDKit `FilterCatalog` | Build each catalog **once**; rebuilding per molecule is ~10× the runtime |
| 5 | `scaffolds.py` | Bemis–Murcko | Acyclic → `None`, not `""` |
| 6 | `fingerprints.py` | ECFP4 + Tanimoto | ECFP**4** is radius **2**; use `BulkTanimotoSimilarity` |
| 7 | `cluster.py` | Butina | O(n²) — will not fit at 200k. Cap it, or use `LeaderPicker` |
| 8 | `score.py` | Composite score + breakdown | Must stay in [0,1] for *any* caller weights |
| 9 | `pipeline.py` | Composes 1–8 across the two phases | Only picklable args; must never raise |

```bash
make test-chem              # the whole spec
.venv/bin/pytest -m chem -k lipinski -x    # one function at a time
```

The contract tests are not tautologies — they encode real properties. Aspirin
must be closer to salicylic acid than to caffeine; standardising twice must
give the same answer as once; the counts must reconcile
(`parsed + parse_failed == submitted`). Read the test before writing the
function.

### A trap worth knowing about

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

### Throughput

Measured on this machine, per core, over a mixed 5k library:

| | mol/s/core | 200k library, 8 cores |
|---|---|---|
| with canonical tautomer | ~780 | ~32 s |
| `canonical_tautomer=False` | ~2400 | ~11 s |

The tautomer pass costs about 3× and buys keto/enol forms of one compound
deduplicating against each other. It is on by default.

---

## Config

Everything is `WINNOW_`-prefixed; see [`.env.example`](.env.example) and
[`config.py`](src/winnow/config.py).

| Variable | Default | |
|---|---|---|
| `WINNOW_MAX_LIBRARY_SIZE` | 250000 | Hard cap per job |
| `WINNOW_CHUNK_SIZE` | 2000 | Molecules per worker unit. Too small and pickling dominates; too large and progress goes lumpy |
| `WINNOW_WORKER_PROCESSES` | 0 | 0 → `os.cpu_count()` |
| `WINNOW_MAX_SYNC_BATCH` | 100 | Cap on the synchronous endpoints |

---

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
src/winnow/
  main.py              app factory + lifespan (owns store and pool)
  config.py            pydantic-settings
  api/
    deps.py            dependency providers — nothing reaches for a global
    routes/            health, jobs, molecules
  schemas/             the wire contract (molecule, filters, job)
  jobs/
    store.py           JobStore ABC + in-memory implementation
    runner.py          async supervisor over the process pool
  chem/                ← the stubs. No FastAPI imports allowed here.
tests/
  test_api_*.py        pass today
  test_jobs_store.py   pass today
  test_chem_*.py       the spec
```

`chem/` deliberately imports nothing from FastAPI: it must stay usable from a
worker process, a notebook, or a CLI without an app instance.

## License

MIT.
