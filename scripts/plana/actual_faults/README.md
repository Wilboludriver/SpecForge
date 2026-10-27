# Actual loaded-engine fault probes

These tools use the fixed source checkouts and selected real target models. Production source, C's numerical harness, models and reference repositories are read-only. Importing or validating a plan starts no server, network socket or GPU context. No GPU execution has been performed for this tool handoff.

The root agent must assign GPUs, HTTP/NCCL/script-dispatch ports and a running Mooncake TCP service before `--execute`. There is no free-port selection. The launcher creates only its own scripted HTTP server and records its owned Process PID. It reuses the official `ScriptedHttpServer`/`ScriptedContext` and RunScript protocol. `launch_server(run_scheduler_process_func=...)` is the official callable seam used to install test wrappers on every TP worker; the original scheduler entry then runs unchanged. No production plugin, extra service or production endpoint is added.

The script/context exists only on TP rank0. It yields actual scheduler iterations while external threads call the actual SpecForge adapter and MooncakeFeatureStore. Test commands are atomic files observed at normal scheduler phase boundaries; the driver never calls a TP collective directly. Existing request broadcasts and bridge match/admission votes remain responsible for all-rank ordering. Every injection matches exact case/sample/rid, rank mask and one-shot receipt; foreign warmup requests are untouched. Global and per-request Python wrappers restore on exit.

## Executable probes

- `match_error`: raise after the actual engine has matched and allocated a fresh Mamba COW slot. The engine may rematch/retry or return an explicit failure; every successful delivery must remain complete and match forced-full reference features.
- `admit_no_token`, `admit_no_membership`, `admit_after_append`: refuse before add, return CONTINUE without membership, or raise after the real PrefillAdder appended. On hybrid models the first actual admission transaction must prove fresh COW on **every rank** and return its fresh slots. No fresh slot means fixture failure, never a skipped pass. TP2 runs each rank as the refusing rank.
- `d2h_partial`: inside the real capture worker, delay its real CUDA copy stream, submit the first native pinned D2H, and throw before the second required feature copy. The Tensor.copy_ wrapper requires the capture thread, pinned destination, CUDA source and exact source feature identity. A real pending stream event and writer-not-drained control response establish the barrier; the real controller drains all issued work. No SampleRef may escape.
- `d2h_cancel`: derive a real delayed CUDA event from the original snapshot event, then cancel through normal HTTP/control. Record `event.query()==false` at cancellation. A real producer event, source weakrefs, controller byte counters and final explicit drain establish lifetime; a timeout alone does not prove drain.
- `publish_fail`: the actual Mooncake native publication succeeds, then its worker reports an injected failure. No ref is adopted; producer fencing must remove all exact deterministic g1 keys before replacement.
- `publish_hold_cancel`, `publish_hold_timeout`: hold the actual sink worker before its native write. The latter requires the real HTTP request to remain pending when the held boundary is observed, then actually time out. Cancel while writer remains held, require replacement begin to reject, release via an acknowledged command, drain writer/removals/cleanup ACK, and retry the same sample/g1 using different tokens. Every successful replacement must survive old callbacks and match its own full reference.
- `prefill_retract`: on every TP rank, intercept the first exact target `get_next_batch_to_run` result only after its real EXTEND admission allocated KV/request/GDN state and a warm HS plan. Drain the owned scheduling CUDA copies once, then invoke unchanged engine `retract_all(..., offload_kv=False)` before any target forward; requeue that same HTTP attempt and return an actual empty `NextBatchPlan` for that scheduler tick. The next real scheduler loop must perform a second genuine admission and one full epoch1/L0 forward. Assert old lease/output drains, actual pool return, `retraction_count=1`, rank-identical retry plans and full features equal to forced-full reference. No generated result is synthesized. Before/after controller/cache budgets are recorded. Every TP rank must select an actual positive old L; rank0 must prove an active HS lease and output reservation. If ordinary query prewarm still yields L0, the fixture fails and needs an additional ordinary warm request; H alone never qualifies. This covers admitted pre-forward retraction; it does not claim decode or mid-forward retraction coverage. Ordinary pause alone is insufficient: max-new0 requests can finish when pause first drains the overlap result.

  Each rank records exact request-owned suffix KV IDs from `cache_protected_len:kv_allocated_len`, its first-match fresh COW destination, allocated ping-pong slots and the separate cache-owned COW source. Real allocator wrappers only observe unchanged frees and always restore. Pass requires every owned ID in the real free calls, a pool capacity increase at least as large as the owned allocation (cache eviction may increase it further), and exact returned-ID membership for static allocator free lists. Virtual unified allocators use their slot-conservation counts with actual virtual-ID free receipts, since their physical free lists cannot prove virtual-ID membership. Cache checkpoint sources are never counted as request-owned. The offline all-rank receipt oracle repeats these checks; CPU negative controls reject missing KV/Mamba returns, unrelated frees, fake count-only proof, double frees and cache sources misclassified as owned. A positive CPU control permits additional real allocator returns caused by checkpoint eviction.
- `server_restart`: dedicated run (`kinds=["server_restart"]`). Keep the external Mooncake service alive; publish one actual S1 fenced sample without beginning a consumer read, prove writer/copy quiescence, and stop only that owned server tree. Start S2 on the same root-assigned ports, require a different epoch, and use the same tracked old store/ref for get, resolve, store abort and normal same-sample adapter retry. All four must explicitly reject stale ownership before any native fetch/remove/put or `/generate`. Preserve the old fence/intents in `unresolved-ownership.json`; present/absent/unknown old keys never imply retirement. Server child directories include the unique outer run name so each independent run has fresh physical namespaces. A fresh unique S2 run/store/sample must still capture/fetch/drain normally and match S1 forced-full golden features. This is a scoped fail-closed result with `cleanup_proven=false`, never a zero-leak claim.
- `hs_only`: warm actual features, flush engine KV using ordinary broadcast HTTP, then require observed H>0 and L=0 with a complete sample.
- `kv_only`: pin a genuine actual engine checkpoint on every rank, create a larger real HS working set until READY coverage becomes zero, release the fixture KV pins, and require uncapped actual engine checkpoint coverage>0 while H=L=0.
- `hs_shorter_than_kv`: a dedicated run whose CPU cache is exactly one complete feature chunk. Pin its first real READY chunk while a longer actual request fills KV; optional HS fills skip under real capacity pressure. Release the fixture lease and require positive H smaller than the actual uncapped legal engine checkpoint. L is chosen only by the unchanged engine. This does not replace C's separate nonaligned C384 H>L fixture.

Default probes cover the first eleven entries through HS-only, including pre-forward retraction. Select KV-only or the dedicated one-chunk fixture explicitly. Coverage probes log actual cow=False queries on every rank, rather than inferring KV hits from HS lengths. The hybrid run must expose real FULL+MAMBA components. These initial scripts require graphs disabled. Graph/overlap ownership has separate root evidence. The dedicated restart probe preserves unknown old ownership and reports failure closed; it never claims uncertain old sample keys can be reclaimed without proof.

## Configuration

Provide a JSON object with these fields. ROOT placeholders below are replaced by the root agent before validation; ports must be distinct integers and commit IDs must be exact clean current commits.

```json
{
  "expected_commits": {"sglang": "ROOT_SGLANG_COMMIT", "SpecForge": "ROOT_SPECFORGE_COMMIT"},
  "allocation": {"cuda_visible_devices": "ROOT_GPU_LIST", "http_port": "ROOT_HTTP_INT", "dispatch_port": "ROOT_DISPATCH_INT", "nccl_port": "ROOT_NCCL_INT"},
  "model": {"name": "qwen35-4b", "hidden_size": 2560, "layers": [1,5,9,13,17,21,25,29], "hybrid": true},
  "engine_kwargs": {
    "model_path": "/workdir/spectrain/spectrain-16g/models/qwen35-4B",
    "tp_size": 2, "port": "ROOT_HTTP_INT", "nccl_port": "ROOT_NCCL_INT",
    "dtype": "bfloat16", "attention_backend": "fa3", "page_size": 1,
    "context_length": 2056, "max_running_requests": 1, "chunked_prefill_size": -1,
    "disable_cuda_graph": true, "enable_deterministic_inference": true,
    "enable_spec_capture": true, "spec_feature_cache": true,
    "spec_capture_method": "dflash", "spec_capture_aux_layer_ids": [1,5,9,13,17,21,25,29],
    "mamba_radix_cache_strategy": "extra_buffer"
  },
  "mooncake_env": {
    "MOONCAKE_PROTOCOL": "tcp", "SGLANG_SPEC_CAPTURE_GPU_PUT": "0",
    "MOONCAKE_LOCAL_HOSTNAME": "ROOT_HOST", "MOONCAKE_METADATA_SERVER": "ROOT_METADATA",
    "MOONCAKE_MASTER_SERVER_ADDR": "ROOT_MASTER", "MOONCAKE_GLOBAL_SEGMENT_SIZE": "1073741824",
    "MOONCAKE_LOCAL_BUFFER_SIZE": "1073741824", "MOONCAKE_RDMA_DEVICES": ""
  },
  "store_setup": {
    "protocol": "tcp", "local_hostname": "ROOT_HOST", "metadata_server": "ROOT_METADATA",
    "master_server_addr": "ROOT_MASTER", "global_segment_size": 1073741824,
    "local_buffer_size": 1073741824, "rdma_devices": ""
  },
  "cases_file": "/workdir/spectrain/SpecTrain-with-Prefix-Cache/preparation/data/native-token-cases.jsonl",
  "feature_tolerances": {"rtol": 0.02, "atol": 0.02},
  "timeout_s": 90, "suite_timeout_s": 1800, "http_timeout_s": 10,
  "delay_cycles": 2000000000,
  "kinds": ["match_error", "admit_no_membership", "admit_after_append", "d2h_partial", "publish_fail", "publish_hold_cancel", "publish_hold_timeout", "d2h_cancel", "hs_only"]
}
```

Model names are exactly `qwen3-4b`, `qwen35-4b`, `qwen36-a3b`; schemas/paths are validated against the fixed user-selected targets. Qwen3 has hidden2560/layers1,9,17,25,33/hybrid=false; A3B has hidden2048/layers1,6,11,16,22,27,32,37/hybrid=true. Dense hybrid supports TP1/TP2; A3B uses TP2. No model or draft alternative is substituted. The tool performs capture/fault verification, not draft training or gradient acceptance.

## Commands

The committed entry point is `src/SpecForge/scripts/plana/run_actual_faults.py`.
For committed runs, put `src/SpecForge/scripts/plana` first in `PYTHONPATH`,
followed by `src/SpecForge` and `src/sglang/python`, so the supervisor and its
`python -m actual_faults.driver` child both load this versioned package.
The `implementation/tools` commands below identify the original development
copy and remain useful for reproducing its recorded CPU evidence.

Use the fixed Python and explicit source imports. CPU self-tests start no services:

```bash
CUDA_VISIBLE_DEVICES= TMPDIR=/workdir/spectrain/temp PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/workdir/spectrain/SpecTrain-with-Prefix-Cache/implementation/tools:/workdir/spectrain/SpecTrain-with-Prefix-Cache/src/SpecForge:/workdir/spectrain/SpecTrain-with-Prefix-Cache/src/sglang/python \
/workdir/spectrain/training-envs/venvs/specforge-main-py312-cu129-sglang0518/bin/python \
-m actual_faults.driver --cpu-self-test
```

With the same imports, `-m actual_faults.driver --config ROOT_CONFIG.json` validates only. Root later runs `--execute --run-dir ROOT_UNIQUE_RUN_DIR` with its explicitly assigned CUDA_VISIBLE_DEVICES. There is no autonomous GPU allocation or port selection.

For actual execution use the root-owned `implementation/tools/run_actual_faults.py --config ROOT_CONFIG.json --service-template ROOT_SERVICE_TEMPLATE.yaml --run-dir ROOT_UNIQUE_RUN_DIR --execute` under the same fixed Python/imports and allocated CUDA environment. This supervisor bounds the full startup/suite lifetime and owns its detached driver process group plus its Mooncake service group. The inner driver alone can wait for an uncertain native I/O or client thread during close; it is not the outer wall-clock bound. Unknown sample ownership remains recorded when that bound terminates a failed run.

## Evidence and failure semantics

The run manifest records exact clean production commits, actual import paths, complete test-tool source hashes, config, deterministic mode and unchanged predeclared tolerances. Before pass, the same commit/dirty/import/tool hashes are checked again. Per-rank JSONL records actual match caps/L, fresh COW source/destination, frees, request membership, component locks, pool counts, ordered collective calls, fault receipts and native publication/event barriers. Full successful samples require exactly aux+last_hidden+IDs/mask, exact N/width/BF16/int64, finite floats and exact per-request IDs/masks; no zero fill or truncation exists.

Golden features use actual forced-full prefill under the same configured target/backend, not an official-only baseline. C retains ownership of the independent official-off/on numeric/loss/gradient oracle. Byte counters, exact remote sample-key absence, active attempts/readers and producer/consumer intents/fences are checked after explicit drain; a READY cache and an accounted idle pinned arena may remain. If writer or receive drain is unknown, the run fails and preserves `unresolved-ownership.json` including fences and lost-begin intent. No timeout, shutdown or missing response is relabeled as successful cleanup. Only the Process created by this launcher is eligible for forced teardown.

## Dedicated natural same-content fill overlap (A3B TP2)

Select `kinds: ["same_content_fill_overlap"]` in a fresh dedicated run, with
`spec_feature_chunk_tokens: 256`, `spec_feature_cache_bytes: 67108864`, unchanged
`max_running_requests: 1`, the normal single capture worker, and optional
`fill_gate_timeout_s` in [5,45] (default45). All normal model/FA3/BF16/deterministic,
FULL+MAMBA, fixed-source and root-allocation requirements above remain mandatory.
This scenario runs before the shared golden/warmup: forced-full reference capture
also populates HS, so prewarming the same key invalidates this fixture.

Its pair arm uses `kind=same_content_fill_overlap` and an `actors` mapping with
exact A/B `rid` and `sample_id`; both have identical native extension-1280 tokens,
independent actual attempts, and separate physical sample keys. A normal fill is
held only after its real pageable CPU aux copy, while the first chunk is WRITING
and last_hidden is still unpublished. B must actually admit/forward on both TP
ranks after A releases its effective engine slot, then submit its real full-range
CUDA snapshot to the existing one-worker controller. The submit wrapper records
queued/charged ownership and queries the genuine producer event, then releases A
in finally before pending-batches=2 can block later owner commands. No extra
reserve call, writer, worker, tensor result or collective is synthesized.

The normal A publisher commits all five complete chunks once; the normal B fill
subsequently observes READY and skips all five keys. The offline scalar oracle
requires the exact natural partial order and per-key owner/reservation receipts,
not just two pending HTTP requests or the global fill_skips counter. If B cannot
naturally reach snapshot before A's bounded gate expires, coverage fails and the
serialization boundary is retained; engine concurrency is never increased.

A/B complete samples are compared to actual forced-full G captured after the
pair using the unchanged predeclared tolerances; full hashes/difference counts
are saved. A later ordinary R must use positive actual L and match G, exercising
complete READY prefix assembly. Original per-request IDs/masks, all four distinct
sample references, exact key absence, server/store reader/fence/intent drain,
cache leases/WRITING/fills and controller private buffers are checked. Accounted
READY cache and idle pinned arena may remain. Existing outer fault supervision
and unresolved-ownership semantics apply on failure.

CPU-only tests (no service/port/GPU):

```bash
CUDA_VISIBLE_DEVICES= TMPDIR=/workdir/spectrain/temp PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=ROOT_SF/scripts/plana:ROOT_SF:ROOT_SG/python \
/workdir/spectrain/training-envs/venvs/specforge-main-py312-cu129-sglang0518/bin/python \
-m unittest discover -s tests/plana -p test_concurrent_fill_oracle.py -v
```

The new test module covers receipt negative controls and actual CPU cache-copy
barrier mechanics only. A passing CPU suite is not actual A3B GPU acceptance.
