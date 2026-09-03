# KDNN — Kunpeng Deep Neural Network Library

This directory holds the integration glue for the third-party **KDNN** library
developed by Huawei as part of openEuler's **KAIL (Kunpeng AI Library) BoostKit**.
KDNN targets the **Kunpeng 920** (ARMv8) CPU.

This is intentionally a **header + BUILD skeleton** — the actual shared library
(`libkdnn.so`) is expected to be provided by the operator. See the
*Distribution* section below.

## What is here

| File | Purpose |
|---|---|
| `kdnn.h` | Minimal C API header (declarations only). Mirrors the public surface of KAIL's KDNN. No implementation. |
| `build_defs.bzl` | Bazel macros: `if_enable_kdnn`, `kdnn_deps`. Loaded from BUILD files via `tensorflow/tensorflow.bzl`. |
| `repository.bzl` | `kdnn_repository` rule (loaded only from `WORKSPACE` / `MODULE.bazel`). |
| `BUILD` | `cc_library` rules for the header + an in-tree stub target. |
| `HARDWARE_VERIFICATION.md` | Step-by-step recipe for end-to-end verification on Kunpeng 920 + libkdnn.so. See "Hardware verification" below. |
| `kdnn_e2e_smoke.py` | Step-7 of the recipe: builds a Sigmoid graph, runs Grappler, asserts `_KdnnSigmoid` appears and matches eager numerically. |

## What is NOT here

- The KDNN implementation itself (`.c`, `.cpp`, `.s` ARM assembly).
- The actual `libkdnn.so` binary.
- Any KAIL BoostKit packaging.

Those are downloaded or symlinked into this tree at *build time* via the
`KDNN_ROOT` environment variable (analogous to the existing `TF_MKL_ROOT`).
See the `kdnn_repository` rule in `repository.bzl`.

## Build flag

```sh
# Explicit opt-in; default off.
bazel build --define=enable_kdnn=true //tensorflow/...
```

The flag is a TRANSITIVE — both `tensorflow/tensorflow.bzl` and the
Kunpeng-aarch64 `config_setting` (declared in `tensorflow/workspace.bzl`)
must be satisfied for the KDNN kernels to compile.

## Distribution

The KDNN source is distributed by Huawei under the openEuler BoostKit. The
TF integration expects one of two models:

1. **Pre-built binary** (recommended for the initial PR): the operator places
   `libkdnn.so` and `kdnn.h` under a directory pointed to by `KDNN_ROOT`.
2. **Source build** (future work): the operator runs a build script that
   produces `libkdnn.so` and places it under `KDNN_ROOT/lib`.

Either model lands the artifacts where the `kdnn_repository` rule can find
them.

## Why this is a separate third_party directory

KDNN is **not** a generic x86 backend — it is ARM-only. The existing MKL
tree would be misleading because the activation is gated by a different
`config_setting` (aarch64 + opt-in flag vs. x86 + default-on). Keeping it
separate also makes it easy to backport independently of MKL.

## License status (UNRESOLVED)

This directory ships with `licenses(["restricted"])` in `BUILD`. Until
that line is changed to `licenses(["notice"])`, the entire directory is
flagged by Google's internal tooling as not safe to import. The blocker
is independent of this code review and cannot be resolved inside this
PR — it requires confirmation from the openEuler / KAIL BoostKit
maintainers that libkdnn.so and kdnn.h may be redistributed under
Apache 2.0 (or a compatible permissive license).

To resolve:

1. File an issue in the openEuler community tracker asking KAIL SIG
   (kail@openeuler.org) to clarify the KDNN distribution terms.
2. Either obtain a written license grant permitting the in-tree
   header + dlopen() integration path used here, OR confirm KDNN is
   released under Apache 2.0.
3. Once confirmed, change `licenses(["restricted"])` to
   `licenses(["notice"])` in `BUILD` and remove the corresponding
   comment block.

## Hardware verification

The skeleton PR has been authored without access to Kunpeng 920 hardware,
so it cannot ship benchmark numbers or end-to-end execution evidence. To
unblock the "Hardware Verification" and "Benchmark Results" action items
in the review thread, this directory ships:

* **`HARDWARE_VERIFICATION.md`** — a 7-step recipe that takes a reviewer
  from a fresh KAIL BoostKit install to a populated PR comment with
  numbers. Designed to be runnable in under 30 minutes by anyone with
  Kunpeng 920 access.
* **`tensorflow/core/kernels/kdnn/kdnn_sigmoid_benchmark.cc`** — a
  microbenchmark target (`//tensorflow/core/kernels/kdnn:kdnn_sigmoid_benchmark_test`)
  that compares `_KdnnSigmoid` against the default CPU `Sigmoid`. On
  non-KDNN platforms it is a no-op that reports "KDNN unavailable on
  this platform" and exits cleanly, so it is safe to keep in the default
  build graph.
* **`kdnn_e2e_smoke.py`** — Step 7 of the recipe. Builds a small
  Sigmoid graph, runs Grappler, asserts the rewrite to `_KdnnSigmoid`
  fires and the rewritten op executes without numerical drift.

When the PR author (or a reviewer with hardware access) runs the recipe,
the resulting numbers and pass/fail lines should be pasted back into the
PR thread. Until then, this directory explicitly does **not** claim
performance figures; the placeholder numbers in
`kdnn_sigmoid_benchmark.cc` are illustrative format markers only.
