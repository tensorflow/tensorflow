# Hardware Verification Recipe for KDNN

This document is the step-by-step recipe an operator follows to verify the
end-to-end KDNN integration on real **Kunpeng 920 (aarch64)** hardware with
a real **`libkdnn.so`**. It exists to satisfy the "Hardware Verification"
action item in PR #124543 review feedback.

The recipe has not yet been executed end-to-end by the PR author (no
Kunpeng 920 host is available in the contributor's environment). It is
written so that:

1. A reviewer with Kunpeng 920 access can run the verification in under
   30 minutes and report results.
2. The PR author can paste the verification output back into the PR
   thread as evidence.

## Prerequisites

* A Kunpeng 920 host running aarch64 Linux (openEuler 20.03 LTS or
  equivalent).
* KAIL BoostKit installed. The recommended version is **Kunpeng BoostKit
  22.0.1 or later** (this is the version whose `kdnn_context_t` is
  documented as thread-safe; see `kdnn_unary_op.h` for the contract).
* `libkdnn.so` and `kdnn.h` extracted to a directory, e.g.
  `/opt/kail/kdnn/{lib,include}`.
* A TensorFlow build tree at this branch (`feature/kdnn-skeleton-124076`
  or later) checked out on the host.

## Step 1 — Wire `KDNN_ROOT`

```sh
export KDNN_ROOT=/opt/kail/kdnn
ls -l $KDNN_ROOT/include/kdnn.h $KDNN_ROOT/lib/libkdnn.so
```

Both files must exist. If KAIL BoostKit installed under a different path,
adjust accordingly.

## Step 2 — Confirm `libkdnn.so` loads

```sh
ldd $KDNN_ROOT/lib/libkdnn.so
```

The output must show no `not found` lines. If there are missing
dependencies, install the corresponding KAIL BoostKit runtime packages
and retry.

## Step 3 — Build with KDNN enabled

```sh
bazel build --define=enable_kdnn=true \
    //tensorflow/core/kernels/kdnn:kdnn_sigmoid_op \
    //tensorflow/compiler/grappler/optimizers:remapper
```

A green build confirms:

* `kdnn_repository` correctly resolved `@kdnn` to the operator-provided
  tree.
* The KDNN header path is correct.
* `kdnn_apply_activation` and friends are link-resolvable (the skeleton
  uses `dlopen()` at runtime, so the build does not require
  `-L$KDNN_ROOT/lib`, but the include path is needed).

## Step 4 — Run the unit test

```sh
bazel test --define=enable_kdnn=true \
    //tensorflow/core/grappler/optimizers:remapper_test \
    --test_arg=--gtest_filter=*KdnnSigmoid*
```

Expected: `IsKDNNEnabled() == true` for all KDNN-tagged tests, the
Sigmoid→`_KdnnSigmoid` rewrite fires, and the rewritten graph executes
with numerical output matching the reference to within the tolerances
declared in `remapper_test.cc`.

## Step 5 — Run the microbenchmark

```sh
bazel test --define=enable_kdnn=true \
    --test_arg=--benchmark_filter=BM_KdnnVsCpuSigmoid \
    //tensorflow/core/kernels/kdnn:kdnn_sigmoid_benchmark_test
```

Expected: four benchmarks run (`Kdnn_8192`, `Cpu_8192`,
`Kdnn_131072`, `Cpu_131072`). The output must include the `_KdnnSigmoid`
runs at non-zero items/sec; if they report "KDNN unavailable on this
platform", `IsKDNNEnabled()` is returning false and steps 1–3 need
debugging.

The headline numbers (bytes/sec, ns/element) should be recorded in the
PR thread. The current skeleton does not include numbers because no
hardware was available to the author.

## Step 6 — Verify the Grappler rewrite fires

```sh
bazel test --define=enable_kdnn=true \
    //tensorflow/compiler/grappler/optimizers:remapper_test \
    --test_arg=--gtest_filter=*RemapperSigmoid*
```

Expected: at least one test asserts that `tf.math.sigmoid` in a 2-D
float32 graph gets rewritten to `_KdnnSigmoid`. The rewrite is gated
on `IsKDNNEnabled()` and on `pass_config_` allowing the rewrite, so
both must be true.

## Step 7 — End-to-end model smoke test

Build and run a small `.pb` model that contains a Sigmoid activation,
with `TF_ENABLE_KDNN_OPTS=1` in the process environment and the
Grappler remapper enabled. Confirm via `tf.compat.v1.RunMetadata` that
the optimized graph contains `_KdnnSigmoid` nodes (not `Sigmoid`).

```sh
TF_ENABLE_KDNN_OPTS=1 \
bazel-bin/tensorflow/python/platform/bfloat16/_pywrap_tensorflow_internal \
    /tmp/sigmoid_model_test.py
```

A test script `kdnn_e2e_smoke.py` is provided alongside this recipe at
`third_party/KDNN/kdnn_e2e_smoke.py` for exactly this purpose.

## Reporting results back to the PR

Run all six steps and capture their output. The format expected in the
PR review thread is:

```
Step 1: KDNN_ROOT=/opt/kail/kdnn → include and lib verified
Step 2: ldd clean
Step 3: bazel build green
Step 4: remapper_test_kdnn_sigmoid ... PASS in 0.83s
Step 5: BM_KdnnVsCpuSigmoid_Kdnn_8192 ... 1.20 ns/elem
        BM_KdnnVsCpuSigmoid_Cpu_8192 ...  0.85 ns/elem
        BM_KdnnVsCpuSigmoid_Kdnn_131072 ... 4.10 ns/elem
        BM_KdnnVsCpuSigmoid_Cpu_131072 ...  2.55 ns/elem
Step 6: RemapperSigmoidRewrite ... PASS in 0.41s, _KdnnSigmoid nodes=1
Step 7: kdnn_e2e_smoke ... PASS in 1.20s, optimized graph contains _KdnnSigmoid
```

The headline result for the PR description is the **speedup ratio**
from Step 5:

    speedup = (BM_KdnnVsCpuSigmoid_Cpu_<N>_ns_per_elem) /
              (BM_KdnnVsCpuSigmoid_Kdnn_<N>_ns_per_elem)

reported separately for `N=8192` and `N=131072`.