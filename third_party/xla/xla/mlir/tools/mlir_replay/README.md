# MLIR Replay tool

This tool is mainly intended for helping debug miscompiles. It takes as inputs
an HLO snapshot proto with input tensors and a compiler trace proto with the
state of the IR after each pass.

This tool is built on top of
[mlir-interpreter](https://github.com/tensorflow/mlir-hlo/tree/master/tools/mlir_interpreter/).

Example usage:

```
# Run a JAX test with debug flags enabled:
$ bazel test :some_jax_test --compilation_mode=opt \
  --test_env=XLA_FLAGS="--xla_cpu_use_xla_runtime --xla_dump_to=/tmp/test-dump --xla_dump_hlo_snapshots" \
  --test_filter=SomeSpecific.TestCase \
  --test_sharding_strategy=disabled --test_strategy=local

# JAX tends to compile many modules, so first check which one is broken:
./mlir_replay \
  --mlir-compilation-trace-dir=/tmp/test-dump

Failures for /tmp/test-dump/module_1234.jit_something.mlir-trace.pb:
  Result mismatch for /tmp/test-dump/module_1234.jit_something.snapshot.56.pb: TensorOrMemref<3xi32>: [1, 2, 3] != TensorOrMemref<3xi32>: [1, 1, 1]
  run :mlir_replay -- --mlir-compilation-trace=/tmp/test-dump/module_1234.jit_something.mlir-trace.pb --hlo-snapshot=/tmp/test-dump/module_1234.jit_something.snapshot.56.pb --print-changes-only --stop-after-first-failure
```

There may be multiple failing modules. You can run the provided command to
replay a particular one:

```
# Run the IR after each pass. Note that JAX typically compiles many modules, so
# you may have check more than one.
# There is one .mlir-trace.pb file per module (containing the intermediate IR)
# and one .snapshot.pb file per execution (containing the inputs and outputs).
$ ./mlir_replay \
  --mlir-compilation-trace=/tmp/test-dump/module_1234.jit_something.mlir-trace.pb \
  --hlo-snapshot=/tmp/test-dump/module_1234.jit_something.snapshot.56.pb \
  --print-changes-only --stop-after-first-failure
Running IR after APass
Results: [1, 2, 3]

Running IR after BPass
Running IR after CPass
Running IR after BrokenPass
Results: [1, 1, 1]
```

