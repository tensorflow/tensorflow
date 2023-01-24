# MLIR Replay tool

This tool is mainly intended for helping debug miscompiles. It takes as inputs
an HLO snapshot proto with input tensors and a compiler trace proto with the
state of the IR after each pass.

This tool is built on top of
[mlir-interpreter](https://github.com/tensorflow/mlir-hlo/tree/master/tools/mlir_interpreter/).

Example usage:

```
$ ./mlir_replay --mlir_compilation_trace=trace.pb --print_changes_only
Running IR after APass
Results: [1, 2, 3]

Running IR after BPass
Running IR after CPass
Running IR after BrokenPass
Results: [1, 1, 1]

Running IR after DPass
```

If you have an HLO snapshot with actual inputs, you can pass it in
`--hlo_snapshot`. Otherwise, random values will be used.
