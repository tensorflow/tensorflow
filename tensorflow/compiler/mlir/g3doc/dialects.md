# MLIR dialects

## Overview


To separate different hardware and software targets, MLIR has “dialects”,
including:

* TensorFlow IR, which represents all things possible in TensorFlow graphs.
* XLA HLO IR, which is designed to take advantage of XLA’s compilation
  abilities (with output to, among other things, TPUs).
* An experimental affine dialect, which focuses on
  [polyhedral representations](https://en.wikipedia.org/wiki/Polytope_model)
  and optimizations.
* LLVM IR, which has a 1:1 mapping between it and LLVM’s own representation,
  allowing MLIR to emit GPU and CPU code through LLVM.
* TensorFlow Lite, which will translate to running code on mobile platforms.

Each dialect consists of a set of defined operations which have invariants
placed on them, like: “This is a binary operator, and the inputs and outputs
have the same types.”

## Adding to MLIR

MLIR has no fixed/built-in list of globally known operations (no “intrinsics”).
Dialects can define entirely custom types, which is how MLIR can model things
like the LLVM IR type system (which has first class aggregates), domain
abstractions important for ML-optimized accelerators like quantized types, and
even the Swift or Clang type systems (which are built around Swift/Clang
declaration nodes) in the future.

If you want to connect a new low-level compiler, you would create a new dialect
and the lowerings between the TensorFlow Graph dialect and your dialect.
This smooths the path for hardware and compiler makers. You can even target
dialects at different levels in the same model; the higher-level optimizers
will respect the unfamiliar parts of the IR and wait for a lower level to handle
it.
