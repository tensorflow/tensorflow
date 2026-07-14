# XLA:CPU Runtime

XLA:CPU runtime is implemented as a collection of `Thunks` that are responsible
for executing individual operations. XLA fusions, for example are jit-compiled
to executables using LLVM, and executed at run time by `KernelThunk`. Operations
that are not compiled have corresponding thunks, i.e., `FFT` operations is
executed as `FftThunk` and relies on DUCC FFT implementation.

Thunks are executed concurrently using `ThunkExecutor`, which launches thunks
when all data dependencies are ready. We rely on buffer assignment to track read
and write conflicts, and compute a directed acyclic graph that defines execution
order.

Conceptually, XLA:CPU runtime is similar to XLA:GPU, which also has thunks.
However, for CPU backend we do a lot more multi-threading to be able to
efficiently use all available cores on the host CPU.
