# Math function library
A library of LLVM IR function definition emitters for common math functions.

Exact mechanisms TBD, but currently:

- Rewrite via llvm::VecDescs
- Leave function calls and bodies intact until final opt pass for readability
- Attempts to be useful to GPU and both CPU pipelines
- Tracks accuracy with TPUs

Comparison with existing math libraries in XLA:

- libc, libm external function calls
  - Function call overhead
  - LLVM can't optimize across function boundaries
- cpu/codegen/polynomial_approximations
  - Rewrites inline
  - No optimizations after
- MLIR PolynomialApproximations.cpp
  - Uses MLIR RewritePatterns to implement the pass
  - Rewrites inline
  - Approximation accuracy goals independent of TPU