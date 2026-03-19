# XLA architecture

XLA (Accelerated Linear Algebra) is a machine learning (ML) compiler that
optimizes linear algebra, providing improvements in execution speed and memory
usage. This page provides a brief overview of the objectives and architecture of
the XLA compiler.

## Objectives

Today, XLA supports several ML framework frontends (including PyTorch,
TensorFlow, and JAX) and is part of the OpenXLA project &ndash; an ecosystem of
open-source compiler technologies for ML that's developed collaboratively by
leading ML hardware and software organizations. Before the OpenXLA project was
created, XLA was developed inside the TensorFlow project, but the fundamental
objectives remain the same:

*   **Improve execution speed.** Compile subgraphs to reduce the execution time
    of short-lived ops and eliminate overhead from the runtime, fuse pipelined
    operations to reduce memory overhead, and specialize known tensor shapes to
    allow for more aggressive constant propagation.

*   **Improve memory usage.** Analyze and schedule memory usage, eliminating
    many intermediate storage buffers.

*   **Reduce reliance on custom ops.** Remove the need for many custom ops by
    improving the performance of automatically fused low-level ops to match the
    performance of custom ops that were originally fused by hand.

*   **Improve portability.** Make it relatively easy to write a new backend for
    novel hardware, so that a large fraction of ML models can run unmodified on
    that hardware. This is in contrast with the approach of specializing
    individual monolithic ops for new hardware, which requires models to be
    rewritten to make use of those ops.

## How it works

The XLA compiler takes model graphs from ML frameworks defined in
[StableHLO](https://github.com/openxla/stablehlo) and compiles them into machine
instructions for various architectures. StableHLO defines a versioned operation
set (HLO = high level operations) that provides a portability layer between ML
frameworks and the compiler.

In general, the compilation process that converts the model graph into a
target-optimized executable includes these steps:

1.  XLA performs several built-in optimization and analysis passes on the
    StableHLO graph that are target-independent, such as
    [CSE](https://en.wikipedia.org/wiki/Common_subexpression_elimination),
    target-independent operation fusion, and buffer analysis for allocating
    runtime memory for the computation. During this optimization stage, XLA also
    converts the StableHLO dialect into an internal HLO dialect.

2.  XLA sends the HLO computation to a backend for further HLO-level
    optimizations, this time with target-specific information and needs in mind.
    For example, the GPU backend may perform operation fusions that are
    beneficial specifically for the GPU programming model and determine how to
    partition the computation into streams. At this stage, backends may also
    pattern-match certain operations or combinations thereof to optimized
    library calls.

3.  The backend then performs target-specific code generation. The CPU and GPU
    backends included with XLA use [LLVM](http://llvm.org) for low-level IR,
    optimization, and code generation. These backends emit the LLVM IR necessary
    to represent the HLO computation in an efficient manner, and then invoke
    LLVM to emit native code from this LLVM IR.

Within this process, the XLA compiler is modular in the sense that it is easy to
slot in an alternative backend to
[target some novel HW architecture](./developing_new_backend.md). The GPU
backend currently supports NVIDIA GPUs via the LLVM NVPTX backend. The CPU
backend supports multiple CPU ISAs.
