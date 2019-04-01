# Multi-Level Intermediate Representation Overview

The MLIR project aims to define a common intermediate representation (IR) that
will unify the infrastructure required to execute high performance machine
learning models in TensorFlow and similar ML frameworks. This project will
include the application of HPC techniques, along with integration of search
algorithms like reinforcement learning. This project aims to reduce the cost to
bring up new hardware, and improve usability for existing TensorFlow users.

## What is this doc?

Whereas the [MLIR draft specification](g3doc/LangRef.md) discusses the details
of the IR in a dry style intended to be a long-lived reference document, this
document discusses higher level issues. This includes:
* How we see the IR being used
* How the compiler will be implemented
* What capabilities the IR enables

## More resources

For more information on MLIR, please see:

*   [The MLIR draft specification](g3doc/LangRef.md), which describes the IR
    itself,
*   [The MLIR rationale document](g3doc/Rationale.md), covering motivation
    behind some decisions,
*   previous external [talks](#talks),

or join the [MLIR mailing list](https://groups.google.com/a/tensorflow.org/forum/#!forum/mlir).

## What is MLIR for?

MLIR is intended to be a hybrid IR which can support multiple different
requirements in a unified infrastructure. For example, this includes:

*   The ability to represent all TensorFlow graphs, including dynamic shapes,
    the user-extensible op ecosystem, TensorFlow variables, etc.
*   Optimizations and transformations typically done on a TensorFlow graph, e.g.
    in Grappler.
*   Quantization and other graph transformations done on a TensorFlow graph or
    the TF Lite representation.
*   Representation of kernels for ML operations in a form suitable for
    optimization.
*   Ability to host high-performance-computing-style loop optimizations across
    kernels (fusion, loop interchange, tiling, etc), and transform memory
    layouts of data.
*   Code generation "lowering" transformations such as DMA insertion, explicit
    cache management, memory tiling, and vectorization for 1D and 2D register
    architectures.
*   Ability to represent target-specific operations, e.g. the MXU on TPUs.

MLIR is a common IR which also supports hardware specific operations. Thus,
any investment into the infrastructure surrounding MLIR (e.g. the compiler
passes that work on it) should yield good returns; many targets can use that
infrastructure and will benefit from it.

MLIR is a powerful representation, but it also has non-goals. We do not try to
support low level machine code generation algorithms (like register allocation
and instruction scheduling). They are a better fit for lower level optimizers
(such as LLVM). Also, we do not intend MLIR to be a source language that
end-users would themselves write kernels in (analogous to CUDA C++). While we'd
love to see a kernel language happen someday, that will be an independent
project that compiles down to MLIR.

## Compiler Infrastructure {#compiler-infrastructure}

We benefitted from the experience gained building HLO, LLVM and SIL when
building MLIR. We will directly adopt existing best practices, e.g. writing and
maintaining an IR spec, building an IR verifier, providing the ability to dump
and parse MLIR files to text, writing extensive unit tests with the
[FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) tool, and
building the infrastructure as a set of modular libraries that can be combined
in new ways. We plan to use the infrastructure developed by the XLA team for
performance analysis and benchmarking.

Other lessons have been incorporated and integrated into the design in subtle
ways. For example, LLVM has non-obvious design mistakes that prevent a
multithreaded compiler from working on multiple functions in an LLVM module at
the same time. MLIR solves these problems by having per-function constant pools
and by making references explicit with function_ref.

# Getting started with MLIR

MLIR has been tested on Linux and MacOS, with a recent clang or with gcc 7.

```
git clone https://github.com/llvm/llvm-project.git
cd llvm-projects/llvm/projects/
git clone https://github.com/tensorflow/mlir
cd ../../
mkdir build
cd build
cmake -G Ninja ../llvm/ 
ninja check-mlir
```

# MLIR talks {#talks}

*   "[MLIR Primer: A Compiler Infrastructure for the End of Moore’s Law](https://drive.google.com/file/d/1hUeAJXcAXwz82RXA5VtO5ZoH8cVQhrOK/view?usp=sharing)",
    Chris Lattner & Jacques Pienaar, Google at
    [Compilers for Machine Learning](https://www.c4ml.org/) workshop at
    [CGO 2019](http://cgo.org/cgo2019/).
