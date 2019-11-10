# Multi-Level Intermediate Representation Overview

The MLIR project aims to define a common intermediate representation (IR) that
will unify the infrastructure required to execute high performance machine
learning models in TensorFlow and similar ML frameworks. This project will
include the application of HPC techniques, along with integration of search
algorithms like reinforcement learning. This project aims to reduce the cost to
bring up new hardware, and improve usability for existing TensorFlow users.

Note that this repository contains the core of the MLIR framework. The
TensorFlow compilers we are building on top of MLIR will be part of the
main TensorFlow repository soon.

# How to Contribute

Thank you for your interest in contributing to MLIR! If you want to contribute
to MLIR, be sure to review the [contribution guidelines](CONTRIBUTING.md).

## More resources

For more information on MLIR, please see:

*   [The MLIR draft specification](g3doc/LangRef.md), which describes the IR
    itself.
*   [The MLIR rationale document](g3doc/Rationale.md), covering motivation
    behind some decisions.
*   Previous external [talks](#mlir-talks).

Join the [MLIR mailing list](https://groups.google.com/a/tensorflow.org/forum/#!forum/mlir)
to hear about announcements and discussions.

We also have an [MLIR SIG](https://github.com/tensorflow/community/blob/master/sigs/mlir/CHARTER.md)
which was created to enable collaboration and form a strong
engineering-driven open community. We have weekly 'Open Design Meetings'. If you’d like
to discuss a particular topic or have questions, please add it to the [agenda doc](https://docs.google.com/document/d/1y_9f1AbfgcoVdJh4_aM6-BaSHvrHl8zuA5G4jv_94K8/edit#).
Details on how to join the meeting are in the agenda doc. You
should also get an invite when you join the mailing list.

Please be mindful of the [TensorFlow Code of Conduct](https://github.com/tensorflow/tensorflow/blob/master/CODE_OF_CONDUCT.md),
which pledges to foster an open and welcoming environment.

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
    kernels (fusion, loop interchange, tiling, etc) and to transform memory
    layouts of data.
*   Code generation "lowering" transformations such as DMA insertion, explicit
    cache management, memory tiling, and vectorization for 1D and 2D register
    architectures.
*   Ability to represent target-specific operations, e.g. the MXU on TPUs.

MLIR is a common IR that also supports hardware specific operations. Thus,
any investment into the infrastructure surrounding MLIR (e.g. the compiler
passes that work on it) should yield good returns; many targets can use that
infrastructure and will benefit from it.

MLIR is a powerful representation, but it also has non-goals. We do not try to
support low level machine code generation algorithms (like register allocation
and instruction scheduling). They are a better fit for lower level optimizers
(such as LLVM). Also, we do not intend MLIR to be a source language that
end-users would themselves write kernels in (analogous to CUDA C++). While we
would love to see a kernel language happen someday, that will be an independent
project that compiles down to MLIR.

## Compiler infrastructure

We benefited from experience gained from building other IRs (HLO, LLVM and SIL)
when building MLIR. We will directly adopt existing best practices, e.g. writing
and maintaining an IR spec, building an IR verifier, providing the ability to
dump and parse MLIR files to text, writing extensive unit tests with the
[FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) tool, and
building the infrastructure as a set of modular libraries that can be combined
in new ways. We plan to use the infrastructure developed by the XLA team for
performance analysis and benchmarking.

Other lessons have been incorporated and integrated into the design in subtle
ways. For example, LLVM has non-obvious design mistakes that prevent a
multithreaded compiler from working on multiple functions in an LLVM module at
the same time. MLIR solves these problems by having per-function constant pools
and by making references explicit with `function_ref`.

# Getting started with MLIR

The following instructions for compiling and testing MLIR assume that you have
`git`, [`ninja`](https://ninja-build.org/), and a working C++ toolchain. In the
future, we aim to align on the same level of platform support as
[LLVM](https://llvm.org/docs/GettingStarted.html#requirements). For now, MLIR
has been tested on Linux and macOS, with recent versions of clang and with
gcc 7.

```sh
git clone https://github.com/llvm/llvm-project.git
git clone https://github.com/tensorflow/mlir llvm-project/llvm/projects/mlir
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host"
cmake --build . --target check-mlir
```

To compile and test on Windows using Visual Studio 2017:

```bat
REM In shell with Visual Studio environment set up, e.g., with command such as
REM   $visual-studio-install\Auxiliary\Build\vcvarsall.bat" x64
REM invoked.
git clone https://github.com/llvm/llvm-project.git
git clone https://github.com/tensorflow/mlir llvm-project\llvm\projects\mlir
mkdir llvm-project\build
cd llvm-project\build
cmake ..\llvm -G "Visual Studio 15 2017 Win64" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="host" -DCMAKE_BUILD_TYPE=Release -Thost=x64
cmake --build . --target check-mlir
```

As a starter, you may try [the tutorial](g3doc/Tutorials/Toy/Ch-1.md) on
building a compiler for a Toy language.

# MLIR talks

* "[MLIR Primer: A Compiler Infrastructure for the End of Moore’s Law](https://ai.google/research/pubs/pub48035.pdf)"
  * Chris Lattner & Jacques Pienaar, Google at
    [Compilers for Machine Learning](https://www.c4ml.org/) workshop at
    [CGO 2019](http://cgo.org/cgo2019/)
* "[MLIR: Multi-Level Intermediate Representation for Compiler
    Infrastructure](https://llvm.org/devmtg/2019-04/talks.html#Keynote_1)"
  * Tatiana Shpeisman & Chris Lattner, Google at
    [EuroLLVM 2019](https://llvm.org/devmtg/2019-04)
* "[Tutorial: Building a Compiler with MLIR](https://llvm.org/devmtg/2019-04/talks.html#Tutorial_1)"
  * Mehdi Amini, Jacques Pienaar, Nicolas Vasilache, Google at
    [EuroLLVM 2019](https://llvm.org/devmtg/2019-04)
