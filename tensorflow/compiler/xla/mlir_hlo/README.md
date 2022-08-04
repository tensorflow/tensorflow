# MLIR-HLO: A Standalone "HLO" MLIR-based Compiler

The code here exists in two places:

*   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/xla/mlir_hlo;
    this is the canonical location and where contributions should be made using
    GitHub pull-requests.
*   https://github.com/tensorflow/mlir-hlo; this is a standalone repository with
    a view to the same code to allow other projects to use this without
    depending on the entire TF monorepo.

This implements a self-contained compiler for a linear algebra set of operations
inspired by XLA
[HLO IR](https://www.tensorflow.org/xla/architecture#how_does_xla_work) using
MLIR components. It is designed to provide an end-to-end flow independent of
TensorFlow and XLA, but usable inside of these projects.

Coding practice and conventions in this repository follow the
[MLIR Developer Guide](https://mlir.llvm.org/getting_started/DeveloperGuide/) in
this repo as part of the intent to act as an incubator for technology to
upstream.

## QuickStart: building and testing

These instructions work on Linux, you may have to adjust for your platform.

To build the code in this repository, you need a clone of the LLVM/MLIR git
repository:

    $ git clone https://github.com/llvm/llvm-project.git


You need to make sure you have the right commit checked out in the LLVM
repository (you need to do this every time you pull from this repo):

    $ (cd llvm-project && git checkout $(cat build_tools/llvm_version.txt))

We provide a script to configure and build LLVM/MLIR:

    $ build_tools/build_mlir.sh ${PWD}/llvm-project/ ${PWD}/llvm-build

Again this is something to do every time you pull from this repository and the
LLVM revision changes.

Finally you can build and test this repository:

    $ mkdir build && cd build
    $ cmake .. -GNinja \
       -DLLVM_ENABLE_LLD=ON \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=On \
       -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir
    $ ninja check-mlir-hlo


## Overview

MLIR-HLO aims to provide an end-to-end compiler for CPU and GPU, as well as
building reusable blocks for other accelerators. This is heavily inspired by the
success of XLA.

[XLA](https://www.tensorflow.org/xla/) (Accelerated Linear Algebra) is a
domain-specific compiler framework and execution environment for linear algebra,
which powers code-generation for ML frameworks like TensorFlow, JAX, and others.

A cornerstone of XLA is the HLO (High Level Optimizer) IR, which offers a
carefully fixed selected list of operations, mostly orthogonal to each other. It
provides an efficient optimizer for computations expressed with this set of
operations and generate codes for hardware platforms like CPU, GPU, and TPUs.
Its goal is to provide a uniform interface to compile and execute these
optimized HLO programs independently of the targeted device. It is not a
front-end ML system like TensorFlow or JAX, rather it is a backend framework
that optimizes HLO and lowers to machine code.

The HLO set of operations is closed and has well defined semantics. HLO
operations operate on immutable Tensors with static shapes (actually bounded
shapes to be exact) and explicit broadcasts.

[MLIR](https://mlir.llvm.org/) is a compiler infrastructure which intends to
come with "battery included", as such it intends to provide all the blocks
required to assemble graph optimization and codegen pipelines. The longer term
roadmap for MLIR is to provide a
[Tensor Compute Primitive](https://llvm.discourse.group/c/mlir/MLIR-TCP-WG/36)
(TCP) dialect, which should hopefully be general enough to model what HLO
represents today (see
[slides](https://drive.google.com/open?id=1iljcpTQ5NPaMfGpoPDFml1XkYxjK_6A4) and
[recording](https://drive.google.com/open?id=1jSPa8TwPKUt0WuLquGc8OgSUVYJHMvWZ)
for a technical discussion on this topic).

The work on MLIR-HLO can be seen as a stepping stone towards building TCP, while
integrating intermediate components into XLA itself by relying on the
well-proven HLO IR and introducing more pieces from upstream MLIR
([Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/),
[Vector](https://mlir.llvm.org/docs/Dialects/Vector/),
[GPU](https://mlir.llvm.org/docs/Dialects/GPU/) dialect, ...).
[This document](https://www.tensorflow.org/mlir/xla_gpu_codegen) provides more
information on the current migration of the XLA GPU codegen.

## MLIR Dialects for XLA-style compilation

This repository defines three dialects to support a HLO-like compilation
pipeline using MLIR:

*   `chlo`: the "client" HLO dialect, intended to be closer to the frontend
    (including implicit broadcast semantics).
*   `mhlo`: "meta"-HLO dialect ; similar to `xla_hlo`, but with extensions for
    dynamic shape support.
*   `lmhlo`: "late"-"meta"-HLO, it is the IR after buffer allocation is
    performed. In XLA the buffer allocation is a side-data structure which keeps
    track of these informations, while this separate dialect materializes it in
    the IR.

We describe these in more details below.

### HLO Client Dialect: `chlo`.

*   It was originally designed to map the
    [XLA client APIs](https://www.tensorflow.org/xla/operation_semantics) (e.g.,
    ops supports implicit broadcast and roughly modeled on XlaBuilder API)
    modulo support for dynamic shapes and additional ops required to support
    dynamic client side HLOs.
*   Ops can be from either the XlaBuilder or XLA helper functions can be
    converted into ops (e.g., given ambiguity in what constitutes these ops,
    there is some freedom to decide), the goal of this dialect is to correspond
    close to client level and enable a thin layer between client use and op
    construction (making it cheap to construct and optimizations on the dialect
    close to optimizations on the client ops).

Entry:

*   The vast majority of old "client" interactions are via the XlaBuilder APIs.
    These APIs are used by TF2XLA kernels, JAX, PyTorch bridge and directly. The
    legalization path (described below) can also reuse the XlaBuilder's APIs to
    construct XLA Client HLO ops directly (this uses MlirXlaBuilder which is a
    subclass of XlaBuilder).
*   The other entry point is during legalization from TensorFlow ops in the TF
    Graph Compiler and other tools (e.g., SavedModel lowering and TFCompile).

Exit:

*   MHLO
*   May be exported to xla::HloInstructionProto by invoking the XlaBuilder APIs
    (with regular XlaBuilder)

The `chlo` dialect started originally as mapping to the XLA client Builder APIs.
It enables it to both be constructed and converted back to existing XLA
interfaces using the XlaBuilder API. Due to the way that translation into and
out of the dialect works, there is no expectation that this dialect roundtrips
to XLA (e.g., it is only intended to be translated to MLIR and then legalized to
another dialect or translated to HloInstructionProto).

The export approach of reusing the XlaBuilders enables reusing a lot of logic
that was already implemented in terms of computing shapes, inserting broadcasts
etc.

An important topic here is that XLA Client HLO ops are not a well defined set.
And in particular what some would consider helper functions, others would
consider ops. It should be easy to move between these and so define a new op
along with the helper function or autogenerate the helper functions from the
descriptions of the ops. For the former, a simple approach would be to simply
consider the context in which the op is being constructed and if an MLIR one,
construct a op in the client dialect instead of further calls into XlaBuilder.
The latter could be implemented by adding the op and a legalization of the op to
other known ops, from which a helper function can get generated that could be
used as regular.

Status: Exists but need to be cleaned up.

### Meta HLO Dialect `mhlo`

*   Dialect is closer to current HLO server ops (e.g., no implicit broadcast)
*   MHLO dialect where we can deviate from the requirements of the client or
    server dialect, in particular:
    *   Control flow ops with implicit capture to enable simpler optimizations
        (e.g., generic LICM, unroll & jam, etc.)
    *   Multiple results ops (e.g., no tuples)
    *   More ops (for example, unique op or assert op), and ops that don't need
        to be added to either client or server dialect.
    *   Op set not constrained by implementation (e.g., hlo.add operating on say
        i79 or !mydialect.weird_type is allowed even though no XLA backend
        supports it). Verification on types happening at the boundaries.
    *   It does not need to preserve some deprecated XLA constructs (e.g.
        stateful RNG HLO).
    *   More dynamic shape support ops without need for updating all
        users/backends.
*   This dialect enables evolving HLO independently from XLA in order to
    experiment with features we'd like to upstream in MLIR TCP. In particular it
    intends to be user-extensible through
    [interfaces](https://mlir.llvm.org/docs/Interfaces/).
*   It should have no TensorFlow, or proto, or other Google internal
    dependencies.
*   It need not be a complete superset of ops compared to XLA HLO dialect.

Entry:

*   Legalization from `chlo` dialect or conversion from XLA HLO.
*   Directly emitted from TF Graph Compiler;
*   Builder call (e.g., EDSL);

Exit:

*   LMHLO, Linalg IREE, directly used in codegen.
*   XLA HLO.

The MHLO dialect has no direct export format, it is only meant as an
intermediate optimization dialect/format. It is also where we can experiment
cheaply with new ops. This format will be where the representation would differ
from existing endpoints.

Status: Exists but need to be cleaned up and evolved, in particular with respect
to supporting dynamic shapes.

MHLO differs from XLA HLO op set in multiple ways, including:
1. MHLO While accepts multiple operands and may produce multiple results
   instead;

### LMHLO

LMHLO corresponds to late `mhlo` and operates on buffer domain (e.g., memref)
with side-effecting operations. The lowering from `mhlo` dialect proceeds by way
of scheduling, memory and buffer allocation. The current mapping is directly on
XLA Client HLOs but without implicit broadcast and with operation on memrefs.
This dialect will instead be rebased on `mhlo` dialect but operating on buffers
still.

Entry:

*   Post buffer assignment on `mhlo` dialect, or from XLA after buffer
    assignment.

Exit:

*   Codegen (LLVM IR in the common cases at the moment)

## End-to-End pipeline

TODO

## Alternative build setups

### Building Python API

Building the MHLO Python API requires building as an LLVM external project.
The below instructions presume that you have this `mlir-hlo` repo and an
`llvm-project` repo checked out side by side.

Note that the python package produced by this procedure includes the `mlir`
package and is not suitable for deployment as-is (but it can be included into
a larger aggregate).

```
mkdir build && cd build
cmake -GNinja -B. ${LLVM_SRC_DIR}/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=mlir_hlo \
    -DLLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR=${MLIR_HLO_SRC_DIR} \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DPython3_EXECUTABLE=$(which python) \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMHLO_ENABLE_BINDINGS_PYTHON=ON

ninja MLIRHLOPythonModules
export PYTHONPATH=$PWD/tools/mlir_hlo/python_packages/mlir_hlo
python -c "import mlir.dialects.mhlo"
```

## External projects that depend on mlir-hlo

External projects that need to depend on `mlir-hlo` (for example via a git
submodule) can use the following setting in their cmake configuration in order
for `find_package(MHLO)` to import all mlir-hlo cmake targets into their build
setup and have access to the required include and lib variables (see generated
`MHLOConfig.cmake`).

```
...
   -DMHLO_DIR=<path to mlir-hlo build dir>/lib/cmake/mlir-hlo
   ...
```
