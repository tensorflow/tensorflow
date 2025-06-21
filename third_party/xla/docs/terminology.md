# XLA Terminology

There are several terms that are used in the context of XLA, MLIR, LLVM, and
other related technologies. Below is a partial list of these terms and their
definitions.

- **OpenXLA**
  - OpenXLA is an open ecosystem of performant, portable, and extensible machine
  learning (ML) infrastructure
  components that simplify ML development by defragmenting the tools between
  frontend frameworks and hardware backends. It includes the XLA compiler,
  StableHLO, VHLO, [PJRT](https://openxla.org/xla/pjrt) and other
  components.
- **XLA**
  - XLA (Accelerated Linear Algebra) is an open source compiler for machine
  learning. The XLA compiler takes models from popular frameworks such as
  PyTorch, TensorFlow, and JAX, and optimizes the models for high-performance
  execution across different hardware platforms including GPUs, CPUs, and ML
  accelerators. The XLA compiler outputs some code to LLVM, some to "standard"
  MLIR, and some to [Triton MLIR](https://triton-lang.org/main/dialects/dialects.html)
  that is processed by (MLIR-based) OpenAI Triton compiler.
- **PJRT**
  - [PJRT](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h) is
  a uniform Device API that simplifies the growing complexity of ML workload
  execution across hardware and frameworks. It provides a hardware and framework
  independent interface for compilers and runtimes.
- **StableHLO**
  - StableHLO is the public interface to OpenXLA, it is a standardized MLIR
  dialect that may be used by different frameworks and compilers in the OpenXLA
  ecosystem. XLA supports StableHLO, and immediately converts it to HLO on the
  input. There are some [StableHLO to StableHLO](https://openxla.org/stablehlo/generated/stablehlo_passes)
  passes implemented using the MLIR framework. It is also possible to convert
  StableHLO to other compilers' IR without using HLO, for example in cases where
  an existing IR is more appropriate.
- **CHLO**
  - CHLO is a collection of higher level operations which are optionally
  decomposable to StableHLO.
- **VHLO**
  - The [VHLO Dialect](https://openxla.org/stablehlo/vhlo) is a MLIR dialect
  that is a compatibility layer on top of StableHLO. It provides a snapshot of
  the StableHLO dialect at a given point in time by versioning individual
  program elements, and is used for serialization and stability.
- **MHLO**
  - MHLO is a standalone MLIR-based representation of XLA's HLO IR. The dialect
  is being evaluated for deprecation, and new users of the dialect should prefer
  to use StableHLO instead.
- **HLO**
  - HLO is an internal graph representation (IR) for the XLA compiler (and also
  supported input). It is **not** based on MLIR, and has its own textual syntax
  and binary (protobuf based) representation.
- **MLIR**
    - [MLIR](https://mlir.llvm.org) is a hybrid IR infrastructure that
    allows users to define "dialects" of operations at varying degrees of
    abstraction, and gradually lower between these opsets, performing
    transformations at each level of granularity. StableHLO and CHLO are two
    examples of MLIR dialects.
- **LLVM**
    - [LLVM](https://llvm.org/) is a compiler backend, and a language that it
    takes as an input. Many compilers generate LLVM code as a first step, and
    then LLVM generates machine code from it. This allows developers to reuse
    code that is similar in different compilers, and also makes supporting
    different target platforms easier. XLA:GPU and CPU backends have
    [LLVM IR emitters](https://github.com/openxla/xla/tree/main/xla/service/llvm_ir)
    for targeting specific hardware.
