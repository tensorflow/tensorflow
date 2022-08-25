# StableHLO

StableHLO is an operation set that expresses ML computations. It has been
originally bootstrapped from
[the MHLO dialect](https://github.com/tensorflow/mlir-hlo#meta-hlo-dialect-mhlo)
and enhances it with additional functionality, including serialization and
versioning.

StableHLO is a portability layer between ML frameworks and ML compilers.
We are aiming for adoption by a wide variety of ML frameworks including
TensorFlow, JAX and PyTorch, as well as ML compilers including XLA and IREE.

## Development

We're using GitHub issues / pull requests to organize development and
GitHub discussions to have longer discussions. We'll also set up a Discord
server shortly.

## Community

With StableHLO, our goal is to create a community to build an amazing
portability layer between ML frameworks and ML compilers. Let's work together
on figuring out the appropriate governance to make this happen.

## Roadmap

* Workstream #1: Stable version of HLO/MHLO.
  Specification, test suite, reference implementation - ETA: H2 2022
* Workstream #2: Evolution beyond what's currently in HLO/MHLO.
  Ongoing work on dynamism, sparsity, quantization and extensibility -
  ETA: H2 2022.
* Workstream #3: Support for ML frameworks (TensorFlow, JAX, PyTorch) and
  ML compilers (XLA and IREE) - ETA: H2 2022.
