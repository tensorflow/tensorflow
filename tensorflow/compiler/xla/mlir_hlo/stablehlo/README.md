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
GitHub discussions to have longer discussions. We also have a `#stablehlo`
channel on [the OpenXLA Discord server](https://discord.gg/PeWUTaecrA).

## Community

With StableHLO, our goal is to create a community to build an amazing
portability layer between ML frameworks and ML compilers. Let's work together
on figuring out the appropriate governance to make this happen.

## Roadmap

* Workstream #1: Stable version of HLO/MHLO, including
  [the spec](https://github.com/openxla/stablehlo/labels/Spec),
  the corresponding dialect with high-quality implementations of
  [prettyprinting](https://github.com/openxla/stablehlo/labels/Prettyprinting),
  [verification](https://github.com/openxla/stablehlo/labels/Verification) and
  [type inference](https://github.com/openxla/stablehlo/labels/Type%20inference),
  and [the interpeter](https://github.com/openxla/stablehlo/labels/Interpreter)
  - ETA: H2 2022.
* Workstream #2: Evolution beyond what's currently in HLO/MHLO.
  Ongoing work on [dynamism](https://github.com/openxla/stablehlo/labels/Dynamism),
  sparsity, quantization and extensibility - ETA: H2 2022.
* Workstream #3: Support for ML frameworks (TensorFlow, JAX, PyTorch) and
  ML compilers (XLA and IREE) - ETA: H2 2022.
