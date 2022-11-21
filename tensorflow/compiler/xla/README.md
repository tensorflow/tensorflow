<p align="center">
  <img width="200" src="./g3doc/images/xlalogo.png"/>
</p>

XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear
algebra that optimizes TensorFlow computations. See the
[documentation](./g3doc/index.md).

This directory is currently migrating to [OpenXLA](https://github.com/openxla/)
and will be the root of the [openxla/xla](https://github.com/openxla/xla)
repository.

== Directory Structure ==

We're currently re-organizing the directory structure, the end result should be
that no sources are directly present at the top-level. Here is the current plan
for the directory layout:

*   backends/ (created from directories under xla/service)
    *   cpu/
    *   gpu/
    *   interpreter/
    *   ...
*   hlo/ (created from xla/service/ mostly, no sources expected directly here)
    *   client/ (created from xla/client)
    *   evaluator/ (created from the relevant files in xla/service)
    *   experimental/ (created from xla/experimental)
    *   ir/ (created from the relevant files in xla/service)
    *   python/ (created from xla/python)
    *   tests/ (created from xla/tests)
    *   transforms/ (created from the relevant files in xla/service)
    *   utils/ (created from the relevant files in xla/service)
*   mlir/ (also exported as the root of https://github.com/tensorflow/mlir-hlo
    and building with CMake)
    *   CMakeLists.txt (just like now for mlir-hlo repo).
    *   backends/ (same as xla/backends/ but for the MLIR specific bits: this is
        a short-term solution pending more convergence / XLA Next)
        *   cpu
        *   gpu (populated from /compiler/xla/mlir/transforms/gpu/passes.td,
            will contain all the glue for e2e GPU compilation)
    *   bindings/
        *   c/ (bootstrapped from mlir/hlo/{include,lib}/mlir-hlo-c)
        *   python/ (bootstrapped from mlir/hlo/python, should talk about some
            low-level LAX?)
    *   integration_tests/ (to be defined / refined)
    *   tools/ (xla-opt, fuzzer, ir-reducer, interpreter/evaluator)
    *   transforms/ (generic / cross dialect transforms)
    *   utils/
*   // below are dialects and transforms folders
    *   framework/ (moved from compiler/mlir/xla/ir/xla_framework_ops.td)
    *   gml_st
        *   gmlst-opt.cc
        *   gmlst-runner.cc (runner tool that can execute IR at ~gmlst level)
        *   ir/
        *   integration_test (tests that run things: Tensor(s) in -> Tensor(s)
            out)
        *   test (IR -> IR tests for passes interaction)
        *   transforms/
            *   bufferize_tiled_loop/
                *   bufferize_tiled_loop.cc
                *   bufferize_tiled_loop.h
            *   ...
    *   lhlo_gpu/
    *   mhlo/
        *   mhlo-opt.cc
        *   analysis/
            *   dataflow/
                *   dataflow.h
                *   dataflow.cc
                *   test_pass.cc // test_only target, linked into opt tool for
                    testing only.
        *   integration_test (tests that run things: Tensor(s) in -> Tensor(s)
            out)
        *   ir/ (dialect definition)
        *   test (IR -> IR tests for passes interaction)
        *   transforms/
            *   materialize_broadcasts/
                *   materialize_broadcasts.cc
                *   materialize_broadcasts.h // headers stays with the source
                *   broadcast_analysis.{cc, h} // private analysis/utils needed
                    for this pass
                *   test/ (.mlir unit-tests are collocated with the pass
                    itself).
            *   …
            *   passes.td // enables group registration for all passes.
        *   utils/
    *   thlo/
    *   runtime/
*   pjrt/ (created from xla/pjrt)
*   rpc/ (created from xla/rpc)
*   runtime/
*   stream_executor/ (moved from TensorFlow)
*   third_party/ (vendoring of TSL base library)
*   tools/ (created from mlir/hlo/tools and xla/tools)
*   translate/ (StableHLO to MHLO, MHLO to HLO, HLO to MHLO, MHLO to TOSA)
