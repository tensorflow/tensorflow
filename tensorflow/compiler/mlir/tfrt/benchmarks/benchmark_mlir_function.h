/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_BENCHMARK_MLIR_FUNCTION_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_BENCHMARK_MLIR_FUNCTION_H_

#include <functional>

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"

namespace tensorflow {

struct InputTensorSpec {
  InputTensorSpec(DataType dtype, llvm::ArrayRef<ssize_t> dims)
      : dtype(dtype), dims(dims.begin(), dims.end()) {}

  DataType dtype;
  llvm::SmallVector<ssize_t> dims;
};

// Benchmark arbitrary Tensorflow dialect MLIR function using inputs of given
// type and shape by compiling it using TF JitRt pipeline and the TFRT runtime.
void RunJitRtBenchmark(::testing::benchmark::State& state,
                       llvm::StringRef mlir_input,
                       llvm::StringRef function_name,
                       llvm::ArrayRef<InputTensorSpec> input_specs,
                       bool vectorize = false, bool codegen_transpose = false);

// Benchmark arbitrary Tensorflow dialect MLIR function using inputs of given
// type and shape by compiling it to BEF with fallback kernels.
void RunTfrtBenchmark(::testing::benchmark::State& state,
                      llvm::StringRef mlir_input, llvm::StringRef function_name,
                      llvm::ArrayRef<InputTensorSpec> input_specs);

// Benchmark arbitrary compute function written as Eigen expression(s).
void RunEigenBenchmark(
    ::testing::benchmark::State& state,
    std::function<void(llvm::ArrayRef<Tensor>,
                       llvm::Optional<Eigen::ThreadPoolDevice>)>
        compute,
    llvm::ArrayRef<InputTensorSpec> input_specs);

// TODO(ezhulenev): Benchmarking macro should generate unit tests to verify
// that benchmarks at least do not crash with the specified inputs.

#define BM_Jitrt(NAME, MLIR_INPUT, FN, INPUT_SPEC)                  \
  static void BM_Jitrt_##NAME(::testing::benchmark::State& state) { \
    RunJitRtBenchmark(state, MLIR_INPUT, FN, INPUT_SPEC);           \
  }                                                                 \
  BENCHMARK(BM_Jitrt_##NAME)->MeasureProcessCPUTime()

#define BM_JitrtV(NAME, MLIR_INPUT, FN, INPUT_SPEC)                   \
  static void BM_Jitrtv_##NAME(::testing::benchmark::State& state) {  \
    RunJitRtBenchmark(state, MLIR_INPUT, FN, INPUT_SPEC, true, true); \
  }                                                                   \
  BENCHMARK(BM_Jitrtv_##NAME)->MeasureProcessCPUTime()

#define BM_Tfrt(NAME, MLIR_INPUT, FN, INPUT_SPEC)                  \
  static void BM_tfrt_##NAME(::testing::benchmark::State& state) { \
    RunTfrtBenchmark(state, MLIR_INPUT, FN, INPUT_SPEC);           \
  }                                                                \
  BENCHMARK(BM_tfrt_##NAME)->MeasureProcessCPUTime()

#define BM_Eigen(NAME, FN, INPUT_SPEC)                              \
  static void BM_eigen_##NAME(::testing::benchmark::State& state) { \
    RunEigenBenchmark(state, FN, INPUT_SPEC);                       \
  }                                                                 \
  BENCHMARK(BM_eigen_##NAME)->MeasureProcessCPUTime()

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_BENCHMARK_MLIR_FUNCTION_H_
