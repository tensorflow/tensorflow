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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_CWISE_UNARY_BENCHMARK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_CWISE_UNARY_BENCHMARK_H_

#include <utility>

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// This header is a part of the library with private visibility and will be
// used only to build benchmarks for different functions in this folder, so
// it is ok to put convenience using-declarations here.

using ::tfrt::AsyncValue;
using ::tfrt::AsyncValuePtr;
using ::tfrt::HostContext;
using ::tfrt::RCReference;
using ::tfrt::RemainingResults;
using ::tfrt::RequestContext;
using ::tfrt::RequestContextBuilder;
using ::tfrt::cpu::jit::Executable;
using ::tfrt::cpu::jit::JitExecutable;
using ::tfrt::cpu::jit::MemrefDesc;
using ::tfrt::cpu::jit::ReturnValueConverter;

// -------------------------------------------------------------------------- //
// Run benchmark by compiling MLIR function using TFRT CPURT API.
// -------------------------------------------------------------------------- //

template <typename T, int rank>
struct MlirBenchmark {
  std::unique_ptr<HostContext> host;
  const Executable* executable;
  tfrt::ExecutionContext exec_ctx;
  ReturnValueConverter<ResultConversionCtx> converter;
};

template <typename T, int rank>
MlirBenchmark<T, rank> PrepareUnaryMlirBenchmark(
    llvm::StringRef mlir_input, llvm::StringRef function_name,
    std::array<MemrefDesc, 1>& operands, size_t num_threads,
    bool lower_from_tensorflow, bool vectorize) {
  static_assert(rank >= 1 && rank <= 4, "We do only support ranks 1 to 4");
  std::unique_ptr<HostContext> host =
      num_threads > 0 ? CreateMultiThreadedHostContext(num_threads)
                      : CreateSingleThreadedHostContext();

  TfCpuRtPipelineOptions tf_cpurt_opts;
  tf_cpurt_opts.vectorize = vectorize;
  JitExecutable& jit_executable = CreateJitExecutable(
      *host, mlir_input, function_name, lower_from_tensorflow, tf_cpurt_opts);

  // Build an ExecutionContext from the HostContext.
  llvm::Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(host.get(), /*resource_context=*/nullptr).build();
  tfrt::ExecutionContext exec_ctx(std::move(*req_ctx));

  auto result_values = std::array<RCReference<AsyncValue>, 1>{{}};
  RemainingResults results(result_values);

  // Free memory owned by the returned memrefs.
  ReturnValueConverter<ResultConversionCtx> converter(results);
  converter.AddConversion(FreeReturnedMemref);

  // Get an executable that might be specialized to the operands.
  llvm::Expected<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(operands, exec_ctx);
  if (auto err = executable.takeError())
    LOG(FATAL) << "Failed to specialize executable";

  // Wait for the compilation completion.
  host->Await({executable->CopyRef()});

  CHECK(!executable->IsError())
      << "Failed to get executable: " << StrCat(executable->GetError());
  CHECK(!(*executable)->IsAsync()) << "async results are not supported";

  return {std::move(host), &executable->get(), exec_ctx, std::move(converter)};
}

template <typename T, int rank>
void TestUnaryMlirBenchmark(llvm::StringRef mlir_input,
                            llvm::StringRef function_name, T scale, T offset,
                            size_t num_threads, bool lower_from_tensorflow,
                            bool vectorize) {
  std::array<ssize_t, rank> input_dims;
  for (int d = 0; d < rank; ++d)
    input_dims[d] = 10;  // The value here does not matter.

  // Generate random input data.
  Eigen::Tensor<T, rank, Eigen::RowMajor> input =
      GenRandomTensor<T, rank>(input_dims, scale, offset);
  std::array<MemrefDesc, 1> operands = {TensorToMemrefDesc(input)};

  MlirBenchmark<T, rank> b = PrepareUnaryMlirBenchmark<T, rank>(
      mlir_input, function_name, operands, num_threads, lower_from_tensorflow,
      vectorize);

  // Initialize call frame with MemrefDesc operands.
  Executable::CallFrame call_frame;
  if (auto err = b.executable->InitializeCallFrame(operands, &call_frame))
    LOG(FATAL) << "Failed to initialize call frame";

  // Execute once.
  b.executable->Execute(call_frame, b.exec_ctx);
  if (auto err =
          b.executable->ReturnResults(b.converter, b.exec_ctx, &call_frame))
    LOG(FATAL) << "Failed to return compiled kernel results";
}

template <typename T, int rank>
void RunUnaryMlirBenchmark(::testing::benchmark::State& state,
                           llvm::StringRef mlir_input,
                           llvm::StringRef function_name, T scale, T offset,
                           size_t num_threads, bool lower_from_tensorflow,
                           bool vectorize) {
  std::array<ssize_t, rank> input_dims;
  for (int d = 0; d < rank; ++d) input_dims[d] = state.range(d);
  // Generate random input data.
  Eigen::Tensor<T, rank, Eigen::RowMajor> input =
      GenRandomTensor<T, rank>(input_dims, scale, offset);
  std::array<MemrefDesc, 1> operands = {TensorToMemrefDesc(input)};

  MlirBenchmark<T, rank> b = PrepareUnaryMlirBenchmark<T, rank>(
      mlir_input, function_name, operands, num_threads, lower_from_tensorflow,
      vectorize);

  // Initialize call frame with MemrefDesc operands.
  Executable::CallFrame call_frame;
  if (auto err = b.executable->InitializeCallFrame(operands, &call_frame))
    LOG(FATAL) << "Failed to initialize call frame";

  for (auto _ : state) {
    call_frame.args[0] = nullptr;  // reset kernel context argument
    b.executable->Execute(call_frame, b.exec_ctx);
    if (auto err =
            b.executable->ReturnResults(b.converter, b.exec_ctx, &call_frame))
      LOG(FATAL) << "Failed to return compiled kernel results";
  }

  state.SetItemsProcessed(state.iterations() * input.size());
}

// -------------------------------------------------------------------------- //
// Run benchmark using Eigen expression evaluation.
// -------------------------------------------------------------------------- //

template <typename T, int rank, bool vectorize, typename ExprBuilder>
void RunUnaryEigenBenchmark(::testing::benchmark::State& state,
                            ExprBuilder expr_builder, T scale, T offset,
                            size_t num_threads) {
  static_assert(rank >= 1 && rank <= 4, "We do only support ranks 1 to 4");
  std::array<ssize_t, rank> input_dims;
  for (int d = 0; d < rank; ++d) input_dims[d] = state.range(d);

  Eigen::Tensor<T, rank, Eigen::RowMajor> input =
      GenRandomTensor<T, rank>(input_dims, scale, offset);

  Eigen::DefaultDevice singleThreadedDevice;
  Eigen::ThreadPool thread_pool(num_threads);
  llvm::Optional<Eigen::ThreadPoolDevice> multiThreadedDevice;
  if (num_threads > 0) multiThreadedDevice.emplace(&thread_pool, num_threads);

  Eigen::DSizes<ssize_t, rank> dsizes;
  for (int d = 0; d < rank; ++d) dsizes[d] = input_dims[d];
  Eigen::Tensor<T, rank, Eigen::RowMajor> dst(dsizes);
  dst.setZero();

  for (auto _ : state) {
    auto expr = expr_builder(input);

    using Dst = decltype(dst);
    using Expr = decltype(expr);
    if (multiThreadedDevice.hasValue()) {
      ExecuteAssignOp</*vectorize=*/true, Eigen::ThreadPoolDevice, Dst,
                      Expr>::run(*multiThreadedDevice, dst, expr);
    } else {
      ExecuteAssignOp</*vectorize=*/true, Eigen::DefaultDevice, Dst, Expr>::run(
          singleThreadedDevice, dst, expr);
    }
  }

  state.SetItemsProcessed(state.iterations() * input.size());
}

}  // namespace tensorflow

// -------------------------------------------------------------------------- //
// Macros to dispatch to different benchmark based on rank, shape and data type.
//
// Input data is generated with: scale * (Eigen::random<T>() + offset).
// For MLIR benchmarks, we also generate a unit test to detect regressions.
// -------------------------------------------------------------------------- //

#define BM_TFMlir(NAME, MLIR_INPUT, FN, RANK, TYPE, SCALE, OFFSET, \
                  NUM_THREADS)                                     \
  TEST(Test_mlir_##NAME##_##TYPE##_##NUM_THREADS, RunOnce) {       \
    TestUnaryMlirBenchmark<TYPE, RANK>(                            \
        MLIR_INPUT, FN, SCALE, OFFSET, NUM_THREADS,                \
        /*lower_from_tensorflow=*/true, /*vectorize=*/false);      \
  }                                                                \
  static void BM_mlir_##NAME##_##TYPE##_##NUM_THREADS(             \
      ::testing::benchmark::State& state) {                        \
    RunUnaryMlirBenchmark<TYPE, RANK>(                             \
        state, MLIR_INPUT, FN, SCALE, OFFSET, NUM_THREADS,         \
        /*lower_from_tensorflow=*/true, /*vectorize=*/false);      \
  }                                                                \
  BENCHMARK(BM_mlir_##NAME##_##TYPE##_##NUM_THREADS)->MeasureProcessCPUTime()

#define BM_TFMlirVectorized(NAME, MLIR_INPUT, FN, RANK, TYPE, SCALE, OFFSET, \
                            NUM_THREADS)                                     \
  TEST(Test_mlir_v_##NAME##_##TYPE##_##NUM_THREADS, RunOnce) {               \
    TestUnaryMlirBenchmark<TYPE, RANK>(                                      \
        MLIR_INPUT, FN, SCALE, OFFSET, NUM_THREADS,                          \
        /*lower_from_tensorflow=*/true, /*vectorize=*/true);                 \
  }                                                                          \
  static void BM_mlir_v_##NAME##_##TYPE##_##NUM_THREADS(                     \
      ::testing::benchmark::State& state) {                                  \
    RunUnaryMlirBenchmark<TYPE, RANK>(                                       \
        state, MLIR_INPUT, FN, SCALE, OFFSET, NUM_THREADS,                   \
        /*lower_from_tensorflow=*/true, /*vectorize=*/true);                 \
  }                                                                          \
  BENCHMARK(BM_mlir_v_##NAME##_##TYPE##_##NUM_THREADS)->MeasureProcessCPUTime()

#define BM_Mlir(NAME, MLIR_INPUT, FN, RANK, TYPE, SCALE, OFFSET, NUM_THREADS) \
  TEST(Test_mlir_##NAME##_##TYPE##_##NUM_THREADS, RunOnce) {                  \
    TestUnaryMlirBenchmark<TYPE, RANK>(                                       \
        MLIR_INPUT, FN, SCALE, OFFSET, NUM_THREADS,                           \
        /*lower_from_tensorflow=*/false, /*vectorize=*/false);                \
  }                                                                           \
  static void BM_mlir_##NAME##_##TYPE##_##NUM_THREADS(                        \
      ::testing::benchmark::State& state) {                                   \
    RunUnaryMlirBenchmark<TYPE, RANK>(                                        \
        state, MLIR_INPUT, FN, SCALE, OFFSET, NUM_THREADS,                    \
        /*lower_from_tensorflow=*/false, /*vectorize=*/false);                \
  }                                                                           \
  BENCHMARK(BM_mlir_##NAME##_##TYPE##_##NUM_THREADS)->MeasureProcessCPUTime()

#define BM_EigenScalar(NAME, FN, RANK, TYPE, SCALE, OFFSET, NUM_THREADS) \
  static void BM_eigen_s_##NAME##_##TYPE##_##NUM_THREADS(                \
      ::testing::benchmark::State& state) {                              \
    RunUnaryEigenBenchmark<TYPE, RANK, false>(state, FN, SCALE, OFFSET,  \
                                              NUM_THREADS);              \
  }                                                                      \
  BENCHMARK(BM_eigen_s_##NAME##_##TYPE##_##NUM_THREADS)->MeasureProcessCPUTime()

#define BM_EigenVectorized(NAME, FN, RANK, TYPE, SCALE, OFFSET, NUM_THREADS) \
  static void BM_eigen_v_##NAME##_##TYPE##_##NUM_THREADS(                    \
      ::testing::benchmark::State& state) {                                  \
    RunUnaryEigenBenchmark<TYPE, RANK, true>(state, FN, SCALE, OFFSET,       \
                                             NUM_THREADS);                   \
  }                                                                          \
  BENCHMARK(BM_eigen_v_##NAME##_##TYPE##_##NUM_THREADS)->MeasureProcessCPUTime()

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_CWISE_UNARY_BENCHMARK_H_
