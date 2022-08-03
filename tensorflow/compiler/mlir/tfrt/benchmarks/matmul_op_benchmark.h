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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_MATMUL_BENCHMARK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_MATMUL_BENCHMARK_H_

#include <utility>

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/utils/host_context.h"

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
using ::tfrt::jitrt::Executable;
using ::tfrt::jitrt::HostContextAsyncTaskRunner;
using ::tfrt::jitrt::JitExecutable;
using ::tfrt::jitrt::MemrefDesc;
using ::tfrt::jitrt::RemainingResultsConverter;

// -------------------------------------------------------------------------- //
// Run benchmark by compiling MLIR function using TFRT JitRt API.
// -------------------------------------------------------------------------- //

template <typename T>
void RunMatMulMlirBenchmark(::testing::benchmark::State& state,
                            llvm::StringRef mlir_input,
                            llvm::StringRef function_name) {
  // MatMul: [m, k] x [k, n]
  ssize_t m = state.range(0);
  ssize_t k = state.range(1);
  ssize_t n = state.range(2);

  std::unique_ptr<HostContext> host = CreateSingleThreadedHostContext();

  TfJitRtPipelineOptions tf_jitrt_opts;
  JitExecutable& jit_executable =
      CreateJitExecutable(*host, mlir_input, function_name,
                          /*lower_from_tensorflow=*/true, tf_jitrt_opts);

  // Build an ExecutionContext from the HostContext.
  llvm::Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(host.get(), /*resource_context=*/nullptr).build();
  tfrt::ExecutionContext exec_ctx(std::move(*req_ctx));

  // Generate random input data.
  std::array<ssize_t, 2> lhs_dims = {m, k};
  std::array<ssize_t, 2> rhs_dims = {k, n};

  Eigen::Tensor<T, 2, Eigen::RowMajor> lhs = GenRandomTensor<T, 2>(lhs_dims);
  Eigen::Tensor<T, 2, Eigen::RowMajor> rhs = GenRandomTensor<T, 2>(rhs_dims);

  std::array<MemrefDesc, 2> operands = {TensorToMemrefDesc(lhs),
                                        TensorToMemrefDesc(rhs)};

  auto result_values = std::array<RCReference<AsyncValue>, 2>{{}};
  RemainingResults results(result_values);

  // Record data ptrs of inputs.
  llvm::SmallVector<void*> input_ptrs;
  for (auto& operand : operands) {
    input_ptrs.push_back(operand.data());
  }

  // Free memory owned by the returned memrefs.
  ResultConversionCtx result_ctx(std::move(input_ptrs));
  RemainingResultsConverter<ResultConversionCtx> converter(results, result_ctx);
  converter.AddConversion(FreeReturnedMemref);

  // Execute async tasks in the HostContext work queue.
  Executable::ExecuteOpts opts;
  HostContextAsyncTaskRunner async_task_runner(host.get());
  opts.async_task_runner = &async_task_runner;

  // Get an executable that might be specialized to the operands.
  llvm::Expected<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(operands);
  if (auto err = executable.takeError())
    LOG(FATAL) << "Failed to specialize executable";

  // Wait for the compilation completion.
  host->Await({executable->CopyRef()});

  CHECK(!executable->IsError())
      << "Failed to get executable: " << StrCat(executable->GetError());
  CHECK(!(*executable)->IsAsync()) << "async results are not supported";

  // Initialize call frame with MemrefDesc operands.
  Executable::CallFrame call_frame;
  if (auto err = (*executable)->InitializeCallFrame(operands, &call_frame))
    LOG(FATAL) << "Failed to initialize call frame";

  for (auto _ : state) {
    (*executable)->Execute(call_frame, opts);
    if (auto err = (*executable)->ReturnResults(converter, &call_frame))
      LOG(FATAL) << "Failed to return compiled kernel results";
  }

  state.SetItemsProcessed(state.iterations() * m * k * n);
}

// -------------------------------------------------------------------------- //
// Run benchmark using Eigen expression evaluation.
// -------------------------------------------------------------------------- //

template <typename T>
void RunMatMulEigenBenchmark(::testing::benchmark::State& state) {
  // MatMul: [m, k] x [k, n]
  ssize_t m = state.range(0);
  ssize_t k = state.range(1);
  ssize_t n = state.range(2);

  // Generate random input data.
  std::array<ssize_t, 2> lhs_dims = {m, k};
  std::array<ssize_t, 2> rhs_dims = {k, n};

  Eigen::Tensor<T, 2, Eigen::RowMajor> lhs = GenRandomTensor<T, 2>(lhs_dims);
  Eigen::Tensor<T, 2, Eigen::RowMajor> rhs = GenRandomTensor<T, 2>(rhs_dims);

  using Device = Eigen::DefaultDevice;
  Device d;

  Eigen::Tensor<T, 2, Eigen::RowMajor> dst(m, n);
  dst.setZero();

  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
  contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);

  for (auto _ : state) {
    auto expr = lhs.contract(rhs, contract_pairs);

    using Dst = decltype(dst);
    using Expr = decltype(expr);
    ExecuteAssignOp</*vectorize=*/true, Device, Dst, Expr>::run(d, dst, expr);
  }

  state.SetItemsProcessed(state.iterations() * m * k * n);
}

}  // namespace tensorflow

// -------------------------------------------------------------------------- //
// Macros to dispatch to different MatMul shapes.
// -------------------------------------------------------------------------- //

#define BM_TFMlir(NAME, MLIR_INPUT, FN, TYPE)                               \
  static void BM_mlir_##NAME##_##TYPE(::testing::benchmark::State& state) { \
    RunMatMulMlirBenchmark<TYPE>(state, MLIR_INPUT, FN);                    \
  }                                                                         \
  BENCHMARK(BM_mlir_##NAME##_##TYPE)

#define BM_Eigen(NAME, TYPE)                                                 \
  static void BM_eigen_##NAME##_##TYPE(::testing::benchmark::State& state) { \
    RunMatMulEigenBenchmark<TYPE>(state);                                    \
  }                                                                          \
  BENCHMARK(BM_eigen_##NAME##_##TYPE)

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_MATMUL_BENCHMARK_H_
