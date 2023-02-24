/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_FUSED_MATMUL_OP_BENCHMARK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_FUSED_MATMUL_OP_BENCHMARK_H_

#include <array>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/utils/host_context.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"

namespace tensorflow {

// This header is a part of the library with private visibility and will be
// used only to build benchmarks for different functions in this folder, so
// it is ok to put convenience using-declarations here.

std::string GetFusedMatmulIR(
    llvm::ArrayRef<int32_t> arg0_shape, llvm::ArrayRef<bool> arg0_dyn_dims,
    llvm::ArrayRef<int32_t> arg1_shape, llvm::ArrayRef<bool> arg1_dyn_dims,
    llvm::ArrayRef<int32_t> arg2_shape, llvm::ArrayRef<bool> arg2_dyn_dims,
    llvm::ArrayRef<int32_t> out_shape, llvm::ArrayRef<bool> out_dyn_dims,
    llvm::StringRef element_type, unsigned activation = 0,
    llvm::StringRef epsilon = "0.000000e+00",
    llvm::StringRef leakyrelu_alpha = "2.000000e-01");

using ::tfrt::AsyncValue;
using ::tfrt::AsyncValuePtr;
using ::tfrt::HostContext;
using ::tfrt::RCReference;
using ::tfrt::RemainingResults;
using ::tfrt::RequestContext;
using ::tfrt::RequestContextBuilder;
using ::tfrt::jitrt::HostContextAsyncTaskRunner;
using ::tfrt::jitrt::RemainingResultsConverter;
using ::xla::runtime::Executable;
using ::xla::runtime::JitExecutable;
using ::xla::runtime::MemrefDesc;

// -------------------------------------------------------------------------- //
// Run benchmark by compiling MLIR function using TFRT JitRt API.
// -------------------------------------------------------------------------- //

template <typename T>
void RunFusedMatMulMlirBenchmark(::testing::benchmark::State& state,
                                 llvm::StringRef mlir_input,
                                 llvm::StringRef function_name) {
  // MatMul: [arg0, arg1] x [arg1, arg2]
  ssize_t arg0 = state.range(0);
  ssize_t arg1 = state.range(1);
  ssize_t arg2 = state.range(2);
  ssize_t arg3 = state.range(3);

  std::unique_ptr<HostContext> host = CreateSingleThreadedHostContext();

  TfJitRtPipelineOptions tf_jitrt_opts;
  tf_jitrt_opts.vectorize = tensorflow::GetJitRtFlags().vectorize;

  JitExecutable& jit_executable =
      CreateJitExecutable(*host, mlir_input, function_name,
                          /*lower_from_tensorflow=*/true, tf_jitrt_opts);

  // Build an ExecutionContext from the HostContext.
  llvm::Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(host.get(), /*resource_context=*/nullptr).build();
  tfrt::ExecutionContext exec_ctx(std::move(*req_ctx));

  // Generate random input data.
  std::array<ssize_t, 2> lhs_dims = {arg0, arg1};
  std::array<ssize_t, 2> rhs_dims = {arg1, arg2};
  std::array<ssize_t, 1> bias_dims = {arg3};

  Eigen::Tensor<T, 2, Eigen::RowMajor> lhs = GenRandomTensor<T, 2>(lhs_dims);
  Eigen::Tensor<T, 2, Eigen::RowMajor> rhs = GenRandomTensor<T, 2>(rhs_dims);
  Eigen::Tensor<T, 1, Eigen::RowMajor> bias = GenRandomTensor<T, 1>(bias_dims);

  std::array<MemrefDesc, 3> operands = {TensorToMemrefDesc(lhs),
                                        TensorToMemrefDesc(rhs),
                                        TensorToMemrefDesc(bias)};

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
  absl::StatusOr<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(operands);
  if (!executable.ok()) LOG(FATAL) << "Failed to specialize executable";

  // Wait for the compilation completion.
  host->Await({executable->CopyRef()});

  CHECK(!executable->IsError())
      << "Failed to get executable: " << executable->GetError().message();
  CHECK(!(*executable)->IsAsync()) << "async results are not supported";

  // Initialize call frame with MemrefDesc operands.
  Executable::CallFrame call_frame;
  if (auto st = (*executable)->InitializeCallFrame(operands, &call_frame);
      !st.ok())
    LOG(FATAL) << "Failed to initialize call frame";

  for (auto _ : state) {
    (*executable)->Execute(call_frame, opts);
    if (auto st = (*executable)->ReturnResults(converter, &call_frame);
        !st.ok())
      LOG(FATAL) << "Failed to return compiled kernel results";
  }

  state.SetItemsProcessed(state.iterations() * arg0 * arg1 * arg2);
}

// -------------------------------------------------------------------------- //
// Run benchmark using Eigen expression evaluation.
// -------------------------------------------------------------------------- //

template <typename T>
void RunFusedMatMulEigenBenchmark(::testing::benchmark::State& state) {
  // MatMul: [arg0, arg1] x [arg1, arg2]
  ssize_t arg0 = state.range(0);
  ssize_t arg1 = state.range(1);
  ssize_t arg2 = state.range(2);
  ssize_t arg3 = state.range(3);

  // Generate random input data.
  std::array<ssize_t, 2> lhs_dims = {arg0, arg1};
  std::array<ssize_t, 2> rhs_dims = {arg1, arg2};
  std::array<ssize_t, 1> bias_dims = {arg3};

  Eigen::Tensor<T, 2, Eigen::RowMajor> lhs = GenRandomTensor<T, 2>(lhs_dims);
  Eigen::Tensor<T, 2, Eigen::RowMajor> rhs = GenRandomTensor<T, 2>(rhs_dims);
  Eigen::Tensor<T, 1, Eigen::RowMajor> bias = GenRandomTensor<T, 1>(bias_dims);

  using Device = Eigen::DefaultDevice;
  Device d;

  CHECK(d.numThreads() == 1) << "Executing Eigen in multi-threaded";

  Eigen::Tensor<T, 2, Eigen::RowMajor> dst(arg0, arg2);
  dst.setZero();

  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
  contract_pairs[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);

  BiasAddArgs<T> bias_add_args;

  bias_add_args.bias_add_data = bias.data();
  auto activation = static_cast<FusedComputationType>(state.range(4));

  using OutputKernelFn =
      std::function<void(const ContractionOutputMapper<T, Eigen::Index>&,
                         const Eigen::TensorContractionParams&, Eigen::Index,
                         Eigen::Index, Eigen::Index, Eigen::Index)>;
  for (auto _ : state) {
    OutputKernelFn kernel;
    switch (activation) {
      case FusedComputationType::kBiasAdd:
        kernel = WithBiasAdd<T>(bias_add_args);
        break;
      case FusedComputationType::kBiasAddWithRelu:
        kernel = WithBiasAddAndRelu<T>(bias_add_args);
        break;
      case FusedComputationType::kBiasAddWithRelu6:
        kernel = WithBiasAddAndRelu6<T>(bias_add_args);
        break;
      case FusedComputationType::kBiasAddWithElu:
        kernel = WithBiasAddAndElu<T>(bias_add_args);
        break;
      case FusedComputationType::kUndefined:
        LOG(FATAL) << "Activation type is undefined";
        break;
      default:
        LOG(FATAL) << "Activation type is not valid";
        break;
    }
    auto expr = lhs.contract(rhs, contract_pairs, kernel);

    using Dst = decltype(dst);
    using Expr = decltype(expr);
    ExecuteAssignOp</*vectorize=*/true, Device, Dst, Expr>::run(d, dst, expr);
  }

  state.SetItemsProcessed(state.iterations() * arg0 * arg1 * arg2);
}

}  // namespace tensorflow

// -------------------------------------------------------------------------- //
// Macros to dispatch to different MatMul shapes.
// -------------------------------------------------------------------------- //

#define INTS(...) __VA_ARGS__
#define BOOLS(...) __VA_ARGS__

#define BM_TFMlir(NAME, ARG0_SHAPE, ARG0_DYN_DIMS, ARG1_SHAPE, ARG1_DYN_DIMS,  \
                  ARG2_SHAPE, ARG2_DYN_DIMS, OUT_SHAPE, OUT_DYN_DIMS, ACT, FN, \
                  TYPE)                                                        \
  static void BM_mlir_##NAME##_##TYPE(::testing::benchmark::State& state) {    \
    RunFusedMatMulMlirBenchmark<TYPE>(                                         \
        state,                                                                 \
        GetFusedMatmulIR({ARG0_SHAPE}, {ARG0_DYN_DIMS}, {ARG1_SHAPE},          \
                         {ARG1_DYN_DIMS}, {ARG2_SHAPE}, {ARG2_DYN_DIMS},       \
                         {OUT_SHAPE}, {OUT_DYN_DIMS}, #TYPE, ACT),             \
        FN);                                                                   \
  }                                                                            \
  BENCHMARK(BM_mlir_##NAME##_##TYPE)

#define BM_TFMlir_DYNAMIC_ALL(M, N, K, ARG, ACT, FN, TYPE)                     \
  BM_TFMlir(FusedMatmulDynamicAll_##M##_##K##_##N##_##ARG##_##ACT, INTS(M, K), \
            BOOLS(kDynamicDim, kDynamicDim), INTS(K, N),                       \
            BOOLS(kDynamicDim, kDynamicDim), INTS(ARG), BOOLS(kDynamicDim),    \
            INTS(M, N), BOOLS(kDynamicDim, kDynamicDim), ACT, FN, TYPE)        \
      ->Args({M, K, N, ARG})

#define BM_TFMlir_STATIC_ALL(M, N, K, ARG, ACT, FN, TYPE)                     \
  BM_TFMlir(FusedMatmulStaticAll_##M##_##K##_##N##_##ARG##_##ACT, INTS(M, K), \
            BOOLS(kStaticDim, kStaticDim), INTS(K, N),                        \
            BOOLS(kStaticDim, kStaticDim), INTS(ARG), BOOLS(kStaticDim),      \
            INTS(M, N), BOOLS(kStaticDim, kStaticDim), ACT, FN, TYPE)         \
      ->Args({M, K, N, ARG})

#define BM_Eigen(NAME, TYPE)                                                 \
  static void BM_eigen_##NAME##_##TYPE(::testing::benchmark::State& state) { \
    RunFusedMatMulEigenBenchmark<TYPE>(state);                               \
  }                                                                          \
  BENCHMARK(BM_eigen_##NAME##_##TYPE)

#define BM_Eigen_WRAPPER(M, N, K, ARG, ACT, TYPE)             \
  BM_Eigen(FusedMatmul_##M##_##K##_##N##_##ARG##_##ACT, TYPE) \
      ->Args({M, K, N, ARG, ACT})

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_FUSED_MATMUL_OP_BENCHMARK_H_
