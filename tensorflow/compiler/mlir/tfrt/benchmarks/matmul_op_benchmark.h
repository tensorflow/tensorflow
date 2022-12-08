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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_MATMUL_OP_BENCHMARK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_MATMUL_OP_BENCHMARK_H_

#include <string>
#include <utility>

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/compiler/mlir/tfrt/utils/host_context.h"

namespace tensorflow {

// This header is a part of the library with private visibility and will be
// used only to build benchmarks for different functions in this folder, so
// it is ok to put convenience using-declarations here.

std::string GetMatmulIR(llvm::ArrayRef<int32_t> lhs_shape,
                        llvm::ArrayRef<bool> lhs_dynamic_dims,
                        llvm::ArrayRef<int32_t> rhs_shape,
                        llvm::ArrayRef<bool> rhs_dynamic_dims,
                        llvm::ArrayRef<int32_t> output_shape,
                        llvm::ArrayRef<bool> output_dynamic_dims);

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
void RunMatMulMlirBenchmark(::testing::benchmark::State& state,
                            // output_name is actually used on debug mode.
                            // NOLINTNEXTLINE
                            std::string output_name, llvm::StringRef mlir_input,
                            llvm::StringRef function_name) {
  // MatMul: [m, k] x [k, n]
  ssize_t m = state.range(0);
  ssize_t k = state.range(1);
  ssize_t n = state.range(2);

  std::unique_ptr<HostContext> host = CreateSingleThreadedHostContext();

  TfJitRtPipelineOptions tf_jitrt_opts;
  tf_jitrt_opts.vectorize = tensorflow::GetJitRtFlags().vectorize;
  tf_jitrt_opts.lower_to_mmt4d = tensorflow::GetJitRtFlags().pack_matmul;
  tf_jitrt_opts.matmul_tile_sizes = {state.range(3), state.range(4),
                                     state.range(5)};

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
  absl::StatusOr<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(operands);
  if (!executable.ok()) LOG(FATAL) << "Failed to specialize executable";

#if defined(DEBUG_XLA_RUNTIME_COMPILER)
  std::string dump_path = "/tmp/";
  std::unique_ptr<llvm::MemoryBuffer> obj = (*executable)->obj_file();
  CHECK(obj) << "Failed to get executable obj file";
  std::string object_filename = output_name;
  if (tf_jitrt_opts.lower_to_mmt4d) object_filename += "_packed";
  object_filename += ".o";
  std::error_code ec;
  llvm::raw_fd_ostream dump_stream(dump_path + object_filename, ec);
  CHECK(!ec) << "Failed to dump object file: " << ec.message();
  dump_stream.write(obj->getBufferStart(), obj->getBufferSize());
#endif

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

  CHECK(d.numThreads() == 1) << "Executing Eigen in multi-threaded";

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

#define INTS(...) __VA_ARGS__
#define BOOLS(...) __VA_ARGS__

#define BM_TFMlir(NAME, LHS_SHAPE, LHS_DYN_DIMS, RHS_SHAPE, RHS_DYN_DIMS,     \
                  OUT_SHAPE, OUT_DYN_DIMS, FN, TYPE)                          \
  static void BM_mlir_##NAME##_##TYPE(::testing::benchmark::State& state) {   \
    RunMatMulMlirBenchmark<TYPE>(                                             \
        state, #NAME,                                                         \
        GetMatmulIR({LHS_SHAPE}, {LHS_DYN_DIMS}, {RHS_SHAPE}, {RHS_DYN_DIMS}, \
                    {OUT_SHAPE}, {OUT_DYN_DIMS}, #TYPE),                      \
        FN);                                                                  \
  }                                                                           \
  BENCHMARK(BM_mlir_##NAME##_##TYPE)

#define BM_TFMlir_DYNAMIC_ALL(M, N, K, T_M, T_N, T_K, FN, TYPE)       \
  BM_TFMlir(MatmulDynamicAll_##M##_##K##_##N##_##T_M##_##T_N##_##T_K, \
            INTS(M, K), BOOLS(kDynamicDim, kDynamicDim), INTS(K, N),  \
            BOOLS(kDynamicDim, kDynamicDim), INTS(M, N),              \
            BOOLS(kDynamicDim, kDynamicDim), FN, TYPE)                \
      ->Args({M, K, N, T_M, T_N, T_K})

#define BM_TFMlir_STATIC_ALL(M, N, K, T_M, T_N, T_K, FN, TYPE)       \
  BM_TFMlir(MatmulStaticAll_##M##_##K##_##N##_##T_M##_##T_N##_##T_K, \
            INTS(M, K), BOOLS(kStaticDim, kStaticDim), INTS(K, N),   \
            BOOLS(kStaticDim, kStaticDim), INTS(M, N),               \
            BOOLS(kStaticDim, kStaticDim), FN, TYPE)                 \
      ->Args({M, K, N, T_M, T_N, T_K})

#define BM_Eigen(NAME, TYPE)                                                 \
  static void BM_eigen_##NAME##_##TYPE(::testing::benchmark::State& state) { \
    RunMatMulEigenBenchmark<TYPE>(state);                                    \
  }                                                                          \
  BENCHMARK(BM_eigen_##NAME##_##TYPE)

#define BM_Eigen_WRAPPER(M, N, K, TYPE) \
  BM_Eigen(Matmul_##M##_##K##_##N, TYPE)->Args({M, K, N})

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_MATMUL_OP_BENCHMARK_H_
