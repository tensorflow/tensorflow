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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_REDUCTION_BENCHMARK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_REDUCTION_BENCHMARK_H_

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"

namespace tensorflow {

// Use type aliases compatible with MLIR type names.
using f32 = float;

ABSL_CONST_INIT extern const bool kStatic;
ABSL_CONST_INIT extern const bool kDynamic;

// This header is a part of the library with private visibility and will be
// used only to build benchmarks for different functions in this folder, so
// it is ok to put convenience using-declarations here.
//
using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::llvm::StringRef;
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

struct MlirSpec {
  MlirSpec(StringRef op_name, StringRef element_type,
           SmallVector<bool, 2> input_dynamic,
           SmallVector<int32_t, 2> dims_to_reduce)
      : op_name(op_name),
        element_type(element_type),
        input_dynamic(std::move(input_dynamic)),
        dims_to_reduce(std::move(dims_to_reduce)) {}
  StringRef op_name;
  StringRef element_type;
  SmallVector<bool, 2> input_dynamic;
  SmallVector<int32, 2> dims_to_reduce;
};

std::string GetIR(StringRef op_name, ArrayRef<int64_t> input_shape,
                  ArrayRef<int64_t> output_shape,
                  ArrayRef<int32_t> dims_to_reduce, StringRef element_type);

template <typename T, int INPUT_RANK>
void RunReductionMlirBenchmark(::testing::benchmark::State& state,
                               size_t num_threads, const MlirSpec& spec) {
  // Input and output shapes to generate IR.
  SmallVector<int64_t, 2> mlir_input_shape, mlir_output_shape;

  // Compute input/output shapes and the number of elements.
  std::array<ssize_t, INPUT_RANK> input_shape;
  int64_t num_elements = 1;
  for (int i = 0; i < INPUT_RANK; ++i) {
    input_shape[i] = state.range(i);
    num_elements *= state.range(i);
    mlir_input_shape.push_back(spec.input_dynamic[i] ? kDynSize
                                                     : state.range(i));
    if (llvm::find(spec.dims_to_reduce, i) == spec.dims_to_reduce.end())
      mlir_output_shape.push_back(mlir_input_shape[i]);
  }

  std::unique_ptr<HostContext> host =
      num_threads > 0 ? CreateMultiThreadedHostContext(num_threads)
                      : CreateSingleThreadedHostContext();

  // Compile JIT executable.
  auto mlir_input = GetIR(spec.op_name, mlir_input_shape, mlir_output_shape,
                          spec.dims_to_reduce, spec.element_type);
  TfCpuRtPipelineOptions tf_cpurt_opts;
  tf_cpurt_opts.vectorize = true;
  JitExecutable& jit_executable =
      CreateJitExecutable(*host, mlir_input, "main",
                          /*lower_from_tensorflow=*/true, tf_cpurt_opts);

  // Build an ExecutionContext from the HostContext.
  llvm::Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(host.get(), /*resource_context=*/nullptr).build();
  tfrt::ExecutionContext exec_ctx(std::move(*req_ctx));

  // Generate random input data.
  Eigen::Tensor<T, INPUT_RANK, Eigen::RowMajor> input =
      GenRandomTensor<T, INPUT_RANK>(input_shape);

  std::array<MemrefDesc, 1> operands = {TensorToMemrefDesc(input)};

  auto result_values = std::array<RCReference<AsyncValue>, 2>{{}};
  RemainingResults results(result_values);

  // Free memory owned by the returned memrefs.
  ReturnValueConverter<ResultConversionCtx> converter(results);
  converter.AddConversion(FreeReturnedMemref);

  // Get an executable that might be specialized to the operands.
  AsyncValuePtr<Executable> executable =
      jit_executable.GetExecutable(operands, exec_ctx);

  // Wait for the compilation completion.
  host->Await({executable.CopyRef()});

  CHECK(!executable.IsError())
      << "Failed to get executable: " << StrCat(executable.GetError());
  CHECK(!executable->IsAsync()) << "async results are not supported";

  // Initialize call frame with MemrefDesc operands.
  Executable::CallFrame call_frame;
  if (auto err = executable->InitializeCallFrame(operands, &call_frame))
    LOG(FATAL) << "Failed to initialize call frame";

  for (auto s : state) {
    executable->Execute(call_frame, exec_ctx);
    if (auto err = executable->ReturnResults(converter, exec_ctx, &call_frame))
      LOG(FATAL) << "Failed to return compiled kernel results";
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_elements);
}

// -------------------------------------------------------------------------- //
// Run benchmark using Eigen expression evaluation.
// -------------------------------------------------------------------------- //

template <typename T, int RANK>
struct InitTensor {
  static Eigen::Tensor<T, RANK, Eigen::RowMajor> Get(
      const std::array<ssize_t, RANK>&);
};

#define INIT_TENSOR(RANK, UNROLL)                         \
  template <typename T>                                   \
  struct InitTensor<T, RANK> {                            \
    static Eigen::Tensor<T, RANK, Eigen::RowMajor> Get(   \
        const std::array<ssize_t, RANK>& shape) {         \
      Eigen::Tensor<T, RANK, Eigen::RowMajor> dst UNROLL; \
      return dst;                                         \
    }                                                     \
  };

template <typename T>
struct InitTensor<T, 0> {
  static Eigen::Tensor<T, 0, Eigen::RowMajor> Get(
      const std::array<ssize_t, 0>&) {
    return Eigen::Tensor<T, 0, Eigen::RowMajor>();
  }
};

INIT_TENSOR(1, (shape[0]));
INIT_TENSOR(2, (shape[0], shape[1]));
INIT_TENSOR(3, (shape[0], shape[1], shape[2]));

struct EigenSpec {
  explicit EigenSpec(SmallVector<int32_t, 2> dims_to_reduce)
      : dims_to_reduce(std::move(dims_to_reduce)) {}
  SmallVector<int32_t, 2> dims_to_reduce;
  size_t num_threads;
};

template <typename T, int INPUT_RANK, int OUTPUT_RANK>
void RunReductionEigenBenchmark(::testing::benchmark::State& state,
                                size_t num_threads, const EigenSpec& spec) {
  std::array<ssize_t, INPUT_RANK - OUTPUT_RANK> dims_to_reduce;
  for (int i = 0; i < dims_to_reduce.size(); ++i) {
    dims_to_reduce[i] = spec.dims_to_reduce[i];
  }

  // Compute input/output shapes and the number of elements.
  std::array<ssize_t, INPUT_RANK> input_shape;
  std::array<ssize_t, OUTPUT_RANK> output_shape;
  int64_t num_elements = 1;
  for (int i = 0, j = 0; i < INPUT_RANK; ++i) {
    input_shape[i] = state.range(i);
    num_elements *= state.range(i);
    if (llvm::find(spec.dims_to_reduce, i) == spec.dims_to_reduce.end())
      output_shape[j++] = input_shape[i];
  }

  Eigen::Tensor<T, INPUT_RANK, Eigen::RowMajor> lhs =
      GenRandomTensor<T, INPUT_RANK>(input_shape);

  Eigen::DefaultDevice singleThreadedDevice;
  Eigen::ThreadPool thread_pool(num_threads);
  llvm::Optional<Eigen::ThreadPoolDevice> multiThreadedDevice;
  if (num_threads > 0) multiThreadedDevice.emplace(&thread_pool, num_threads);

  auto dst = InitTensor<T, OUTPUT_RANK>::Get(output_shape);
  dst.setZero();

  for (auto s : state) {
    auto expr = lhs.sum(dims_to_reduce);

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

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          num_elements);
}

// -------------------------------------------------------------------------- //
// Macros to dispatch to different shapes.
// -------------------------------------------------------------------------- //

// MLIR benchmarks
#define BM_TFMlir(NAME, TYPE, NUM_THREADS, INPUT_RANK, SPEC)               \
  static void BM_mlir__##INPUT_RANK##D_##NAME##_##TYPE##_##NUM_THREADS(    \
      ::testing::benchmark::State& state) {                                \
    RunReductionMlirBenchmark<TYPE, INPUT_RANK>(state, NUM_THREADS, SPEC); \
  }                                                                        \
  BENCHMARK(BM_mlir__##INPUT_RANK##D_##NAME##_##TYPE##_##NUM_THREADS)      \
      ->MeasureProcessCPUTime()

#define ARGS_1D \
  Args({3})->Args({8})->Args({80})->Args({800})->Args({8000})->Args({8131})

#define BM_TFMlir1(NAME, TYPE, NUM_THREADS, SPEC) \
  BM_TFMlir(NAME, TYPE, NUM_THREADS, 1, SPEC)->ARGS_1D

#define ARGS_2D          \
  Args({2, 80})          \
      ->Args({8, 6})     \
      ->Args({80, 1})    \
      ->Args({80, 60})   \
      ->Args({81, 61})   \
      ->Args({800, 600}) \
      ->Args({802, 602})
#define BM_TFMlir2(NAME, TYPE, NUM_THREADS, SPEC) \
  BM_TFMlir(NAME, TYPE, NUM_THREADS, 2, SPEC)->ARGS_2D

// Eigen benchmarks
#define BM_Eigen(NAME, TYPE, NUM_THREADS, INPUT_RANK, OUTPUT_RANK, SPEC) \
  static void BM_eigen_##INPUT_RANK##D_##NAME##_##TYPE##_##NUM_THREADS(  \
      ::testing::benchmark::State& state) {                              \
    RunReductionEigenBenchmark<TYPE, INPUT_RANK, OUTPUT_RANK>(           \
        state, NUM_THREADS, SPEC);                                       \
  }                                                                      \
  BENCHMARK(BM_eigen_##INPUT_RANK##D_##NAME##_##TYPE##_##NUM_THREADS)    \
      ->MeasureProcessCPUTime()

#define BM_Eigen1(NAME, TYPE, NUM_THREADS) \
  BM_Eigen(NAME, TYPE, NUM_THREADS, 1, 0, EigenSpec({0}))->ARGS_1D

#define BM_Eigen2(NAME, TYPE, NUM_THREADS, OUTPUT_RANK, SPEC) \
  BM_Eigen(NAME, TYPE, NUM_THREADS, 2, OUTPUT_RANK, SPEC)->ARGS_2D

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_REDUCTION_BENCHMARK_H_
