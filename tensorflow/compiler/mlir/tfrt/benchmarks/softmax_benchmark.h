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

#ifndef TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_SOFTMAX_BENCHMARK_H_
#define TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_SOFTMAX_BENCHMARK_H_

#include <string>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

// Use type aliases compatible with MLIR type names.
using f32 = float;

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
  MlirSpec(StringRef element_type, SmallVector<bool, 2> input_dynamic)
      : element_type(element_type), input_dynamic(std::move(input_dynamic)) {}
  StringRef element_type;
  SmallVector<bool, 2> input_dynamic;
};

std::string GetSoftMaxIR(ArrayRef<int64_t> shape, StringRef element_type);

template <typename T, int INPUT_RANK>
void RunSoftmaxMlirBenchmark(::testing::benchmark::State& state,
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
  }

  std::unique_ptr<HostContext> host =
      num_threads > 0 ? CreateMultiThreadedHostContext(num_threads)
                      : CreateSingleThreadedHostContext();

  // Compile JIT executable.
  auto mlir_input = GetSoftMaxIR(input_shape, spec.element_type);
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
  llvm::Expected<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(operands, exec_ctx);
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

  for (auto s : state) {
    (*executable)->Execute(call_frame, exec_ctx);
    if (auto err =
            (*executable)->ReturnResults(converter, exec_ctx, &call_frame))
      LOG(FATAL) << "Failed to return compiled kernel results";
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 6 *
                          num_elements);
}

// -------------------------------------------------------------------------- //
// Run benchmark using Eigen expression evaluation.
// -------------------------------------------------------------------------- //

// Eigen code implementing SoftmaxFunctor::operator() carefully taken from
// tensorflow/core/kernels/softmax_op_functor.h
template <typename Device, typename T>
struct SoftmaxEigenImpl {
  static void Compute(const Device& d, T logits, T softmax) {
    const int kBatchDim = 0;
    const int kClassDim = 1;

    const int batch_size = logits.dimension(kBatchDim);
    const int num_classes = logits.dimension(kClassDim);

// These arrays are used to reduce along the class dimension, and broadcast
// the resulting value to all classes.
#if !defined(EIGEN_HAS_INDEX_LIST)
    Eigen::DSizes<int, 1> along_class(kClassDim);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);
#else
    Eigen::IndexList<Eigen::type2index<kClassDim>> along_class;
    Eigen::IndexList<int, Eigen::type2index<1>> batch_by_one;
    batch_by_one.set(0, batch_size);
    Eigen::IndexList<Eigen::type2index<1>, int> one_by_class;
    one_by_class.set(1, num_classes);
#endif
    // shifted_logits = logits - max(logits along classes);
    auto shifted_logits = (logits - logits.maximum(along_class)
                                        .eval()
                                        .reshape(batch_by_one)
                                        .broadcast(one_by_class));
    softmax.device(d) = shifted_logits.exp();
    softmax.device(d) = (softmax * softmax.sum(along_class)
                                       .inverse()
                                       .eval()
                                       .reshape(batch_by_one)
                                       .broadcast(one_by_class));
  }
};

// Functor used by SoftmaxOp to do the computations.
template <typename Device, typename T>
struct SoftmaxFunctor {
  // Computes Softmax or LogSoftmax activation.
  //
  // logits: dim: batch_size, num_classes.
  // softmax: dims: batch_size, num_classes.
  // log: boolean
  void operator()(const Device& d, T logits, T softmax) {
    SoftmaxEigenImpl<Device, T>::Compute(d, logits, softmax);
  }
};
template <typename T, int RANK>
void RunSoftmaxEigenBenchmark(::testing::benchmark::State& state,
                              size_t num_threads) {
  // Compute input/output shapes and the number of elements.
  std::array<ssize_t, RANK> input_shape;
  int64_t num_elements = 1;
  for (int i = 0; i < RANK; ++i) {
    input_shape[i] = state.range(i);
    num_elements *= state.range(i);
  }

  Eigen::Tensor<T, RANK, Eigen::RowMajor> input =
      GenRandomTensor<T, RANK>(input_shape);

  Eigen::DefaultDevice single_threaded_device;
  Eigen::ThreadPool thread_pool(num_threads);
  llvm::Optional<Eigen::ThreadPoolDevice> multi_threaded_device;
  if (num_threads > 0) multi_threaded_device.emplace(&thread_pool, num_threads);

  auto dst = InitEigenTensor<T, RANK>::Get(input_shape);
  dst.setZero();

  for (auto s : state) {
    using Dst = decltype(dst);

    SoftmaxFunctor<Eigen::DefaultDevice, Dst> functor;
    functor(single_threaded_device, input, dst);
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * 6 *
                          num_elements);
}

// -------------------------------------------------------------------------- //
// Macros to dispatch to different shapes.
// -------------------------------------------------------------------------- //

// MLIR benchmarks
#define BM_TFMlir(NAME, TYPE, NUM_THREADS, INPUT_RANK, SPEC)             \
  static void BM_mlir__##INPUT_RANK##D_##NAME##_##TYPE##_##NUM_THREADS(  \
      ::testing::benchmark::State& state) {                              \
    RunSoftmaxMlirBenchmark<TYPE, INPUT_RANK>(state, NUM_THREADS, SPEC); \
  }                                                                      \
  BENCHMARK(BM_mlir__##INPUT_RANK##D_##NAME##_##TYPE##_##NUM_THREADS)    \
      ->MeasureProcessCPUTime()

#define BM_TFMlir2(NAME, TYPE, NUM_THREADS, SPEC) \
  BM_TFMlir(NAME, TYPE, NUM_THREADS, 2, SPEC)
#define BM_TFMlir2_SingleThread(NAME, TYPE, SPEC) \
  BM_TFMlir(NAME, TYPE, 0, 2, SPEC)

// Eigen benchmarks
#define BM_Eigen(NAME, TYPE, NUM_THREADS, RANK)                   \
  static void BM_eigen_##RANK##D_##NAME##_##TYPE##_##NUM_THREADS( \
      ::testing::benchmark::State& state) {                       \
    RunSoftmaxEigenBenchmark<TYPE, RANK>(state, NUM_THREADS);     \
  }                                                               \
  BENCHMARK(BM_eigen_##RANK##D_##NAME##_##TYPE##_##NUM_THREADS)   \
      ->MeasureProcessCPUTime()

#define BM_Eigen2(NAME, TYPE, NUM_THREADS) BM_Eigen(NAME, TYPE, NUM_THREADS, 2)

#define BM_Eigen2_SingleThread(NAME, TYPE) BM_Eigen2(NAME, TYPE, 0)

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TFRT_BENCHMARKS_SOFTMAX_BENCHMARK_H_
