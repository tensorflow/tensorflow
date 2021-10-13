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

#include "tensorflow/compiler/mlir/tfrt/benchmarks/benchmark_mlir_function.h"

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

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

// Returns random tensors generated based on the input specs.
static llvm::SmallVector<Tensor> GetInputTensors(
    llvm::ArrayRef<InputTensorSpec> input_specs) {
  llvm::SmallVector<Tensor> input_tensors;

  for (const InputTensorSpec& spec : input_specs) {
    TensorShape shape;
    CHECK(TensorShapeUtils::MakeShape(spec.dims, &shape).ok());
    input_tensors.emplace_back(spec.dtype, shape);

    // Initialize tensors with random data.
    switch (spec.dtype) {
      case DT_FLOAT:
        input_tensors.back().flat<float>().setRandom();
        break;
      default:
        CHECK(false) << "Unsupported dtype: " << spec.dtype;
    }
  }

  return input_tensors;
}

void RunMlirBenchmark(::testing::benchmark::State& state,
                      llvm::StringRef mlir_input, llvm::StringRef function_name,
                      llvm::ArrayRef<InputTensorSpec> input_specs) {
  // Number of worker threads.
  int64_t num_threads = state.range(0);

  // Host context for running compute tasks.
  std::unique_ptr<HostContext> host =
      num_threads > 0 ? CreateMultiThreadedHostContext(num_threads)
                      : CreateSingleThreadedHostContext();

  TfCpuRtPipelineOptions tf_cpurt_opts;
  JitExecutable& jit_executable =
      CreateJitExecutable(*host, mlir_input, function_name,
                          /*lower_from_tensorflow=*/true, tf_cpurt_opts);

  // Build an ExecutionContext from the HostContext.
  llvm::Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(host.get(), /*resource_context=*/nullptr).build();
  tfrt::ExecutionContext exec_ctx(std::move(*req_ctx));

  // Generate random inputs based on the tensor specs.
  llvm::SmallVector<Tensor> input_tensors = GetInputTensors(input_specs);

  // Convert input tensors to memref descriptors.
  llvm::SmallVector<MemrefDesc> operands;
  for (const Tensor& tensor : input_tensors)
    operands.emplace_back(TensorToMemrefDesc(tensor));

  // Get an executable that might be specialized to the operands.
  AsyncValuePtr<Executable> executable =
      jit_executable.GetExecutable(operands, exec_ctx);

  // Wait for the compilation completion.
  host->Await({executable.CopyRef()});

  CHECK(!executable.IsError())
      << "Failed to get executable: " << StrCat(executable.GetError());
  CHECK(!executable->IsAsync()) << "async results are not supported";

  // Placeholders for returned values.
  llvm::SmallVector<RCReference<AsyncValue>> result_values;
  for (int i = 0; i < executable->signature().num_results(); ++i)
    result_values.emplace_back();
  RemainingResults results(result_values);

  // Free memory owned by the returned memrefs.
  ReturnValueConverter<ResultConversionCtx> converter(results);
  converter.AddConversion(FreeReturnedMemref);

  // Initialize call frame with MemrefDesc operands.
  Executable::CallFrame call_frame;
  if (auto err = executable->InitializeCallFrame(operands, &call_frame))
    LOG(FATAL) << "Failed to initialize call frame";

  for (auto _ : state) {
    executable->Execute(call_frame, exec_ctx);
    if (auto err = executable->ReturnResults(converter, exec_ctx, &call_frame))
      LOG(FATAL) << "Failed to return compiled kernel results";
  }
}

// Benchmark arbitrary compute function written as Eigen expression(s).
void RunEigenBenchmark(
    ::testing::benchmark::State& state,
    std::function<void(llvm::ArrayRef<Tensor>,
                       llvm::Optional<Eigen::ThreadPoolDevice>)>
        compute,
    llvm::ArrayRef<InputTensorSpec> input_specs) {
  // Number of worker threads.
  int64_t num_threads = state.range(0);

  // Maybe construct an Eigen thread pool device for evaluating expressions.
  Eigen::ThreadPool thread_pool(num_threads);
  llvm::Optional<Eigen::ThreadPoolDevice> device;
  if (num_threads > 0) device.emplace(&thread_pool, num_threads);

  // Generate random inputs based on the tensor specs.
  llvm::SmallVector<Tensor> input_tensors = GetInputTensors(input_specs);

  // Call the user defined compute function.
  for (auto _ : state) {
    compute(input_tensors, device);
  }
}

}  // namespace tensorflow
