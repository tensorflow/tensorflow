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

#include <functional>
#include <memory>
#include <utility>

#include "llvm/Support/SourceMgr.h"
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_executor.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfrt/utils/host_context.h"
#include "tensorflow/core/framework/tensor.h"
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {

using ::tfrt::ArrayRef;
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
      case DT_INT64:
        input_tensors.back().flat<int64_t>().setRandom();
        break;
      default:
        CHECK(false) << "Unsupported dtype: " << spec.dtype;
    }
  }

  return input_tensors;
}

// -------------------------------------------------------------------------- //
// Run function benchmark via the TF JitRt compilation.
// -------------------------------------------------------------------------- //

void RunJitRtBenchmark(::testing::benchmark::State& state,
                       llvm::StringRef mlir_input,
                       llvm::StringRef function_name,
                       llvm::ArrayRef<InputTensorSpec> input_specs,
                       bool vectorize, bool codegen_transpose) {
  // Number of worker threads.
  int64_t num_threads = state.range(0);

  // Host context for running compute tasks.
  std::unique_ptr<HostContext> host =
      num_threads > 0 ? CreateMultiThreadedHostContext(num_threads)
                      : CreateSingleThreadedHostContext();

  TfJitRtPipelineOptions tf_jitrt_opts;
  tf_jitrt_opts.vectorize = vectorize;
  JitExecutable& jit_executable =
      CreateJitExecutable(*host, mlir_input, function_name,
                          /*lower_from_tensorflow=*/true, tf_jitrt_opts);

  // Build an ExecutionContext from the HostContext.
  llvm::Expected<RCReference<RequestContext>> req_ctx =
      RequestContextBuilder(host.get(), /*resource_context=*/nullptr).build();
  tfrt::ExecutionContext exec_ctx(std::move(*req_ctx));

  // Generate random inputs based on the tensor specs.
  llvm::SmallVector<Tensor> input_tensors = GetInputTensors(input_specs);

  // Record data ptrs of inputs.
  llvm::SmallVector<void*> input_ptrs;
  // Convert input tensors to memref descriptors.
  llvm::SmallVector<MemrefDesc> operands;
  for (const Tensor& tensor : input_tensors) {
    input_ptrs.push_back(tensor.data());
    operands.emplace_back(TensorToMemrefDesc(tensor));
  }

  // Get an executable that might be specialized to the operands.
  absl::StatusOr<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(operands);
  if (!executable.ok())
    LOG(FATAL) << "Failed to specialize executable: "
               << executable.status().message();

  // Wait for the compilation completion.
  host->Await({executable->CopyRef()});

  CHECK(!executable->IsError())
      << "Failed to get executable: " << executable->GetError().message();
  CHECK(!(*executable)->IsAsync()) << "async results are not supported";

  // Placeholders for returned values.
  unsigned num_results = (*executable)->num_results();
  llvm::SmallVector<RCReference<AsyncValue>> result_values(num_results);
  RemainingResults results(result_values);

  // Free memory owned by the returned memrefs.
  ResultConversionCtx result_ctx(std::move(input_ptrs));
  RemainingResultsConverter<ResultConversionCtx> converter(results, result_ctx);
  converter.AddConversion(FreeReturnedMemref);

  // Initialize call frame with MemrefDesc operands.
  Executable::CallFrame call_frame;
  if (auto st = (*executable)->InitializeCallFrame(operands, &call_frame);
      !st.ok())
    LOG(FATAL) << "Failed to initialize call frame";

  // Execute async tasks in the HostContext work queue.
  Executable::ExecuteOpts opts;
  HostContextAsyncTaskRunner async_task_runner(host.get());
  opts.async_task_runner = &async_task_runner;

  // Execute compiled kernel and return results.
  auto execute = [&]() {
    call_frame.args[0] = nullptr;  // reset kernel context argument
    (*executable)->Execute(call_frame, opts);
    if (auto st = (*executable)->ReturnResults(converter, &call_frame);
        !st.ok())
      LOG(FATAL) << "Failed to return compiled kernel results";
  };

  // Warm up to compile the kernel outside of the benchmark loop.
  execute();

  for (auto _ : state) {
    execute();
  }
}

// -------------------------------------------------------------------------- //
// Run function benchmark via the TF->TFRT fallback lowering.
// -------------------------------------------------------------------------- //

void RunTfrtBenchmark(::testing::benchmark::State& state,
                      llvm::StringRef mlir_input, llvm::StringRef function_name,
                      ArrayRef<InputTensorSpec> input_specs) {
  // Number of worker threads (intra-op concurrency for the fallback ops).
  int64_t num_threads = state.range(0);
  RuntimeFallbackExecutor executor(num_threads);

  executor.Prepare(mlir_input);

  // Generate random inputs based on the tensor specs.
  llvm::SmallVector<Tensor> input_tensors = GetInputTensors(input_specs);

  for (auto _ : state) {
    executor.Execute(function_name, input_tensors);
  }
}

// -------------------------------------------------------------------------- //
// Run arbitrary benchark written as a function.
// -------------------------------------------------------------------------- //

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
