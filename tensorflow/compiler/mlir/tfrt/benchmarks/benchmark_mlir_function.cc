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

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>

#include "llvm/Support/SourceMgr.h"
#include "mlir/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/runtime_fallback/runtime/kernel_utils.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tfrt/bef_converter/mlir_to_bef.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {

using ::tfrt::ArrayRef;
using ::tfrt::AsyncValue;
using ::tfrt::AsyncValuePtr;
using ::tfrt::BEFFile;
using ::tfrt::ExecutionContext;
using ::tfrt::Function;
using ::tfrt::HostContext;
using ::tfrt::MakeAvailableAsyncValueRef;
using ::tfrt::RCReference;
using ::tfrt::RemainingResults;
using ::tfrt::RequestContext;
using ::tfrt::RequestContextBuilder;
using ::tfrt::ResourceContext;

using ::tfrt::cpu::jit::Executable;
using ::tfrt::cpu::jit::JitExecutable;
using ::tfrt::cpu::jit::MemrefDesc;
using ::tfrt::cpu::jit::ReturnValueConverter;

using ::tensorflow::Env;
using ::tensorflow::thread::ThreadPool;
using ::tensorflow::thread::ThreadPoolInterface;

using ::tensorflow::tfrt_stub::FallbackTensor;

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
// Run function benchmark via the TF CPURT compilation.
// -------------------------------------------------------------------------- //

void RunCpurtBenchmark(::testing::benchmark::State& state,
                       llvm::StringRef mlir_input,
                       llvm::StringRef function_name,
                       llvm::ArrayRef<InputTensorSpec> input_specs,
                       bool vectorize) {
  // Number of worker threads.
  int64_t num_threads = state.range(0);

  // Host context for running compute tasks.
  std::unique_ptr<HostContext> host =
      num_threads > 0 ? CreateMultiThreadedHostContext(num_threads)
                      : CreateSingleThreadedHostContext();

  TfCpuRtPipelineOptions tf_cpurt_opts;
  tf_cpurt_opts.vectorize = vectorize;
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
  llvm::Expected<AsyncValuePtr<Executable>> executable =
      jit_executable.GetExecutable(operands, exec_ctx);
  if (auto err = executable.takeError())
    LOG(FATAL) << "Failed to specialize executable: " << tfrt::StrCat(err);

  // Wait for the compilation completion.
  host->Await({executable->CopyRef()});

  CHECK(!executable->IsError())
      << "Failed to get executable: " << tfrt::StrCat(executable->GetError());
  CHECK(!(*executable)->IsAsync()) << "async results are not supported";

  // Placeholders for returned values.
  unsigned num_results = (*executable)->num_results();
  llvm::SmallVector<RCReference<AsyncValue>> result_values(num_results);
  RemainingResults results(result_values);

  // Free memory owned by the returned memrefs.
  ReturnValueConverter<ResultConversionCtx> converter(results);
  converter.AddConversion(FreeReturnedMemref);

  // Initialize call frame with MemrefDesc operands.
  Executable::CallFrame call_frame;
  if (auto err = (*executable)->InitializeCallFrame(operands, &call_frame))
    LOG(FATAL) << "Failed to initialize call frame";

  for (auto _ : state) {
    call_frame.args[0] = nullptr;  // reset kernel context argument
    (*executable)->Execute(call_frame, exec_ctx);
    if (auto err =
            (*executable)->ReturnResults(converter, exec_ctx, &call_frame))
      LOG(FATAL) << "Failed to return compiled kernel results";
  }
}

// -------------------------------------------------------------------------- //
// Run function benchmark via the TF->TFRT fallback lowering.
// -------------------------------------------------------------------------- //

// Thread pool for running `intra-op` tasks scheduled by the fallback kernels.
class IntraOpTheadPool : public ThreadPoolInterface {
 public:
  explicit IntraOpTheadPool(int num_threads)
      : tpool_(Env::Default(), "intra-op", std::max(1, num_threads)) {}

  void Schedule(std::function<void()> fn) override {
    tpool_.Schedule(std::move(fn));
  }

  int NumThreads() const override { return tpool_.NumThreads(); }
  int CurrentThreadId() const override { return tpool_.CurrentThreadId(); }
  void Cancel() override {}

 private:
  ThreadPool tpool_;
};

// Run TFRT fallback initialization function to instantiate all fallback
// kernels ahead of executing the compute function.
static void RunTfrtInitializer(const ExecutionContext& exec_ctx,
                               BEFFile* bef_file,
                               llvm::StringRef fallback_init_func) {
  const Function* func = bef_file->GetFunction(fallback_init_func);
  CHECK(func) << "TFRT initialization function was not found";
  CHECK_EQ(func->argument_types().size(), 1);

  llvm::SmallVector<RCReference<AsyncValue>, 1> results;
  results.resize(func->result_types().size());
  CHECK_EQ(results.size(), 1);

  func->Execute(exec_ctx, tfrt::GetReadyChain().GetAsyncValue(), results);

  HostContext* host = exec_ctx.host();
  host->Await(results);

  CHECK(!results[0]->IsError()) << "Failed to run TFRT initialization function";
}

void RunTfrtBenchmark(::testing::benchmark::State& state,
                      llvm::StringRef mlir_input, llvm::StringRef function_name,
                      ArrayRef<InputTensorSpec> input_specs) {
  // Number of worker threads (intra-op concurrency for the fallback ops).
  int64_t num_threads = state.range(0);

  // We only support benchmarks written in the Tensorflow dialect.
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);

  llvm::SourceMgr source_mgr;
  source_mgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(mlir_input, "benchmark"), llvm::SMLoc());

  // Parse a kernel source code into the MLIR Module.
  mlir::OwningModuleRef module(mlir::parseSourceFile(source_mgr, &context));
  CHECK(module) << "failed to parse mlir module";

  // Collect all diagnostics emitted while lowering parsed kernel module.
  std::string diagnostic_str;
  llvm::raw_string_ostream os(diagnostic_str);
  mlir::SourceMgrDiagnosticHandler handler(source_mgr, module->getContext(),
                                           os);

  // Convert TF to TFRT fallback dialect.
  TfrtPipelineOptions core_rt_opts;
  core_rt_opts.hoist_invariant_ops = true;
  core_rt_opts.enable_native_ops = false;
  core_rt_opts.cost_threshold = 1024;
  core_rt_opts.upper_cost_threshold = 100000;
  core_rt_opts.merge_inter_dependent_streams = true;
  core_rt_opts.func_use_fallback_tensor = true;

  mlir::PassManager pm(module->getContext());
  pm.addPass(CreateTfToTfrtConversionPass(core_rt_opts));

  CHECK(mlir::succeeded(pm.run(*module)))
      << "Failed to lower module to TFRT: " << os.str();

  // Create a thread pool for running intra-op tasks.
  IntraOpTheadPool intra_op(num_threads);

  // Create a HostContext for running TFRT functions. Concurrent work queue acts
  // similar to the Tensorflow `inter-op` thread pool, so we'll match the size.
  auto host = num_threads ? CreateMultiThreadedHostContext(num_threads)
                          : CreateSingleThreadedHostContext();
  tfrt::RegisterStaticKernels(host->GetMutableRegistry());

  // Convert module to BEF.
  auto bef_buffer =
      tfrt::ConvertMLIRToBEF(*module, /*disable_optional_sections=*/false);
  CHECK(!bef_buffer.empty()) << "Failed to convert module to BEF";

  // Build an ExecutionContext from the HostContext.
  ResourceContext resource_context;
  auto builder = RequestContextBuilder(host.get(), &resource_context);

  // Get tensorflow::EagerContext for the kernel fallback.
  auto* eager_context_resource =
      resource_context
          .GetOrCreateResource<tensorflow::tfd::EagerContextResource>(
              tensorflow::tfd::kEagerContextResourceName);
  auto expected_eager_context = eager_context_resource->GetTFEagerContext();
  auto* eager_context = expected_eager_context.get();

  // Initialize fallback kernels state with a custom intra-op thread pool.
  auto status = tensorflow::tfd::SetUpKernelFallbackCompatRequestContext(
      &builder, /*runner_table=*/nullptr, eager_context, &intra_op);
  CHECK(status.ok()) << "Failed to setup request context: "
                     << status.error_message();

  auto req_ctx = std::move(builder).build();
  if (auto err = req_ctx.takeError())
    LOG(FATAL) << "Failed to build a request context";

  ExecutionContext exec_ctx(std::move(*req_ctx));

  auto bef_file = BEFFile::Open(bef_buffer, host->GetKernelRegistry(),
                                host->diag_handler(), host->allocator());
  CHECK(bef_file) << "Failed to open BEF";

  // Run TFRT initialization function to pre-instantiate fallback kernels.
  RunTfrtInitializer(exec_ctx, bef_file.get(), "_tfrt_fallback_init");

  // Get the kernel entrypoint function.
  const Function* compute = bef_file->GetFunction(function_name);
  CHECK(compute) << "Entrypoint function not found";

  // Generate random inputs based on the tensor specs.
  llvm::SmallVector<Tensor> input_tensors = GetInputTensors(input_specs);

  // Prepare function arguments from ready Chain and input Tensors.
  llvm::SmallVector<AsyncValue*> arguments;
  arguments.push_back(tfrt::GetReadyChain().release());
  for (const Tensor& input_tensor : input_tensors) {
    auto av = MakeAvailableAsyncValueRef<FallbackTensor>(input_tensor);
    arguments.push_back(av.release());
  }

  // Space for returned values.
  llvm::SmallVector<RCReference<AsyncValue>> results;

  for (auto _ : state) {
    // Reset results in preparation for the function call.
    results.clear();
    results.resize(compute->result_types().size());

    compute->Execute(exec_ctx, arguments, results);

    // Wait for the function execution to finish, as well as the side-effects.
    host->Await(results);

    // First result is always a chain, check if it has error.
    if (auto* error = results[0]->GetErrorIfPresent())
      LOG(FATAL) << "Failed to execute a function";
  }

  // Deallocate arguments.
  for (auto* argument : arguments) argument->DropRef();
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
