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
#include "tensorflow/core/tfrt/mlrt/kernel/batch_kernel.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"
#include "absl/base/optimization.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/batching_util/batch_resource_base.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/runtime_fallback/runtime/fallback_batch_kernel.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner_cache.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/span.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/execute.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/register_span.h"
#include "tensorflow/core/tfrt/mlrt/kernel/context.h"
#include "tensorflow/core/tfrt/mlrt/kernel/kernel_runner_utils.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tfrt/concurrency/chain.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow {
namespace tf_mlrt {
namespace {

constexpr char kMlrtBatchFunctionName[] = "MlrtBatchFunction";
constexpr char kOpKernelRunnerCacheResourceName[] = "MlrtOpKernelCache";

// A thread local variable for passing the mlrt::ExecutionContext in the same
// thread.
//
// TODO(chky): This is a workaround, though it is better than using an
// additional tensor argument. A better implementation would be to remove the
// tensorflow::OpKernel interface used here.
mlrt::ExecutionContext*& GetBatchFunctionMlrtContext() {
  thread_local mlrt::ExecutionContext* context = nullptr;
  return context;
}

// An RAII object for saving and restoring the thread local
// mlrt::ExecutionContext.
class ScopedBatchFunctionMlrtContext {
 public:
  explicit ScopedBatchFunctionMlrtContext(
      mlrt::ExecutionContext* current_context) {
    last_context_ = GetBatchFunctionMlrtContext();
    GetBatchFunctionMlrtContext() = current_context;
  }

  ScopedBatchFunctionMlrtContext(const ScopedBatchFunctionMlrtContext&) =
      delete;
  ScopedBatchFunctionMlrtContext& operator=(
      const ScopedBatchFunctionMlrtContext&) = delete;

  ~ScopedBatchFunctionMlrtContext() {
    GetBatchFunctionMlrtContext() = last_context_;
  }

 private:
  mlrt::ExecutionContext* last_context_ = nullptr;
};

template <typename Frame>
void BatchFunctionInvokeHelper(Frame& frame) {
  ScopedBatchFunctionMlrtContext scoped_context(&frame.execution_context());

  const auto& fallback_request_state = frame.context().fallback_request_state();

  auto* runner_cache =
      frame.context()
          .resource_context()
          .template GetOrCreateResource<tfrt_stub::OpKernelRunnerCache>(
              kOpKernelRunnerCacheResourceName);

  auto attr_builder = [node_def_text = frame.node_def_text(), f = frame.f()](
                          tensorflow::AttrValueMap* attr_value_map) {
    tensorflow::NodeDef node_def;
    // TODO(182876485): Remove the conditional selection after protobuf version
    // is bumped up.
    if (!google::protobuf::TextFormat::ParseFromString(
#if defined(PLATFORM_GOOGLE)
            node_def_text,
#else
            std::string(node_def_text),
#endif
            &node_def)) {
      return absl::InternalError(
          absl::StrCat("CreateOp: failed to parse NodeDef: ", node_def_text));
    }

    *attr_value_map = node_def.attr();

    auto ptr_value = absl::bit_cast<int64_t>(f);
    (*attr_value_map)["opaque_function_handle"].set_i(ptr_value);

    return absl::OkStatus();
  };

  tfrt::Location loc;
  loc.data = absl::bit_cast<intptr_t>(frame.f());

  auto kernel_runner = runner_cache->GetOrCreate(
      loc, kMlrtBatchFunctionName, frame.device_name(), frame.args().size(),
      attr_builder, fallback_request_state.device_manager(),
      fallback_request_state.process_function_library_runtime());

  if (ABSL_PREDICT_FALSE(!kernel_runner.ok())) {
    frame.execution_context().Fail(std::move(kernel_runner).status());
    return;
  }

  DCHECK((*kernel_runner)->IsAsync());
  ExecuteKernelRunner</*IsAsync=*/true>(
      frame, frame.context(), fallback_request_state, **kernel_runner);
}

// A customized BatchResource whose batch function is a mlrt::bc::Function.
class MlrtBatchResource : public tensorflow::serving::BatchResourceBase {
  struct MlrtBatchTask : BatchTask {
    explicit MlrtBatchTask(mlrt::ExecutionContext* caller_context)
        : caller_context(caller_context) {
      DCHECK(caller_context);
    }
    mlrt::ExecutionContext* caller_context = nullptr;

   private:
    std::unique_ptr<BatchTask> CreateDerivedTask() override {
      return std::make_unique<MlrtBatchTask>(this->caller_context);
    }
  };

 public:
  using BatchFunctionType = mlrt::bc::Function;

  static mlrt::bc::Function CastHandleToFunction(int64_t handle) {
    return absl::bit_cast<mlrt::bc::Function>(handle);
  }

  // This can only be called in Compute() and ComputeAsync() because thread
  // local is used to pass the context.
  static absl::StatusOr<std::unique_ptr<BatchTask>> CreateBatchTask(
      OpKernelContext*) {
    return {std::make_unique<MlrtBatchTask>(GetBatchFunctionMlrtContext())};
  }

  // This can only be called in Compute() and ComputeAsync() because thread
  // local is used to pass the context.
  static absl::StatusOr<tfrt::ResourceContext*> GetClientGraphResourceContext(
      OpKernelContext*) {
    const auto& context =
        GetBatchFunctionMlrtContext()->GetUserContext<Context>();
    const auto& fallback_request_state = context.fallback_request_state();
    // If `client_graph_resource_context` is null, it implies that it's safe to
    // fall back to the per-model resource context.
    return fallback_request_state.client_graph_resource_context() != nullptr
               ? fallback_request_state.client_graph_resource_context()
               : &context.resource_context();
  }

  static absl::string_view GetBatchFunctionName(
      const BatchFunctionType& batch_function) {
    return batch_function.name();
  }

  static absl::Status Create(OpKernelContext* c,
                             const serving::BatchResourceOptions& options,
                             mlrt::bc::Function function,
                             bool enable_large_batch_splitting,
                             bool disable_padding,
                             std::unique_ptr<MlrtBatchResource>* resource) {
    BatcherT::Options batcher_options;
    batcher_options.num_batch_threads = options.num_batch_threads;
    if (options.mixed_priority_batching_policy ==
        serving::MixedPriorityBatchingPolicy::kPriorityMerge) {
      batcher_options.use_global_scheduler = true;
      batcher_options.rank_queues = true;
    }
    std::shared_ptr<BatcherT> batcher;
    TF_RETURN_IF_ERROR(BatcherT::Create(batcher_options, &batcher));

    resource->reset(new MlrtBatchResource(
        function, std::move(batcher),
        GetBatcherQueueOptions(
            options.num_batch_threads, options.max_batch_size,
            options.batch_timeout_micros, options.max_enqueued_batches,
            options.allowed_batch_sizes, enable_large_batch_splitting,
            disable_padding,
            /* batch_padding_policy= */ options.batch_padding_policy,
            options.low_priority_max_batch_size,
            options.low_priority_batch_timeout_micros,
            options.low_priority_max_enqueued_batches,
            options.low_priority_allowed_batch_sizes,
            options.mixed_priority_batching_policy),
        options.allowed_batch_sizes));
    return absl::OkStatus();
  }

  static absl::Status Create(
      OpKernelContext* c,
      AdaptiveBatcherT::Options adaptive_shared_batch_scheduler_options,
      int32_t max_batch_size, int32_t batch_timeout_micros,
      int32_t max_enqueued_batches,
      const std::vector<int32_t>& allowed_batch_sizes,
      mlrt::bc::Function function, bool disable_padding,
      std::unique_ptr<MlrtBatchResource>* resource) {
    std::shared_ptr<AdaptiveBatcherT> batcher;
    TF_RETURN_IF_ERROR(AdaptiveBatcherT::Create(
        adaptive_shared_batch_scheduler_options, &batcher));

    resource->reset(new MlrtBatchResource(
        function, std::move(batcher),
        GetAdaptiveBatcherQueueOptions(max_batch_size, batch_timeout_micros,
                                       max_enqueued_batches,
                                       true /* enable large batch split */,
                                       allowed_batch_sizes, disable_padding),
        allowed_batch_sizes));
    return absl::OkStatus();
  }

  string DebugString() const final { return "MlrtBatchResource"; }

  mlrt::bc::Function batch_function() const { return batch_function_; }

 private:
  MlrtBatchResource(mlrt::bc::Function batch_function,
                    std::shared_ptr<BatcherT> batcher,
                    const BatcherT::QueueOptions& batcher_queue_options,
                    const std::vector<int32_t>& allowed_batch_sizes)
      : BatchResourceBase(
            /*has_process_batch_function=*/true, std::move(batcher),
            batcher_queue_options, allowed_batch_sizes),
        batch_function_(batch_function) {}

  MlrtBatchResource(mlrt::bc::Function batch_function,
                    std::shared_ptr<AdaptiveBatcherT> batcher,
                    const AdaptiveBatcherT::QueueOptions& batcher_queue_options,
                    const std::vector<int32_t>& allowed_batch_sizes)
      : BatchResourceBase(
            /*has_process_batch_function=*/true, std::move(batcher),
            batcher_queue_options, allowed_batch_sizes),
        batch_function_(batch_function) {}

  void ProcessFuncBatchImpl(
      const BatchTask& last_task, absl::Span<const Tensor> inputs,
      std::vector<Tensor>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override;

  mlrt::bc::Function batch_function_;
};

void MlrtBatchResource::ProcessFuncBatchImpl(
    const BatchTask& last_task, absl::Span<const Tensor> inputs,
    std::vector<Tensor>* combined_outputs,
    std::function<void(const absl::Status&)> done) const {
  std::vector<mlrt::Value> arguments;
  arguments.reserve(inputs.size());
  for (const auto& input : inputs) {
    arguments.emplace_back(tfrt_stub::FallbackTensor(input));
  }

  std::vector<mlrt::Value> results(batch_function_.output_regs().size());

  const auto& task = down_cast<const MlrtBatchTask&>(last_task);
  DCHECK(task.context);
  mlrt::ExecutionContext& caller_context = *task.caller_context;

  auto& caller_tf_context = caller_context.GetUserContext<tf_mlrt::Context>();
  const auto& caller_fallback_request_state =
      caller_tf_context.fallback_request_state();

  // Connect to the batch step id propagated from batch task.
  int64_t step_id = caller_fallback_request_state.step_id();

  // Copy per-request states to create a new KernelFallbackCompatRequestState.
  //
  // TODO(chky): Consider adding copy ctor for KernelFallbackCompatRequestState.
  tfd::KernelFallbackCompatRequestState fallback_request_state(
      caller_fallback_request_state.runner(),
      &caller_fallback_request_state.device_manager(), step_id,
      caller_fallback_request_state.runner_table(),
      caller_fallback_request_state.resource_array(),
      caller_fallback_request_state.intra_op_threadpool(),
      caller_fallback_request_state.session_metadata(),
      &caller_fallback_request_state.process_function_library_runtime());

  fallback_request_state.set_cost_recorder(
      caller_fallback_request_state.cost_recorder());

  fallback_request_state.set_client_graph_resource_context(
      caller_fallback_request_state.client_graph_resource_context());

  fallback_request_state.set_cancellation_manager(
      caller_fallback_request_state.cancellation_manager());
  fallback_request_state.set_runtime_config(
      caller_fallback_request_state.runtime_config());

  tsl::profiler::TraceMeProducer activity(
      // To TraceMeConsumers in WorkQueue.
      [step_id] {
        return tsl::profiler::TraceMeEncode("RunMlrtFunction",
                                            {{"id", step_id}, {"_r", 1}});
      },
      tsl::profiler::ContextType::kTfrtExecutor, step_id,
      tsl::profiler::TraceMeLevel::kInfo);
  auto trace_me_context_id = activity.GetContextId();

  // Copy the ExecutionContext and its user contexts for async execution.
  auto user_contexts = caller_context.CopyUserContexts();
  mlrt::ExecutionContext execution_context(&caller_context.loaded_executable(),
                                           std::move(user_contexts),
                                           caller_context.user_error_loggers());
  execution_context.GetUserContext<tf_mlrt::Context>()
      .set_fallback_request_state(&fallback_request_state);

  auto* work_queue = caller_context.work_queue();
  DCHECK(work_queue);
  execution_context.set_work_queue(work_queue);

  auto chain = tsl::MakeConstructedAsyncValueRef<tsl::Chain>();

  execution_context.set_exit_handler(
      [chain]() mutable { chain.SetStateConcrete(); });

  execution_context.CallByMove(batch_function_, absl::MakeSpan(arguments),
                               absl::MakeSpan(results));

  work_queue->AddTask([&execution_context, &trace_me_context_id]() {
    tsl::profiler::TraceMeConsumer activity(
        [&] { return "RunMlrtFunction::Execute"; },
        tsl::profiler::ContextType::kTfrtExecutor, trace_me_context_id);
    mlrt::Execute(execution_context);
  });

  work_queue->Await(chain.CopyRCRef());

  if (execution_context.status().ok()) {
    combined_outputs->reserve(results.size());
    for (const auto& output : results) {
      combined_outputs->push_back(
          output.Get<tfrt_stub::FallbackTensor>().tensor());
    }
  }

  done(execution_context.status());
}

// The custom KernelFrame for tf_mlrt.batch_function op.
struct BatchFunctionOp : mlrt::KernelFrame {
  using KernelFrame::KernelFrame;

  static constexpr char kName[] = "tf_mlrt.batch_function";
  static constexpr bool kUseCustomDevice = false;

  mlrt::RegisterValueSpan<tfrt_stub::FallbackTensor> args() const {
    return arguments();
  }

  absl::string_view device_name() const {
    return attributes().GetAs<mlrt::bc::String>(0).Get();
  }

  tensorflow::Device* device() const {
    return context().fallback_request_state().cpu_device();
  }

  mlrt::bc::Function f() const {
    uint32_t func_idx = attributes().GetAs<uint32_t>(1);
    return execution_context()
        .loaded_executable()
        .executable()
        .functions()[func_idx];
  }

  absl::string_view node_def_text() const {
    return attributes().GetAs<mlrt::bc::String>(2).Get();
  }

  Context& context() const {
    return execution_context().GetUserContext<Context>();
  }

  void Invoke() { BatchFunctionInvokeHelper(*this); }
};

struct BatchFunctionWithDeviceOp : mlrt::KernelFrame {
  using KernelFrame::KernelFrame;

  static constexpr char kName[] = "tf_mlrt.batch_function.device";
  static constexpr bool kUseCustomDevice = true;

  // This is NOT the custom device name. Keep this for backwards compatibility.
  absl::string_view device_name() const {
    return attributes().GetAs<mlrt::bc::String>(0).Get();
  }

  mlrt::bc::Function f() const {
    uint32_t func_idx = attributes().GetAs<uint32_t>(1);
    return execution_context()
        .loaded_executable()
        .executable()
        .functions()[func_idx];
  }

  absl::string_view node_def_text() const {
    return attributes().GetAs<mlrt::bc::String>(2).Get();
  }

  Context& context() const {
    return execution_context().GetUserContext<Context>();
  }

  void Invoke() { BatchFunctionInvokeHelper(*this); }
  mlrt::RegisterValueSpan<tfrt_stub::FallbackTensor> args() const {
    return arguments().drop_front();
  }

  mlrt::bc::Span<uint8_t> last_uses() const {
    return KernelFrame::last_uses().drop_front();
  }

  const std::shared_ptr<tensorflow::Device>& device() const {
    return arguments()[0].Get<std::shared_ptr<tensorflow::Device>>();
  }
};

REGISTER_KERNEL_BUILDER(
    Name(kMlrtBatchFunctionName).Device(DEVICE_CPU),
    tfrt_stub::BatchFunctionFallbackKernel<MlrtBatchResource>);

// TFRT does not depend on the device annotation.
// MLRT Batch function will not actually execute on GPU, but rather on CPU.
// This kernel is registered on accelerator to get through the check.
REGISTER_KERNEL_BUILDER(
    Name(kMlrtBatchFunctionName).Device(DEVICE_GPU),
    tfrt_stub::BatchFunctionFallbackKernel<MlrtBatchResource>);

// Identical to BatchFunction except it has 2 extra TFRT attributes and it does
// not have `f` attribute. Users will not invoke this op directly.
REGISTER_OP(kMlrtBatchFunctionName)
    .Input("in_tensors: Tin")
    .Input("captured_tensors: Tcaptured")
    .Output("out_tensors: Tout")
    .Attr("num_batch_threads: int")
    .Attr("max_batch_size: int")
    .Attr("batch_timeout_micros: int")
    .Attr("max_enqueued_batches: int = 10")
    .Attr("allowed_batch_sizes: list(int) = []")
    .Attr("container: string = ''")
    .Attr("shared_name: string = ''")
    .Attr("batching_queue: string = ''")
    .Attr("Tin: list(type)")
    .Attr("Tcaptured: list(type) >= 0")
    .Attr("Tout: list(type)")
    .Attr("enable_large_batch_splitting: bool = false")
    .Attr("disable_padding: bool = false")
    // An opaque function handle, which is an int64_t, for passing the batch
    // function.
    .Attr("opaque_function_handle: int")
    .SetShapeFn(shape_inference::UnknownShape);

}  // namespace

// TODO(rohitju, chky): This additional Register is not ideal but unavoidable
// since the batch kernel libraries are very large. We should refactor the
// runtime_fallback lib to have only the necessary deps as a clean up and remove
// this Register function.
void RegisterTfMlrtBatchKernels(mlrt::KernelRegistry& registry) {
  registry.Register<BatchFunctionOp>();
  registry.Register<BatchFunctionWithDeviceOp>();
}

}  // namespace tf_mlrt
}  // namespace tensorflow
