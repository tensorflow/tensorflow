/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batch_kernels.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_resource_base.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler_utils.h"
#include "tensorflow/core/kernels/batching_util/bounded_executor.h"
#include "tensorflow/core/kernels/batching_util/concat_split_util.h"
#include "tensorflow/core/kernels/batching_util/periodic_function.h"
#include "tensorflow/core/kernels/batching_util/warmup.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace {
// Op attributes.
constexpr char kEnableAdaptiveSchedulerAttr[] = "_enable_adaptive_scheduler";
constexpr char kMinInflightBatchesAttr[] = "_min_inflight_batches";
constexpr char kInitialInflightBatchesAttr[] = "_initial_inflight_batches";
constexpr char kMaxInflightBatchesAttr[] = "_max_inflight_batches";
constexpr char kBatchesToAverageOverAttr[] = "_batches_to_average_over";
constexpr char kFullBatchSchedulingBoostMicros[] =
    "_full_batch_scheduling_boost_micros";

// Default thread count in the per-process batching thread pool.
constexpr int64_t kBatchThreadPoolSize = 128;
}  // namespace

// Per-model inflight batches parameters.
const int64_t kMinInflightBatches = 1;
const int64_t kInitialInflightBatches = 2;
const int64_t kBatchesToAverageOver = 10;
const int64_t kMaxInflightBatches = 64;

void RecordBatchSplitUsage(
    std::optional<bool> maybe_enable_large_batch_splitting,
    absl::string_view model_name) {
  static auto* cell = monitoring::Gauge<std::string, 1>::New(
      "/tensorflow/serving/batching/enable_large_batch_splitting",
      "Tracks the usage of attribute `enable_large_batch_splitting` for "
      "BatchFunction kernel in a saved model.",
      "model_name");
  if (maybe_enable_large_batch_splitting.has_value()) {
    if (maybe_enable_large_batch_splitting.value()) {
      cell->GetCell(std::string(model_name))->Set("true");
    } else {
      cell->GetCell(std::string(model_name))->Set("false");
    }
  } else {
    cell->GetCell(std::string(model_name))->Set("unset");
  }
}

void RecordBatchParamNumBatchThreads(int64_t num_batch_threads,
                                     absl::string_view model_name) {
  static auto* cell = monitoring::Gauge<int64_t, 1>::New(
      "/tensorflow/serving/batching/num_batch_threads",
      "Tracks the number of batch threads of a model.", "model_name");
  cell->GetCell(std::string(model_name))->Set(num_batch_threads);
}

absl::string_view GetModelName(OpKernelContext* ctx) {
  if (ctx->session_metadata() == nullptr ||
      ctx->session_metadata()->name().empty()) {
    return "model_name_unset";
  }
  return ctx->session_metadata()->name();
}

using ::tensorflow::concat_split_util::Concat;
using ::tensorflow::concat_split_util::Split;

int32 NumBatchThreadsFromEnvironmentWithDefault(int default_num_batch_threads) {
  int32_t num;
  const char* val = std::getenv("TF_NUM_BATCH_THREADS");

  return (val && absl::SimpleAtoi(val, &num)) ? num : default_num_batch_threads;
}

static thread::ThreadPool* GetOrCreateBatchThreadsPool() {
  static thread::ThreadPool* shared_thread_pool = [&]() -> thread::ThreadPool* {
    serving::BoundedExecutor::Options options;

    options.num_threads =
        NumBatchThreadsFromEnvironmentWithDefault(kBatchThreadPoolSize);

    options.thread_name = std::string("adaptive_batch_threads");

    auto status_or_executor = serving::BoundedExecutor::Create(options);
    if (!status_or_executor.ok()) {
      LOG(WARNING) << "Failed to create a batch threads pool with error "
                   << status_or_executor.status();
      return nullptr;
    }
    static serving::BoundedExecutor* executor =
        status_or_executor.value().release();
    return new thread::ThreadPool(executor);
  }();
  return shared_thread_pool;
}

// A class encapsulating the state and logic for batching tensors.
class BatchResource : public serving::BatchResourceBase {
 public:
  struct BatchTask : serving::BatchResourceBase::BatchTask {
    FunctionLibraryRuntime::Handle fhandle;

    explicit BatchTask(FunctionLibraryRuntime::Handle fhandle)
        : fhandle(fhandle) {}

   protected:
    std::unique_ptr<serving::BatchResourceBase::BatchTask> CreateDerivedTask()
        override {
      return std::make_unique<BatchTask>(fhandle);
    }
  };

  static absl::Status Create(bool has_process_batch_function,
                             int32_t num_batch_threads,
                             int32_t max_execution_batch_size,
                             int32_t batch_timeout_micros,
                             int32_t max_enqueued_batches,
                             const std::vector<int32>& allowed_batch_sizes,
                             bool enable_large_batch_splitting,
                             std::unique_ptr<BatchResource>* resource) {
    return Create(has_process_batch_function, num_batch_threads,
                  max_execution_batch_size, batch_timeout_micros,
                  max_enqueued_batches, allowed_batch_sizes,
                  /*low_priority_max_batch_size=*/0,
                  /*low_priority_batch_timeout_micros=*/0,
                  /*low_priority_max_enqueued_batches=*/0,
                  /*low_priority_allowed_batch_sizes=*/{},
                  /*mixed_priority_batching_policy=*/
                  serving::MixedPriorityBatchingPolicy::
                      kLowPriorityPaddingWithMaxBatchSize,
                  enable_large_batch_splitting,
                  /*batch_padding_policy=*/"PAD_UP", resource);
  }

  static absl::Status Create(
      bool has_process_batch_function, int32_t num_batch_threads,
      int32_t max_execution_batch_size, int32_t batch_timeout_micros,
      int32_t max_enqueued_batches,
      const std::vector<int32>& allowed_batch_sizes,
      int32_t low_priority_max_batch_size,
      int32_t low_priority_batch_timeout_micros,
      int32_t low_priority_max_enqueued_batches,
      const std::vector<int32>& low_priority_allowed_batch_sizes,
      serving::MixedPriorityBatchingPolicy mixed_priority_batching_policy,
      bool enable_large_batch_splitting, absl::string_view batch_padding_policy,
      std::unique_ptr<BatchResource>* resource) {
    BatcherT::Options batcher_options;
    batcher_options.num_batch_threads = num_batch_threads;
    std::shared_ptr<BatcherT> batcher;
    TF_RETURN_IF_ERROR(BatcherT::Create(batcher_options, &batcher));

    resource->reset(new BatchResource(
        has_process_batch_function, std::move(batcher),
        GetBatcherQueueOptions(
            num_batch_threads, max_execution_batch_size, batch_timeout_micros,
            max_enqueued_batches, allowed_batch_sizes,
            enable_large_batch_splitting,
            /*disable_padding=*/false, batch_padding_policy,
            low_priority_max_batch_size, low_priority_batch_timeout_micros,
            low_priority_max_enqueued_batches, low_priority_allowed_batch_sizes,
            mixed_priority_batching_policy),
        allowed_batch_sizes));
    return absl::OkStatus();
  }

  static absl::Status Create(
      bool has_process_batch_function,
      AdaptiveBatcherT::Options adaptive_shared_batch_scheduler_options,
      int32_t max_batch_size, int32_t batch_timeout_micros,
      int32_t max_enqueued_batches,
      const std::vector<int32>& allowed_batch_sizes,
      std::unique_ptr<BatchResource>* resource) {
    std::shared_ptr<AdaptiveBatcherT> batcher;
    TF_RETURN_IF_ERROR(AdaptiveBatcherT::Create(
        adaptive_shared_batch_scheduler_options, &batcher));

    resource->reset(new BatchResource(
        has_process_batch_function, std::move(batcher),
        GetAdaptiveBatcherQueueOptions(
            max_batch_size, batch_timeout_micros, max_enqueued_batches,
            /*enable_large_batch_splitting=*/true, allowed_batch_sizes,
            /*disable_padding=*/false),
        allowed_batch_sizes));
    return absl::OkStatus();
  }

  string DebugString() const final { return "BatchResource"; }

 private:
  BatchResource(bool has_process_batch_function,
                std::shared_ptr<BatcherT> batcher,
                const BatcherT::QueueOptions& batcher_queue_options,
                std::vector<int32> allowed_batch_sizes)
      : BatchResourceBase(has_process_batch_function, std::move(batcher),
                          batcher_queue_options,
                          std::move(allowed_batch_sizes)) {}

  BatchResource(bool has_process_batch_function,
                std::shared_ptr<AdaptiveBatcherT> batcher,
                const AdaptiveBatcherT::QueueOptions& batcher_queue_options,
                std::vector<int32> allowed_batch_sizes)
      : BatchResourceBase(has_process_batch_function, std::move(batcher),
                          batcher_queue_options,
                          std::move(allowed_batch_sizes)) {}

  void ProcessFuncBatchImpl(
      const serving::BatchResourceBase::BatchTask& last_task,
      absl::Span<const Tensor> inputs, std::vector<Tensor>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override {
    auto* last_task_context = last_task.context;
    FunctionLibraryRuntime::Options opts;
    opts.step_container = last_task_context->step_container();
    opts.cancellation_manager = last_task_context->cancellation_manager();
    opts.collective_executor = last_task_context->collective_executor();
    opts.stats_collector = last_task_context->stats_collector();
    opts.runner = last_task_context->runner();
    opts.run_all_kernels_inline = last_task_context->run_all_kernels_inline();
    // We do not set 'opts.rendezvous', since if the function is run multiple
    // times in parallel with the same rendezvous, a _Send node from one run
    // might be matched with a _Recv node of a different run. Not setting the
    // rendezvous causes a new rendezvous to be used for each run.
    Notification done_notif;

    auto* flib = last_task_context->function_library();
    FunctionLibraryRuntime::Handle fhandle =
        down_cast<const BatchTask&>(last_task).fhandle;
    flib->Run(opts, fhandle, inputs, combined_outputs,
              [&](const absl::Status& run_status) {
                done(run_status);
                done_notif.Notify();
              });
    // By waiting for the notification we are ensuring that this thread isn't
    // used for processing other batches, which gives the batches time to
    // coalesce upstream. So overall the number of batches going through the
    // devices goes down, improving latency and throughput in most cases.
    done_notif.WaitForNotification();
  }
};

BatchFunctionKernel::BatchFunctionKernel(OpKernelConstruction* c)
    : AsyncOpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("container", &container_));
  OP_REQUIRES_OK(c, c->GetAttr("shared_name", &shared_name_));
  OP_REQUIRES_OK(c, c->GetAttr("batching_queue", &batcher_queue_));
  OP_REQUIRES_OK(c, c->GetAttr("num_batch_threads", &num_batch_threads_));
  OP_REQUIRES_OK(c, c->GetAttr("max_batch_size", &max_batch_size_));
  OP_REQUIRES_OK(c, c->GetAttr("batch_timeout_micros", &batch_timeout_micros_));
  OP_REQUIRES_OK(c, c->GetAttr("max_enqueued_batches", &max_enqueued_batches_));
  OP_REQUIRES_OK(c, c->GetAttr("allowed_batch_sizes", &allowed_batch_sizes_));
  OP_REQUIRES_OK(c, c->GetAttr("low_priority_max_batch_size",
                               &low_priority_max_batch_size_));
  OP_REQUIRES_OK(c, c->GetAttr("low_priority_batch_timeout_micros",
                               &low_priority_batch_timeout_micros_));
  OP_REQUIRES_OK(c, c->GetAttr("low_priority_allowed_batch_sizes",
                               &low_priority_allowed_batch_sizes_));
  OP_REQUIRES_OK(c, c->GetAttr("low_priority_max_enqueued_batches",
                               &low_priority_max_enqueued_batches_));
  OP_REQUIRES_OK(c,
                 c->GetAttr("mixed_priority_policy", &mixed_priority_policy_));
  OP_REQUIRES_OK(c, c->GetAttr("batch_padding_policy", &batch_padding_policy_));

  OP_REQUIRES_OK(c, c->GetAttr("f", &func_));

  if (c->HasAttr("enable_large_batch_splitting")) {
    OP_REQUIRES_OK(c, c->GetAttr("enable_large_batch_splitting",
                                 &enable_large_batch_splitting_));
    has_attribute_enable_large_batch_splitting_ = true;
  }

  // Helper function `SetAdaptiveBatchSchedulerOptions` calls
  // `OP_REQUIRES_OK`, which exits the current function upon error.
  // So validate status of `op-kernel-construction`.
  SetAdaptiveBatchSchedulerOptions(c, num_batch_threads_);
  if (!c->status().ok()) {
    return;
  }

  if (enable_adaptive_batch_threads_) {
    // One scheduler instance contains a couple of queue instances,
    // `batcher_queue_` is the key to find queue for this batch-op in the
    // graph.
    // Use `shared_name_` and name() as prefix for `batcher_queue_`.
    // Note name() is node_def.name so unique per graph def.
    batcher_queue_ = name() + "/" + shared_name_ + batcher_queue_;
  }

  if (shared_name_.empty()) {
    // If shared_name is not supplied, use name instead (prevent collisions by
    // default).
    shared_name_ = name();
  }

  OP_REQUIRES_OK(c, ValidateAllowedBatchSizes());
}

bool BatchFunctionKernel::IsExpensive() { return false; }

void BatchFunctionKernel::ComputeAsync(OpKernelContext* c, DoneCallback done) {
  RecordBatchSplitUsage(has_attribute_enable_large_batch_splitting_
                            ? std::make_optional(enable_large_batch_splitting_)
                            : std::nullopt,
                        GetModelName(c));
  RecordBatchParamNumBatchThreads(num_batch_threads_, GetModelName(c));

  std::function<absl::Status(BatchResource**)> creator;

  FunctionLibraryRuntime::Handle handle;
  OP_REQUIRES_OK_ASYNC(c, GetOrCreateFunctionHandle(c, &handle), done);

  if (adaptive_batch_scheduler_options_ != std::nullopt) {
    creator = [this,
               session_metadata = c->session_metadata()](BatchResource** r) {
      serving::AdaptiveSharedBatchScheduler<
          serving::BatchResourceBase::BatchTask>::Options
          adaptive_shared_batch_scheduler_options;
      adaptive_shared_batch_scheduler_options.thread_pool_name =
          "adaptive_batch_threads";
      adaptive_shared_batch_scheduler_options.thread_pool =
          GetOrCreateBatchThreadsPool();

      // When we explicitly specify 'thread_pool', you'd think ASBS would ignore
      // 'num_batch_threads', but in fact ASBS still uses num_batch_threads as
      // the max number of in-flight batches.  It makes no sense to have more
      // in-flight batches than threads (it would result in strictly bad
      // batching decisions), so we cap this parameter (which otherwise comes
      // from the saved model) to the actual number of batch threads (which
      // comes from a process-wide environment variable).
      //
      // We have to apply the same capping to min_ and initial_
      // in_flight_batches_limit below to produce valid configurations.
      adaptive_shared_batch_scheduler_options.num_batch_threads = std::min(
          NumBatchThreadsFromEnvironmentWithDefault(kBatchThreadPoolSize),
          adaptive_batch_scheduler_options_->max_in_flight_batches_limit);

      // adaptive_shared_batch_scheduler_options.full_batch_scheduling_boost_micros
      // is 0 (default value) intentionally, so tasks are scheduled in a FIFO
      // way.
      // Two rationales to use default value (zero) for
      // `full_batch_scheduling_boost_micros`
      // 1) In this way, tasks scheduling policy is FIFO. Compared with round
      // robin (what shared batch scheduler does), FIFO ensures that model
      // with low QPS (i.e., models enqueue fewer tasks in the shared queue)
      // will be processed timely.
      // 2) If set, `full_batch_scheduling_boost_micros` should be of order
      // the batch processing latency (which varies on a model basis).
      // If a non-zero value is not set properly, it harms tail latency.
      adaptive_shared_batch_scheduler_options.min_in_flight_batches_limit =
          std::min(
              NumBatchThreadsFromEnvironmentWithDefault(kBatchThreadPoolSize),
              adaptive_batch_scheduler_options_->min_in_flight_batches_limit);
      adaptive_shared_batch_scheduler_options
          .initial_in_flight_batches_limit = std::min(
          NumBatchThreadsFromEnvironmentWithDefault(kBatchThreadPoolSize),
          adaptive_batch_scheduler_options_->initial_in_flight_batches_limit);
      adaptive_shared_batch_scheduler_options.batches_to_average_over =
          adaptive_batch_scheduler_options_->batches_to_average_over;
      if (adaptive_batch_scheduler_options_
              ->full_batch_scheduling_boost_micros != -1) {
        adaptive_shared_batch_scheduler_options
            .full_batch_scheduling_boost_micros =
            adaptive_batch_scheduler_options_
                ->full_batch_scheduling_boost_micros;
        adaptive_shared_batch_scheduler_options.fifo_scheduling = false;
      } else {
        adaptive_shared_batch_scheduler_options.fifo_scheduling = true;
      }
      std::unique_ptr<BatchResource> new_resource;
      TF_RETURN_IF_ERROR(BatchResource::Create(
          /*has_process_batch_function=*/true,
          adaptive_shared_batch_scheduler_options, max_batch_size_,
          batch_timeout_micros_, max_enqueued_batches_, allowed_batch_sizes_,
          &new_resource));
      if (session_metadata) {
        new_resource->set_session_metadata(*session_metadata);
      }
      *r = new_resource.release();
      return absl::OkStatus();
    };
  } else {
    creator = [this,
               session_metadata = c->session_metadata()](BatchResource** r) {
      TF_ASSIGN_OR_RETURN(
          serving::MixedPriorityBatchingPolicy mixed_priority_batching_policy,
          serving::GetMixedPriorityBatchingPolicy(mixed_priority_policy_));

      std::unique_ptr<BatchResource> new_resource;
      TF_RETURN_IF_ERROR(BatchResource::Create(
          /*has_process_batch_function=*/true, num_batch_threads_,
          max_batch_size_, batch_timeout_micros_, max_enqueued_batches_,
          allowed_batch_sizes_, low_priority_max_batch_size_,
          low_priority_batch_timeout_micros_,
          low_priority_max_enqueued_batches_, low_priority_allowed_batch_sizes_,
          mixed_priority_batching_policy, enable_large_batch_splitting_,
          batch_padding_policy_, &new_resource));
      if (session_metadata) {
        new_resource->set_session_metadata(*session_metadata);
      }
      *r = new_resource.release();
      return absl::OkStatus();
    };
  }

  BatchResource* br;
  OP_REQUIRES_OK_ASYNC(c,
                       c->resource_manager()->LookupOrCreate(
                           container_, shared_name_, &br, creator),
                       done);
  const uint64_t guid = random::New64();
  auto create_batch_task_fn =
      [handle]() -> absl::StatusOr<
                     std::unique_ptr<serving::BatchResourceBase::BatchTask>> {
    return {std::make_unique<BatchResource::BatchTask>(handle)};
  };
  absl::Status status;
  if (serving::ShouldWarmupAllBatchSizes(c)) {
    status = br->RegisterWarmupInputs(guid, c, batcher_queue_,
                                      create_batch_task_fn, done);
  } else {
    status =
        br->RegisterInput(guid, c, batcher_queue_, create_batch_task_fn, done);
  }
  br->Unref();
  OP_REQUIRES_OK_ASYNC(c, status, done);
  // Assume br calls done, so nothing to do here.
}

absl::Status BatchFunctionKernel::InstantiateFunction(
    OpKernelContext* c, FunctionLibraryRuntime::Handle* handle) const {
  // TODO(b/173748062): Merge this instantiation logic with PartitionedCall.
  FunctionLibraryRuntime* flib = c->function_library();
  if (!flib) {
    return errors::Internal("No function library");
  }

  FunctionLibraryRuntime::InstantiateOptions opts;
  opts.target = flib->device() == nullptr ? "" : flib->device()->name();
  opts.is_multi_device_function = true;
  const ConfigProto* config = flib->config_proto();
  if (config) {
    opts.config_proto = *config;
  }

  Device* cpu_device;
  TF_RETURN_IF_ERROR(flib->device_mgr()->LookupDevice("CPU:0", &cpu_device));

  const FunctionDef* fdef =
      flib->GetFunctionLibraryDefinition()->Find(func_.name());
  if (!fdef) {
    return errors::NotFound("Failed to find definition for function \"",
                            func_.name(), "\"");
  }
  OpInputList in_tensors;
  TF_RETURN_IF_ERROR(c->input_list("in_tensors", &in_tensors));
  for (int i = 0; i < in_tensors.size(); i++) {
    if (in_tensors[i].dtype() == DT_RESOURCE) {
      return errors::InvalidArgument(
          "BatchFunction cannot take resource inputs but input ", i,
          " is a resource.");
    } else {
      // Currently, inputs are on CPU since they are concatenated on CPU
      opts.input_devices.push_back(cpu_device->name());
    }
  }
  OpInputList captured_tensors;
  TF_RETURN_IF_ERROR(c->input_list("captured_tensors", &captured_tensors));
  for (const Tensor& t : captured_tensors) {
    if (t.dtype() == DT_RESOURCE) {
      const ResourceHandle& rhandle = t.flat<ResourceHandle>()(0);
      opts.input_devices.push_back(rhandle.device());
    } else {
      opts.input_devices.push_back(cpu_device->name());
    }
  }
  const OpDef& signature = fdef->signature();
  for (int i = 0; i < signature.output_arg_size(); i++) {
    // Currently, outputs must be on CPU since they are split on CPU.
    opts.output_devices.push_back(cpu_device->name());
  }
  if (opts.input_devices.size() != signature.input_arg_size()) {
    return errors::InvalidArgument(
        "Function takes ", signature.input_arg_size(), " argument(s) but ",
        opts.input_devices.size(), " argument(s) were passed");
  }
  return flib->Instantiate(func_.name(), AttrSlice(&func_.attr()), opts,
                           handle);
}

absl::Status BatchFunctionKernel::GetOrCreateFunctionHandle(
    OpKernelContext* c, FunctionLibraryRuntime::Handle* handle) {
  mutex_lock ml(mu_);
  if (!fhandle_) {
    TF_RETURN_IF_ERROR(InstantiateFunction(c, handle));
    fhandle_ = *handle;
  } else {
    *handle = fhandle_.value();
  }
  return absl::OkStatus();
}

// Validates 'allowed_batch_sizes_'. The entries must increase monotonically.
// If large batch split is not enabled, the last one must equal
// `max_batch_size_`. otherwise the last element must be smaller than or equal
// to `max_batch_size_`.
absl::Status BatchFunctionKernel::ValidateAllowedBatchSizes() const {
  if (allowed_batch_sizes_.empty()) {
    return absl::OkStatus();
  }
  int32_t last_size = 0;
  for (size_t i = 0; i < allowed_batch_sizes_.size(); ++i) {
    const int32_t size = allowed_batch_sizes_.at(i);
    if (i > 0 && size <= last_size) {
      return errors::InvalidArgument(
          "allowed_batch_sizes entries must be monotonically increasing");
    }

    if ((!enable_large_batch_splitting_) &&
        (i == allowed_batch_sizes_.size() - 1) && (size != max_batch_size_)) {
      return errors::InvalidArgument(
          "final entry in allowed_batch_sizes must equal max_batch_size when "
          "enable_large_batch_splitting is False");
    }

    last_size = size;
  }
  return absl::OkStatus();
}

// Initialize vars by reading from op-kernel-construction.
// Vars
// - enable_adaptive_batch_threads_
//   true if value of attribute `kEnableAdaptiveSchedulerAttr` is true, or
//   if `num_batch_threads` is not positive.
// - adaptive_batch_scheduler_options_
//   Read from corresponding attributes as long as they are set.
void BatchFunctionKernel::SetAdaptiveBatchSchedulerOptions(
    OpKernelConstruction* c, int32_t num_batch_threads) {
  if (c->HasAttr(kEnableAdaptiveSchedulerAttr)) {
    OP_REQUIRES_OK(c, c->GetAttr(kEnableAdaptiveSchedulerAttr,
                                 &enable_adaptive_batch_threads_));
  }

  if (num_batch_threads <= 0) {
    enable_adaptive_batch_threads_ = true;
  }

  if (!enable_adaptive_batch_threads_) {
    // adaptive_batch_scheduler_options_ is nullopt.
    return;
  }

  // adaptive_batch_scheduler_options_ is not nullopt
  AdaptiveBatchSchedulerOptions options;

  if (c->HasAttr(kBatchesToAverageOverAttr)) {
    OP_REQUIRES_OK(c, c->GetAttr(kBatchesToAverageOverAttr,
                                 &options.batches_to_average_over));
  }

  if (c->HasAttr(kMinInflightBatchesAttr)) {
    OP_REQUIRES_OK(c, c->GetAttr(kMinInflightBatchesAttr,
                                 &options.min_in_flight_batches_limit));
  }

  if (c->HasAttr(kInitialInflightBatchesAttr)) {
    OP_REQUIRES_OK(c, c->GetAttr(kInitialInflightBatchesAttr,
                                 &options.initial_in_flight_batches_limit));
  }

  if (c->HasAttr(kMaxInflightBatchesAttr)) {
    OP_REQUIRES_OK(c, c->GetAttr(kMaxInflightBatchesAttr,
                                 &options.max_in_flight_batches_limit));
  }

  if (c->HasAttr(kFullBatchSchedulingBoostMicros)) {
    OP_REQUIRES_OK(c, c->GetAttr(kFullBatchSchedulingBoostMicros,
                                 &options.full_batch_scheduling_boost_micros));
  }

  // At this point, the batch kernel is configured to use adaptive scheduling.
  // To validate or return error at kernel construction time, invokes
  // `GetOrCreateBatchThreadsPool` and validates returned `thread_pool` is
  // valid.
  // Note`GetOrCreateBatchThreadsPool` creates the thread pool once and
  // re-uses the thread-pool instance afterwards.
  thread::ThreadPool* thread_pool = GetOrCreateBatchThreadsPool();
  OP_REQUIRES(
      c, thread_pool != nullptr,
      errors::FailedPrecondition("Failed to create batch threads pool"));

  adaptive_batch_scheduler_options_ = options;
}
REGISTER_KERNEL_BUILDER(Name("BatchFunction").Device(DEVICE_CPU),
                        BatchFunctionKernel);
// Currently all inputs and outputs are on the host.
// TODO(b/173748277): Accept inputs/outputs on the device.
REGISTER_KERNEL_BUILDER(Name("BatchFunction")
                            .Device(DEVICE_GPU)
                            .HostMemory("in_tensors")
                            .HostMemory("captured_tensors")
                            .HostMemory("out_tensors"),
                        BatchFunctionKernel);
REGISTER_KERNEL_BUILDER(Name("BatchFunction")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("in_tensors")
                            .HostMemory("captured_tensors")
                            .HostMemory("out_tensors"),
                        BatchFunctionKernel);

class BatchKernel : public AsyncOpKernel {
 public:
  explicit BatchKernel(OpKernelConstruction* c) : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("container", &container_));
    OP_REQUIRES_OK(c, c->GetAttr("shared_name", &shared_name_));
    // If shared_name is not supplied, use name instead (prevent collisions by
    // default).
    if (shared_name_.empty()) {
      shared_name_ = name();
    }
    OP_REQUIRES_OK(c, c->GetAttr("batching_queue", &batcher_queue_));
    OP_REQUIRES_OK(c, c->GetAttr("num_batch_threads", &num_batch_threads_));
    OP_REQUIRES_OK(c, c->GetAttr("max_batch_size", &max_batch_size_));
    OP_REQUIRES_OK(c,
                   c->GetAttr("batch_timeout_micros", &batch_timeout_micros_));
    OP_REQUIRES_OK(c,
                   c->GetAttr("max_enqueued_batches", &max_enqueued_batches_));
    OP_REQUIRES_OK(c, c->GetAttr("allowed_batch_sizes", &allowed_batch_sizes_));
    OP_REQUIRES_OK(c, ValidateAllowedBatchSizes());
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) final {
    BatchResource* br;
    std::function<absl::Status(BatchResource**)> creator =
        [this](BatchResource** r) {
          std::unique_ptr<BatchResource> new_resource;
          TF_RETURN_IF_ERROR(BatchResource::Create(
              /*has_process_batch_function=*/false, num_batch_threads_,
              max_batch_size_, batch_timeout_micros_, max_enqueued_batches_,
              allowed_batch_sizes_, false, &new_resource));
          *r = new_resource.release();
          return absl::OkStatus();
        };
    OP_REQUIRES_OK_ASYNC(c,
                         c->resource_manager()->LookupOrCreate(
                             container_, shared_name_, &br, creator),
                         done);
    const absl::Status status = br->RegisterInput(
        random::New64(), c, batcher_queue_,
        []() -> absl::StatusOr<
                 std::unique_ptr<serving::BatchResourceBase::BatchTask>> {
          return {std::make_unique<BatchResource::BatchTask>(kInvalidHandle)};
        },
        done);
    br->Unref();
    OP_REQUIRES_OK_ASYNC(c, status, done);
    // Assume br calls done, so nothing to do here.
  }

  // Validates 'allowed_batch_sizes_'. The entries must increase
  // monotonically, and the last one must equal 'max_batch_size_'.
  absl::Status ValidateAllowedBatchSizes() const {
    if (allowed_batch_sizes_.empty()) {
      return absl::OkStatus();
    }
    int32_t last_size = 0;
    for (size_t i = 0; i < allowed_batch_sizes_.size(); ++i) {
      const int32_t size = allowed_batch_sizes_.at(i);
      if (i > 0 && size <= last_size) {
        return errors::InvalidArgument(
            "allowed_batch_sizes entries must be monotonically increasing");
      }
      if (i == allowed_batch_sizes_.size() - 1 && size != max_batch_size_) {
        return errors::InvalidArgument(
            "final entry in allowed_batch_sizes must equal max_batch_size");
      }
      last_size = size;
    }
    return absl::OkStatus();
  }

 private:
  string container_;
  string shared_name_;
  string batcher_queue_;
  int32 num_batch_threads_;
  int32 max_batch_size_;
  int32 batch_timeout_micros_;
  int32 max_enqueued_batches_;
  std::vector<int32> allowed_batch_sizes_;
};

REGISTER_KERNEL_BUILDER(Name("Batch").Device(DEVICE_CPU), BatchKernel);

// A class encapsulating the state and logic for unbatching tensors.
//
// UnbatchResource keeps two data structures indexed by batch-key: one which has
// the continuations for all concurrent kernels which are waiting for tensors
// and another which has tensors which are waiting for their corresponding
// kernels to run. Whenever a kernel runs, we either grab its tensor if it's
// waiting already, or we insert it in the queue and then look at its tensor to
// see if it can be used to dispatch any stored continuations.
class UnbatchResource : public ResourceBase {
 public:
  explicit UnbatchResource(int32_t timeout_micros)
      : timeout_micros_(timeout_micros),
        timeout_enforcer_(new serving::PeriodicFunction(
            [this] { EnforceTimeout(); }, 1000 /* 1 ms */)) {}

  ~UnbatchResource() override {
    // Tear down 'timeout_enforcer_' first, since it accesses other state in
    // this class.
    timeout_enforcer_ = nullptr;
  }

  string DebugString() const final { return "UnbatchResource"; }

  absl::Status Compute(OpKernelContext* context,
                       AsyncOpKernel::DoneCallback done) {
    const Tensor& data_t = context->input(0);
    const Tensor& batch_index_t = context->input(1);

    if (batch_index_t.shape().dim_size(0) > data_t.shape().dim_size(0)) {
      return errors::InvalidArgument(
          "Wrong shape for index tensor. Expected 0th dimension size to be no "
          "greater than ",
          data_t.shape().dim_size(0),
          "; Got: ", batch_index_t.shape().dim_size(0), ".");
    }
    if (batch_index_t.shape().dim_size(1) != 3) {
      return errors::InvalidArgument(
          "Wrong shape for index tensor. Expected 1st dimension size to be 3 ; "
          "Got: ",
          batch_index_t.shape().dim_size(1), ".");
    }

    if (!TensorShapeUtils::IsScalar(context->input(2).shape())) {
      return errors::InvalidArgument(
          "Input id should be scalar; "
          "Got: ",
          context->input(2).DebugString(), ".");
    }
    const int64_t batch_key = context->input(2).scalar<int64_t>()();
    const bool nonempty_input = batch_index_t.dim_size(0) > 0;

    // If we have a non-empty tensor, slice it up.
    // (It is important to do this outside of the critical section below.)
    // The following variables are populated iff 'nonempty_input==true'.
    std::vector<int64_t> sizes;
    std::vector<int64_t> batch_keys;
    std::vector<Tensor> split_inputs;
    if (nonempty_input) {
      auto batch_indices =
          batch_index_t.shaped<int64_t, 2>({batch_index_t.dim_size(0), 3});
      for (int i = 0; i < batch_index_t.dim_size(0); ++i) {
        sizes.push_back(batch_indices(i, 2) - batch_indices(i, 1));
        batch_keys.push_back(batch_indices(i, 0));
      }

      TF_RETURN_IF_ERROR(Split(context, data_t, sizes, &split_inputs));
    }

    // Critical section.
    std::vector<AsyncOpKernel::DoneCallback> done_callbacks_to_call;
    absl::Status status = [&]() -> absl::Status {
      mutex_lock ml(mu_);

      // Check to see whether the tensor we want is already ready.
      auto tensor_it = waiting_tensors_.find(batch_key);
      if (tensor_it != waiting_tensors_.end()) {
        context->set_output(0, tensor_it->second.tensor);
        waiting_tensors_.erase(tensor_it);
        done_callbacks_to_call.push_back(done);
        return absl::OkStatus();
      }

      const uint64 deadline_micros =
          Env::Default()->NowMicros() + timeout_micros_;

      // Add ourselves to the waitlist for tensors.
      if (!waiting_callbacks_
               .emplace(batch_key,
                        WaitingCallback{deadline_micros, context, done})
               .second) {
        return errors::AlreadyExists(
            "Multiple session runs with the same batch key.");
      }

      // If we have a non-empty tensor, finish the waitlisted runs,
      // and store any remaining pieces.
      if (nonempty_input) {
        for (size_t i = 0; i < batch_keys.size(); ++i) {
          auto runs_it = waiting_callbacks_.find(batch_keys[i]);
          if (runs_it != waiting_callbacks_.end()) {
            runs_it->second.context->set_output(0, split_inputs[i]);
            done_callbacks_to_call.push_back(runs_it->second.done);
            waiting_callbacks_.erase(runs_it);
          } else {
            // Note: the deadline here is in case we are arriving late and the
            // kernel that should rendezvous with this tensor has already waited
            // and timed out.
            if (!waiting_tensors_
                     .emplace(batch_keys[i],
                              WaitingTensor{deadline_micros, split_inputs[i]})
                     .second) {
              return errors::AlreadyExists(
                  "Multiple tensors returned for same batch key.");
            }
          }
        }
      }

      return absl::OkStatus();
    }();

    for (const AsyncOpKernel::DoneCallback& done_callback :
         done_callbacks_to_call) {
      done_callback();
    }

    return status;
  }

 private:
  // Evicts waiting tensors and callbacks that have exceeded their deadline.
  void EnforceTimeout() {
    const uint64 now = Env::Default()->NowMicros();
    std::vector<WaitingCallback> evicted_callbacks;

    {
      mutex_lock ml(mu_);

      for (auto it = waiting_tensors_.begin(); it != waiting_tensors_.end();) {
        const WaitingTensor& waiting_tensor = it->second;
        if (waiting_tensor.deadline_micros < now) {
          it = waiting_tensors_.erase(it);
        } else {
          ++it;
        }
      }

      for (auto it = waiting_callbacks_.begin();
           it != waiting_callbacks_.end();) {
        const WaitingCallback& waiting_callback = it->second;
        if (waiting_callback.deadline_micros < now) {
          evicted_callbacks.push_back(waiting_callback);
          it = waiting_callbacks_.erase(it);
        } else {
          ++it;
        }
      }
    }

    for (const WaitingCallback& evicted_callback : evicted_callbacks) {
      evicted_callback.context->CtxFailureWithWarning(errors::DeadlineExceeded(
          "Batched data did not arrive within timeout window."));
      evicted_callback.done();
    }
  }

  struct WaitingTensor {
    uint64 deadline_micros;
    Tensor tensor;
  };

  struct WaitingCallback {
    uint64 deadline_micros;
    OpKernelContext* context;
    AsyncOpKernel::DoneCallback done;
  };

  const int32 timeout_micros_;

  mutex mu_;

  // Maps keyed by BatchKey of tensors waiting for callbacks and callbacks
  // waiting for tensors.
  std::unordered_map<int64_t, WaitingTensor> waiting_tensors_
      TF_GUARDED_BY(mu_);
  std::unordered_map<int64_t, WaitingCallback> waiting_callbacks_
      TF_GUARDED_BY(mu_);

  // A thread that evicts waiting tensors and callbacks that have exceeded their
  // deadline.
  std::unique_ptr<serving::PeriodicFunction> timeout_enforcer_;
};

class UnbatchKernel : public AsyncOpKernel {
 public:
  explicit UnbatchKernel(OpKernelConstruction* c) : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("container", &container_));
    OP_REQUIRES_OK(c, c->GetAttr("shared_name", &shared_name_));
    // If shared_name is not supplied, use name instead (prevent collisions by
    // default).
    if (shared_name_.empty()) {
      shared_name_ = name();
    }
    OP_REQUIRES_OK(c, c->GetAttr("timeout_micros", &timeout_micros_));
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) final {
    UnbatchResource* ubr;
    std::function<absl::Status(UnbatchResource**)> creator =
        [this](UnbatchResource** r) {
          *r = new UnbatchResource(timeout_micros_);
          return absl::OkStatus();
        };
    OP_REQUIRES_OK_ASYNC(c,
                         c->resource_manager()->LookupOrCreate(
                             container_, shared_name_, &ubr, creator),
                         done);
    auto status = ubr->Compute(c, done);
    ubr->Unref();
    OP_REQUIRES_OK_ASYNC(c, status, done);
    // Assume ubr calls done, so nothing to do here.
  }

 private:
  string container_;
  string shared_name_;
  int32 timeout_micros_;
};
REGISTER_KERNEL_BUILDER(Name("Unbatch").Device(DEVICE_CPU), UnbatchKernel);

// A class encapsulating the state and logic for batching tensors
// deterministically for the gradient of unbatch.
class UnbatchGradResource : public ResourceBase {
 public:
  UnbatchGradResource() {}

  string DebugString() const final { return "UnbatchGradResource"; }

  // Flushes the information for one batch, given its context and done
  // callback. Clears all information about it from the available_tensors_.
  absl::Status OutputBatch(OpKernelContext* context,
                           const AsyncOpKernel::DoneCallback& done)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    const Tensor& batch_index_t = context->input(1);
    auto batch_index =
        batch_index_t.shaped<int64_t, 2>({batch_index_t.dim_size(0), 3});
    std::vector<Tensor> tensors;
    for (int i = 0; i < batch_index_t.dim_size(0); ++i) {
      auto available_it = available_tensors_.find(batch_index(i, 0));
      if (available_it == available_tensors_.end()) {
        return errors::Internal("bad bookkeeping of available tensors.");
      }
      tensors.push_back(available_it->second);
      available_tensors_.erase(available_it);
    }

    const DataType type = tensors[0].dtype();
    Tensor concatenated_tensor;
    switch (type) {
#define CASE(type)                                                            \
  case DataTypeToEnum<type>::value:                                           \
    TF_RETURN_IF_ERROR(Concat<type>(context, tensors, &concatenated_tensor)); \
    context->set_output(0, concatenated_tensor);                              \
    break;
      TF_CALL_ALL_TYPES(CASE);
#undef CASE
      default:
        return errors::InvalidArgument("Unsupported data type: ", type);
    }
    done();
    return absl::OkStatus();
  }

  // Ingests data from one invocation of the op.
  absl::Status Compute(OpKernelContext* context,
                       const AsyncOpKernel::DoneCallback& done) {
    const Tensor& data_t = context->input(0);
    const Tensor& batch_index_t = context->input(1);
    const Tensor& grad_t = context->input(2);
    const Tensor& batch_key_t = context->input(3);

    mutex_lock ml(mu_);
    if (!TensorShapeUtils::IsScalar(batch_key_t.shape())) {
      return errors::InvalidArgument("Expected `id` to be scalar. Received ",
                                     batch_key_t.DebugString());
    }

    const int64_t batch_key = context->input(3).scalar<int64_t>()();
    // Mark our tensor as available.
    if (!available_tensors_.emplace(batch_key, grad_t).second) {
      return errors::InvalidArgument("Two runs with the same batch key.");
    }

    // Check whether we have a valid input tensor and, if so, create its
    // dispatch logic.
    if (data_t.NumElements() > 0) {
      if (batch_index_t.NumElements() == 0) {
        return errors::InvalidArgument(
            "batch_index is empty while the tensor isn't.");
      }
      std::unordered_set<int64_t> missing_tensors;
      if (batch_index_t.NumElements() != batch_index_t.dim_size(0) * 3) {
        return errors::InvalidArgument(
            "batch_index should contain ", batch_index_t.dim_size(0) * 3,
            " elements. Received ", batch_index_t.NumElements());
      }
      const auto batch_index =
          batch_index_t.shaped<int64_t, 2>({batch_index_t.dim_size(0), 3});
      for (int i = 0; i < batch_index_t.dim_size(0); ++i) {
        const int64_t batch_key = batch_index(i, 0);
        if (available_tensors_.find(batch_key) == available_tensors_.end()) {
          missing_tensors.emplace(batch_key);
        }
      }
      if (missing_tensors.empty()) {
        return OutputBatch(context, done);
      }
      if (!available_batches_
               .emplace(batch_key, Batch{missing_tensors, context, done})
               .second) {
        return errors::InvalidArgument(
            "Batch key with valid batch used twice.");
      }
      for (const int64_t i : missing_tensors) {
        if (!desired_tensor_to_batch_map_.emplace(i, batch_key).second) {
          return errors::InvalidArgument(
              "Missing tensor wanted by more than one batch.");
        }
      }
    } else {
      // If we don't have a valid input tensor we can output an empty tensor and
      // call our done closure.
      TensorShape output_shape(grad_t.shape());
      output_shape.set_dim(0, 0);
      Tensor* output = nullptr;
      TF_RETURN_IF_ERROR(context->allocate_output(0, output_shape, &output));
      done();
    }

    // Search to see whether our tensor is desired by any existing batch.
    auto desire_it = desired_tensor_to_batch_map_.find(batch_key);
    if (desire_it != desired_tensor_to_batch_map_.end()) {
      // Mark our tensor as no longer missing.
      auto batch_it = available_batches_.find(desire_it->second);
      desired_tensor_to_batch_map_.erase(desire_it);
      if (batch_it == available_batches_.end()) {
        return errors::InvalidArgument("Batch no longer exists.");
      }
      batch_it->second.missing_tensors.erase(batch_key);
      // If all tensors are available we should concatenate them and dispatch
      // the batch.
      if (batch_it->second.missing_tensors.empty()) {
        TF_RETURN_IF_ERROR(
            OutputBatch(batch_it->second.context, batch_it->second.done));
        available_batches_.erase(batch_it);
      }
    }
    return absl::OkStatus();
  }

 private:
  mutex mu_;

  // Represents a still-incomplete batch of tensors. When all tensors become
  // available they will be concatenated in the right order and sent through the
  // context.
  struct Batch {
    // Batch keys for tensors which are still missing from this batch. When this
    // is empty the Tensors can be concatenated and forwarded.
    std::unordered_set<int64_t> missing_tensors;

    // Context and callback for the session responsible for finishing this
    // batch.
    OpKernelContext* context;
    AsyncOpKernel::DoneCallback done;
  };

  // Map from batch key of the session which will output the batched gradients
  // to still-incomplete batches.
  std::unordered_map<int64_t, Batch> available_batches_;

  // Map from batch key to tensors which are waiting for their batches to be
  // available.
  std::unordered_map<int64_t, Tensor> available_tensors_;

  // Map from batch key of a tensor which is not yet available to the batch key
  // of the batch to which it belongs.
  std::unordered_map<int64_t, int64_t> desired_tensor_to_batch_map_;
};

class UnbatchGradKernel : public AsyncOpKernel {
 public:
  explicit UnbatchGradKernel(OpKernelConstruction* c) : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("container", &container_));
    OP_REQUIRES_OK(c, c->GetAttr("shared_name", &shared_name_));
    // If shared_name is not supplied, use name instead (prevent collisions by
    // default).
    if (shared_name_.empty()) {
      shared_name_ = name();
    }
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) final {
    UnbatchGradResource* ubr;
    std::function<absl::Status(UnbatchGradResource**)> creator =
        [](UnbatchGradResource** r) {
          *r = new UnbatchGradResource();
          return absl::OkStatus();
        };
    OP_REQUIRES_OK_ASYNC(c,
                         c->resource_manager()->LookupOrCreate(
                             container_, shared_name_, &ubr, creator),
                         done);
    absl::Status status = ubr->Compute(c, done);
    ubr->Unref();
    OP_REQUIRES_OK_ASYNC(c, status, done);
    // Assume ubr calls done, so nothing to do here.
  }

 private:
  string container_;
  string shared_name_;
};
REGISTER_KERNEL_BUILDER(Name("UnbatchGrad").Device(DEVICE_CPU),
                        UnbatchGradKernel);

}  // namespace tensorflow
