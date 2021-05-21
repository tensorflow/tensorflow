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

#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_resource_base.h"
#include "tensorflow/core/kernels/batching_util/concat_split_util.h"
#include "tensorflow/core/kernels/batching_util/periodic_function.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/numbers.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace {
// Op attributes.
constexpr char kEnableAdaptiveSchedulerAttr[] = "_enable_adaptive_scheduler";
constexpr char kMinInflightBatchesAttr[] = "_min_inflight_batches";
constexpr char kInitialInflightBatchesAttr[] = "_initial_inflight_batches";
constexpr char kMaxInflightBatchesAttr[] = "_max_inflight_batches";
constexpr char kBatchesToAverageOverAttr[] = "_batches_to_average_over";

// Per-model inflight batches parameters.
constexpr int64 kMinInflightBatches = 16;
constexpr int64 kInitialInflightBatches = 16;
constexpr int64 kBatchesToAverageOver = 10;
constexpr int64 kMaxInflightBatches = 64;

// The max number of threads in the per-process thread pool, shared by
// all executions of batch-op.
constexpr int64 kBatchThreadPoolSize = 128;
}  // namespace

auto* batch_op_split_usage = monitoring::Gauge<string, 1>::New(
    "/tensorflow/serving/batching/enable_large_batch_splitting",
    "Tracks the usage of attribute `enable_large_batch_splitting` for "
    "BatchFunction kernel in a saved model.",
    "model_name");

void RecordBatchSplitUsage(
    absl::optional<bool> maybe_enable_large_batch_splitting,
    const string& model_name) {
  if (maybe_enable_large_batch_splitting.has_value()) {
    if (maybe_enable_large_batch_splitting.value()) {
      batch_op_split_usage->GetCell(model_name)->Set("true");
    } else {
      batch_op_split_usage->GetCell(model_name)->Set("false");
    }
  } else {
    batch_op_split_usage->GetCell(model_name)->Set("unset");
  }
}

void RecordBatchParamNumBatchThreads(int64 num_batch_threads,
                                     const string& model_name) {
  static auto* cell = monitoring::Gauge<int64, 1>::New(
      "/tensorflow/serving/batching/num_batch_threads",
      "Tracks the number of batch threads of a model.", "model_name");
  cell->GetCell(model_name)->Set(num_batch_threads);
}

const string& GetModelName(OpKernelContext* ctx) {
  static string* kModelNameUnset = new string("model_name_unset");
  if (!ctx->session_metadata()) return *kModelNameUnset;
  if (ctx->session_metadata()->name().empty()) return *kModelNameUnset;
  return ctx->session_metadata()->name();
}

using ::tensorflow::concat_split_util::Concat;
using ::tensorflow::concat_split_util::Split;

static thread::ThreadPool* GetOrCreateBatchThreadsPool(
    const string& thread_name, int num_batch_threads) {
  static thread::ThreadPool* pool =
      new thread::ThreadPool(Env::Default(), thread_name, num_batch_threads);
  return pool;
}

// A class encapsulating the state and logic for batching tensors.
class BatchResource : public serving::BatchResourceBase {
 public:
  static Status Create(int32 num_batch_threads, int32 max_execution_batch_size,
                       int32 batch_timeout_micros, int32 max_enqueued_batches,
                       const std::vector<int32>& allowed_batch_sizes,
                       FunctionLibraryRuntime::Handle fhandle,
                       FunctionLibraryRuntime* flib,
                       bool enable_large_batch_splitting,
                       std::unique_ptr<BatchResource>* resource) {
    BatcherT::Options batcher_options;
    batcher_options.num_batch_threads = num_batch_threads;
    std::shared_ptr<BatcherT> batcher;
    TF_RETURN_IF_ERROR(BatcherT::Create(batcher_options, &batcher));

    resource->reset(new BatchResource(
        fhandle, flib, std::move(batcher),
        GetBatcherQueueOptions(num_batch_threads, max_execution_batch_size,
                               batch_timeout_micros, max_enqueued_batches,
                               allowed_batch_sizes,
                               enable_large_batch_splitting),
        allowed_batch_sizes));
    return Status::OK();
  }

  static Status Create(
      AdaptiveBatcherT::Options adaptive_shared_batch_scheduler_options,
      int32 max_batch_size, int32 batch_timeout_micros,
      int32 max_enqueued_batches, const std::vector<int32>& allowed_batch_sizes,
      FunctionLibraryRuntime::Handle fhandle, FunctionLibraryRuntime* flib,
      std::unique_ptr<BatchResource>* resource) {
    std::shared_ptr<AdaptiveBatcherT> batcher;
    TF_RETURN_IF_ERROR(AdaptiveBatcherT::Create(
        adaptive_shared_batch_scheduler_options, &batcher));

    resource->reset(new BatchResource(
        fhandle, flib, std::move(batcher),
        GetAdaptiveBatcherQueueOptions(
            max_batch_size, batch_timeout_micros, max_enqueued_batches,
            true /* enable large batch split */, allowed_batch_sizes),
        allowed_batch_sizes));
    return Status::OK();
  }

  string DebugString() const final { return "BatchResource"; }

 private:
  BatchResource(FunctionLibraryRuntime::Handle fhandle,
                FunctionLibraryRuntime* flib, std::shared_ptr<BatcherT> batcher,
                const BatcherT::QueueOptions& batcher_queue_options,
                std::vector<int32> allowed_batch_sizes)
      : BatchResourceBase(
            /*has_process_batch_function=*/fhandle != kInvalidHandle,
            std::move(batcher), batcher_queue_options,
            std::move(allowed_batch_sizes)),
        fhandle_(fhandle),
        flib_(flib) {}

  BatchResource(FunctionLibraryRuntime::Handle fhandle,
                FunctionLibraryRuntime* flib,
                std::shared_ptr<AdaptiveBatcherT> batcher,
                const AdaptiveBatcherT::QueueOptions& batcher_queue_options,
                std::vector<int32> allowed_batch_sizes)
      : BatchResourceBase(
            /*has_process_batch_function=*/fhandle != kInvalidHandle,
            std::move(batcher), batcher_queue_options,
            std::move(allowed_batch_sizes)),
        fhandle_(fhandle),
        flib_(flib) {}

  void ProcessFuncBatchImpl(
      const BatchTask& last_task, absl::Span<const Tensor> inputs,
      std::vector<Tensor>* combined_outputs,
      std::function<void(const Status&)> done) const override {
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

    flib_->Run(opts, fhandle_, inputs, combined_outputs,
               [&](const Status& run_status) {
                 done(run_status);
                 done_notif.Notify();
               });
    // By waiting for the notification we are ensuring that this thread isn't
    // used for processing other batches, which gives the batches time to
    // coalesce upstream. So overall the number of batches going through the
    // devices goes down, improving latency and throughput in most cases.
    done_notif.WaitForNotification();
  }

  FunctionLibraryRuntime::Handle fhandle_;
  FunctionLibraryRuntime* flib_;
};

class BatchFunctionKernel : public AsyncOpKernel {
 public:
  explicit BatchFunctionKernel(OpKernelConstruction* c) : AsyncOpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("container", &container_));
    OP_REQUIRES_OK(c, c->GetAttr("shared_name", &shared_name_));
    OP_REQUIRES_OK(c, c->GetAttr("batching_queue", &batcher_queue_));
    OP_REQUIRES_OK(c, c->GetAttr("num_batch_threads", &num_batch_threads_));
    OP_REQUIRES_OK(c, c->GetAttr("max_batch_size", &max_batch_size_));
    OP_REQUIRES_OK(c,
                   c->GetAttr("batch_timeout_micros", &batch_timeout_micros_));
    OP_REQUIRES_OK(c,
                   c->GetAttr("max_enqueued_batches", &max_enqueued_batches_));
    OP_REQUIRES_OK(c, c->GetAttr("allowed_batch_sizes", &allowed_batch_sizes_));

    OP_REQUIRES_OK(c, c->GetAttr("f", &func_));
    flib_ = c->function_library();

    if (c->HasAttr("enable_large_batch_splitting")) {
      OP_REQUIRES_OK(c, c->GetAttr("enable_large_batch_splitting",
                                   &enable_large_batch_splitting_));
      has_attribute_enable_large_batch_splitting_ = true;
    } else {
      enable_large_batch_splitting_ = false;
      has_attribute_enable_large_batch_splitting_ = false;
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
      // Note name() is unique per session (from session metadata).
      batcher_queue_ = name() + "/" + shared_name_ + batcher_queue_;
    }

    if (shared_name_.empty()) {
      // If shared_name is not supplied, use name instead (prevent collisions by
      // default).
      shared_name_ = name();
    }

    OP_REQUIRES_OK(c, ValidateAllowedBatchSizes());
  }

  bool IsExpensive() override { return false; }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) final {
    RecordBatchSplitUsage(
        has_attribute_enable_large_batch_splitting_
            ? absl::make_optional(enable_large_batch_splitting_)
            : absl::nullopt,
        GetModelName(c));
    // TODO(b/173255290): Add num_batch_threads_ parameter to TFRT batch kernel.
    RecordBatchParamNumBatchThreads(num_batch_threads_, GetModelName(c));

    std::function<Status(BatchResource**)> creator;

    FunctionLibraryRuntime::Handle handle;
    OP_REQUIRES_OK_ASYNC(c, GetOrCreateFunctionHandle(c, &handle), done);

    if (adaptive_batch_scheduler_options_ != absl::nullopt) {
      creator = [this, handle](BatchResource** r) {
        serving::AdaptiveSharedBatchScheduler<
            serving::BatchResourceBase::BatchTask>::Options
            adaptive_shared_batch_scheduler_options;
        adaptive_shared_batch_scheduler_options.thread_pool_name =
            "adaptive_batch_threads";
        adaptive_shared_batch_scheduler_options.num_batch_threads =
            adaptive_batch_scheduler_options_->max_in_flight_batches_limit;
        adaptive_shared_batch_scheduler_options.thread_pool =
            GetOrCreateBatchThreadsPool(std::string("adaptive_batch_threads"),
                                        kBatchThreadPoolSize);
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
            adaptive_batch_scheduler_options_->min_in_flight_batches_limit;
        adaptive_shared_batch_scheduler_options
            .initial_in_flight_batches_limit =
            adaptive_batch_scheduler_options_->initial_in_flight_batches_limit;
        adaptive_shared_batch_scheduler_options.batches_to_average_over =
            adaptive_batch_scheduler_options_->batches_to_average_over;
        adaptive_shared_batch_scheduler_options.fifo_scheduling = true;
        std::unique_ptr<BatchResource> new_resource;
        TF_RETURN_IF_ERROR(BatchResource::Create(
            adaptive_shared_batch_scheduler_options, max_batch_size_,
            batch_timeout_micros_, max_enqueued_batches_, allowed_batch_sizes_,
            handle, flib_, &new_resource));
        *r = new_resource.release();
        return Status::OK();
      };
    } else {
      creator = [this, handle](BatchResource** r) {
        std::unique_ptr<BatchResource> new_resource;
        TF_RETURN_IF_ERROR(BatchResource::Create(
            num_batch_threads_, max_batch_size_, batch_timeout_micros_,
            max_enqueued_batches_, allowed_batch_sizes_, handle, flib_,
            enable_large_batch_splitting_, &new_resource));
        *r = new_resource.release();
        return Status::OK();
      };
    }

    BatchResource* br;
    OP_REQUIRES_OK_ASYNC(c,
                         c->resource_manager()->LookupOrCreate(
                             container_, shared_name_, &br, creator),
                         done);
    const Status status =
        br->RegisterInput(random::New64(), c, batcher_queue_, done);
    br->Unref();
    OP_REQUIRES_OK_ASYNC(c, status, done);
    // Assume br calls done, so nothing to do here.
  }

  Status InstantiateFunction(OpKernelContext* c,
                             FunctionLibraryRuntime::Handle* handle) const {
    // TODO(b/173748062): Merge this instantiation logic with PartitionedCall.
    if (!flib_) {
      return errors::Internal("No function library");
    }

    FunctionLibraryRuntime::InstantiateOptions opts;
    opts.target = flib_->device() == nullptr ? "" : flib_->device()->name();
    opts.is_multi_device_function = true;
    const ConfigProto* config = flib_->config_proto();
    if (config) {
      opts.config_proto = *config;
    }

    Device* cpu_device;
    TF_RETURN_IF_ERROR(flib_->device_mgr()->LookupDevice("CPU:0", &cpu_device));

    const FunctionDef* fdef =
        flib_->GetFunctionLibraryDefinition()->Find(func_.name());
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
    return flib_->Instantiate(func_.name(), AttrSlice(&func_.attr()), opts,
                              handle);
  }

  Status GetOrCreateFunctionHandle(OpKernelContext* c,
                                   FunctionLibraryRuntime::Handle* handle) {
    mutex_lock ml(mu_);
    if (!fhandle_) {
      TF_RETURN_IF_ERROR(InstantiateFunction(c, handle));
      fhandle_ = *handle;
    } else {
      *handle = fhandle_.value();
    }
    return Status::OK();
  }

  // Validates 'allowed_batch_sizes_'. The entries must increase monotonically.
  // If large batch split is not enabled, the last one must equal
  // `max_batch_size_`. otherwise the last element must be smaller than or equal
  // to `max_batch_size_`.
  Status ValidateAllowedBatchSizes() const {
    if (allowed_batch_sizes_.empty()) {
      return Status::OK();
    }
    int32 last_size = 0;
    for (size_t i = 0; i < allowed_batch_sizes_.size(); ++i) {
      const int32 size = allowed_batch_sizes_.at(i);
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
    return Status::OK();
  }

 private:
  // Initialize vars by reading from op-kernel-construction.
  // Vars
  // - enable_adaptive_batch_threads_
  //   true if value of attribute `kEnableAdaptiveSchedulerAttr` is true, or
  //   if `num_batch_threads` is not positive.
  // - adaptive_batch_scheduler_options_
  //   Read from corresponding attributes as long as they are set.
  void SetAdaptiveBatchSchedulerOptions(OpKernelConstruction* c,
                                        int32 num_batch_threads) {
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

    adaptive_batch_scheduler_options_ = options;
  }

  string container_;
  string shared_name_;
  string batcher_queue_;
  int32 num_batch_threads_;
  int32 max_batch_size_;
  int32 batch_timeout_micros_;
  int32 max_enqueued_batches_;
  std::vector<int32> allowed_batch_sizes_;
  NameAttrList func_;
  absl::optional<FunctionLibraryRuntime::Handle> fhandle_ TF_GUARDED_BY(mu_);
  FunctionLibraryRuntime* flib_;
  bool enable_large_batch_splitting_;
  bool has_attribute_enable_large_batch_splitting_;
  bool enable_adaptive_batch_threads_ = false;
  mutex mu_;

  // Parameters for adaptive batch scheduler only.
  // Note 'num_batch_threads_' above is shared by two implementations of batch
  // scheduler.
  struct AdaptiveBatchSchedulerOptions {
    int32 min_in_flight_batches_limit = kMinInflightBatches;
    int32 initial_in_flight_batches_limit = kInitialInflightBatches;
    int32 max_in_flight_batches_limit = kMaxInflightBatches;
    int32 batches_to_average_over = kBatchesToAverageOver;
  };
  absl::optional<AdaptiveBatchSchedulerOptions>
      adaptive_batch_scheduler_options_ = absl::nullopt;
};

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
    std::function<Status(BatchResource**)> creator = [this](BatchResource** r) {
      std::unique_ptr<BatchResource> new_resource;
      TF_RETURN_IF_ERROR(BatchResource::Create(
          num_batch_threads_, max_batch_size_, batch_timeout_micros_,
          max_enqueued_batches_, allowed_batch_sizes_, kInvalidHandle,
          /*flib=*/nullptr, false, &new_resource));
      *r = new_resource.release();
      return Status::OK();
    };
    OP_REQUIRES_OK_ASYNC(c,
                         c->resource_manager()->LookupOrCreate(
                             container_, shared_name_, &br, creator),
                         done);
    const Status status =
        br->RegisterInput(random::New64(), c, batcher_queue_, done);
    br->Unref();
    OP_REQUIRES_OK_ASYNC(c, status, done);
    // Assume br calls done, so nothing to do here.
  }

  // Validates 'allowed_batch_sizes_'. The entries must increase
  // monotonically, and the last one must equal 'max_batch_size_'.
  Status ValidateAllowedBatchSizes() const {
    if (allowed_batch_sizes_.empty()) {
      return Status::OK();
    }
    int32 last_size = 0;
    for (size_t i = 0; i < allowed_batch_sizes_.size(); ++i) {
      const int32 size = allowed_batch_sizes_.at(i);
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
    return Status::OK();
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
  explicit UnbatchResource(int32 timeout_micros)
      : timeout_micros_(timeout_micros),
        timeout_enforcer_(new serving::PeriodicFunction(
            [this] { EnforceTimeout(); }, 1000 /* 1 ms */)) {}

  ~UnbatchResource() override {
    // Tear down 'timeout_enforcer_' first, since it accesses other state in
    // this class.
    timeout_enforcer_ = nullptr;
  }

  string DebugString() const final { return "UnbatchResource"; }

  Status Compute(OpKernelContext* context, AsyncOpKernel::DoneCallback done) {
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

    const int64 batch_key = context->input(2).scalar<int64>()();
    const bool nonempty_input = batch_index_t.dim_size(0) > 0;

    // If we have a non-empty tensor, slice it up.
    // (It is important to do this outside of the critical section below.)
    // The following variables are populated iff 'nonempty_input==true'.
    std::vector<int64> sizes;
    std::vector<int64> batch_keys;
    std::vector<Tensor> split_inputs;
    if (nonempty_input) {
      auto batch_indices =
          batch_index_t.shaped<int64, 2>({batch_index_t.dim_size(0), 3});
      for (int i = 0; i < batch_index_t.dim_size(0); ++i) {
        sizes.push_back(batch_indices(i, 2) - batch_indices(i, 1));
        batch_keys.push_back(batch_indices(i, 0));
      }

      TF_RETURN_IF_ERROR(Split(context, data_t, sizes, &split_inputs));
    }

    // Critical section.
    std::vector<AsyncOpKernel::DoneCallback> done_callbacks_to_call;
    Status status = [&]() -> Status {
      mutex_lock ml(mu_);

      // Check to see whether the tensor we want is already ready.
      auto tensor_it = waiting_tensors_.find(batch_key);
      if (tensor_it != waiting_tensors_.end()) {
        context->set_output(0, tensor_it->second.tensor);
        waiting_tensors_.erase(tensor_it);
        done_callbacks_to_call.push_back(done);
        return Status::OK();
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

      return Status::OK();
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
  std::unordered_map<int64, WaitingTensor> waiting_tensors_ TF_GUARDED_BY(mu_);
  std::unordered_map<int64, WaitingCallback> waiting_callbacks_
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
    std::function<Status(UnbatchResource**)> creator =
        [this](UnbatchResource** r) {
          *r = new UnbatchResource(timeout_micros_);
          return Status::OK();
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
  Status OutputBatch(OpKernelContext* context,
                     const AsyncOpKernel::DoneCallback& done)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    const Tensor& batch_index_t = context->input(1);
    auto batch_index =
        batch_index_t.shaped<int64, 2>({batch_index_t.dim_size(0), 3});
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
    return Status::OK();
  }

  // Ingests data from one invocation of the op.
  Status Compute(OpKernelContext* context,
                 const AsyncOpKernel::DoneCallback& done) {
    const Tensor& data_t = context->input(0);
    const Tensor& batch_index_t = context->input(1);
    const Tensor& grad_t = context->input(2);

    mutex_lock ml(mu_);

    const int64 batch_key = context->input(3).scalar<int64>()();
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
      std::unordered_set<int64> missing_tensors;
      const auto batch_index =
          batch_index_t.shaped<int64, 2>({batch_index_t.dim_size(0), 3});
      for (int i = 0; i < batch_index_t.dim_size(0); ++i) {
        const int64 batch_key = batch_index(i, 0);
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
      for (const int64 i : missing_tensors) {
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
    return Status::OK();
  }

 private:
  mutex mu_;

  // Represents a still-incomplete batch of tensors. When all tensors become
  // available they will be concatenated in the right order and sent through the
  // context.
  struct Batch {
    // Batch keys for tensors which are still missing from this batch. When this
    // is empty the Tensors can be concatenated and forwarded.
    std::unordered_set<int64> missing_tensors;

    // Context and callback for the session responsible for finishing this
    // batch.
    OpKernelContext* context;
    AsyncOpKernel::DoneCallback done;
  };

  // Map from batch key of the session which will output the batched gradients
  // to still-incomplete batches.
  std::unordered_map<int64, Batch> available_batches_;

  // Map from batch key to tensors which are waiting for their batches to be
  // available.
  std::unordered_map<int64, Tensor> available_tensors_;

  // Map from batch key of a tensor which is not yet available to the batch key
  // of the batch to which it belongs.
  std::unordered_map<int64, int64> desired_tensor_to_batch_map_;
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
    std::function<Status(UnbatchGradResource**)> creator =
        [](UnbatchGradResource** r) {
          *r = new UnbatchGradResource();
          return Status::OK();
        };
    OP_REQUIRES_OK_ASYNC(c,
                         c->resource_manager()->LookupOrCreate(
                             container_, shared_name_, &ubr, creator),
                         done);
    Status status = ubr->Compute(c, done);
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
