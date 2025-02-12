/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/executor.h"

#include <algorithm>
#include <atomic>
#include <memory>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "tensorflow/core/activity_watcher/activity.h"
#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/entry.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/graph_view.h"
#include "tensorflow/core/common_runtime/immutable_executor_state.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/common_runtime/propagator_state.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/common_runtime/simple_propagator_state.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_segment.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_node_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/manual_constructor.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/annotated_traceme.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/context_types.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/managed_stack_trace.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"
#include "tsl/platform/tracing.h"

namespace tensorflow {

namespace {

// 1-D, 0 element tensor.
static const Tensor* const kEmptyTensor = new Tensor;

// Helper routines for collecting step stats.
namespace nodestats {
inline int64_t NowInNsec() { return EnvTime::NowNanos(); }

void SetScheduled(NodeExecStatsInterface* stats, int64_t micros) {
  if (!stats) return;
  stats->SetScheduled(micros * EnvTime::kMicrosToNanos);
}

void SetAllStart(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordExecutorStarted();
}

void SetOpStart(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordComputeStarted();
}

void SetOpEnd(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordComputeEnded();
}

void SetAllEnd(NodeExecStatsInterface* stats) {
  if (!stats) return;
  stats->RecordExecutorEnded();
}

void SetOutput(NodeExecStatsInterface* stats, int slot, const Tensor* v) {
  if (!stats) return;
  stats->SetOutput(slot, v);
}

void SetMemory(NodeExecStatsInterface* stats, OpKernelContext* ctx) {
  if (!stats) return;
  stats->SetMemory(ctx);
}

}  // namespace nodestats

// Time the execution of kernels (in CPU cycles).  Used to dynamically identify
// inexpensive kernels which can be dispatched inline.
struct KernelTimer {
  uint64 start_cycles = profile_utils::CpuUtils::GetCurrentClockCycle();

  uint64 ElapsedCycles() {
    return profile_utils::CpuUtils::GetCurrentClockCycle() - start_cycles;
  }
};

// TODO(b/152925936): Re-evaluate these constants with current usage patterns.
typedef absl::InlinedVector<TensorValue, 4UL> TensorValueVec;
typedef absl::InlinedVector<AllocatorAttributes, 4UL> AllocatorAttributeVec;

class ExecutorImpl : public Executor {
 public:
  explicit ExecutorImpl(const LocalExecutorParams& p) : immutable_state_(p) {}

  absl::Status Initialize(const Graph& graph) {
    TF_RETURN_IF_ERROR(immutable_state_.Initialize(graph));
    kernel_stats_.Initialize(immutable_state_.graph_view());
    return absl::OkStatus();
  }

 private:
  void RunAsyncInternal(const Args& args, DoneCallback done) override;

  template <class PropagatorStateType>
  friend class ExecutorState;

  // Stores execution time information about the kernels in an executor's graph.
  class KernelStats {
   public:
    KernelStats() = default;

    void Initialize(const GraphView& gview) {
      is_expensive_.resize(gview.num_nodes());
      cost_estimates_ =
          std::make_unique<std::atomic_uint_fast64_t[]>(gview.num_nodes());
      for (int32_t i = 0; i < gview.num_nodes(); ++i) {
        if (gview.node(i)) {
          is_expensive_[i] =
              gview.node(i)->kernel && gview.node(i)->kernel->IsExpensive();
          cost_estimates_[i] = kInitialCostEstimateCycles;
        }
      }
    }

    // Returns true iff the given node is considered "expensive". The
    // executor uses this flag to optimize graph execution, for example
    // by "inlining" inexpensive kernels.
    bool IsExpensive(const NodeItem& node) const {
      return is_expensive_[node.node_id] &&
             (cost_estimates_[node.node_id].load(std::memory_order_relaxed) >
              kOpIsExpensiveThresholdCycles);
    }

    // Returns the value of kernel->IsExpensive().
    bool HasExpensiveMarker(const NodeItem& node) const {
      return is_expensive_[node.node_id];
    }

    // Updates the dynamic cost estimate, which is used to determine whether the
    // given node is expensive. The new cost estimate is a weighted average of
    // the old cost estimate and the latest cost. We only update cost estimates
    // for kernels for which IsExpensive() return true.
    void UpdateCostEstimate(const NodeItem& node, uint64 elapsed_cycles) {
      // N.B. Updates to `cost_estimate` are atomic but unlocked.  Simultaneous
      // updates may result in one or more updates being ignored.  This does not
      // affect correctness but may slow down the update frequency.
      std::atomic_uint_fast64_t& cost_estimate = cost_estimates_[node.node_id];
      auto prev_estimate = cost_estimate.load(std::memory_order_relaxed);

      uint64 new_estimate =
          ((kCostDecay - 1) * prev_estimate + elapsed_cycles) / kCostDecay;

      cost_estimate.store(new_estimate, std::memory_order_relaxed);
    }

   private:
    // Initial time (in CPU cycles) we expect an operation to take.  Used to
    // determine whether an operation should be place in a threadpool.
    // Operations start out "expensive".
    static constexpr uint64 kInitialCostEstimateCycles = 100 * 1000 * 1000;
    static constexpr uint64 kOpIsExpensiveThresholdCycles = 8000;
    static constexpr uint64 kCostDecay = 10;

    std::vector<bool> is_expensive_;
    // std::unique_ptr<std::atomic<bool>[]> is_expensive_;
    std::unique_ptr<std::atomic_uint_fast64_t[]> cost_estimates_;
  };

  ImmutableExecutorState immutable_state_;
  KernelStats kernel_stats_;

  ExecutorImpl(const ExecutorImpl&) = delete;
  void operator=(const ExecutorImpl&) = delete;
};

// The state associated with one invocation of ExecutorImpl::Run.
//
// ExecutorState dispatches nodes when they become ready, and delegates to an
// instance of `PropagatorStateType` to keep track of how many predecessors of a
// are still pending.
//
// The template argument `class PropagatorStateType` must define the following
// public members:
// * A type `TaggedNode`, representing a node to be processed, with public
//   members:
//   * `const NodeItem& get_node_item() const`
//   * `bool get_is_dead() const`
// * A type `TaggedNodeReadyQueue`, representing a queue of nodes to be
//   processed, with public members (having the same meanings as in an
//   `std::vector<TaggedNode>`):
//   * `void push_back(const TaggedNode& node)`
//   * `TaggedNode front() const`
//   * `void pop_front()`
//   * `bool empty() const`
// * A type `TaggedNodeSeq`, representing a list of nodes to be scheduled, with
//   public members (having the same meanings as in an
//   `std::vector<TaggedNode>`):
//   * `size_t size() const`
//   * `bool empty() const`
//   * `void clear()`
//   * `const_iterator begin() const`
//   * `const_iterator end() const`
// * A public constructor, `PropagatorStateType(const ImmutableExecutorState&
//   immutable_state, int64 step_id)`.
// * The following public methods:
//   * `void ActivateRoots(gtl::ArraySlice<const NodeItem*> roots,
//     TaggedNodeSeq* ready)`, which creates `TaggedNode` instances for the
//     nodes in `roots` and adds them to `*ready`
//   * `void PropagateOutputs(const TaggedNode& tagged_node, EntryVector*
//     outputs, TaggedNodeSeq* ready)`, which propagates `outputs` from the
//     given `tagged_node` to the destinations of its output edges, and adds
//     any newly runnable nodes to `*ready`
//   * `Entry* GetInputTensors(const TaggedNode& tagged_node) const`, which
//     returns a pointer to the input tensors for the given `tagged_node`
//   * `FrameAndIter GetFrameAndIter(const TaggedNode& tagged_node) const`,
//     which creates a `FrameAndIter` for the given `tagged_node`
//   * `void DumpState()`, which dumps the dynamic state of the executing graph
//   * `void MaybeMarkStarted(const TaggedNode& tagged_node)`, which records
//     that a node has started
//   * `void MaybeMarkCompleted(const TaggedNode& tagged_node)`, which records
//     that a node has completed
//
// See `PropagatorState` in "./propagator_state.h" for an example of a type that
// can be used to instantiate `PropagatorStateType`.
template <class PropagatorStateType>
class ExecutorState {
 public:
  ExecutorState(const Executor::Args& args,
                const ImmutableExecutorState& immutable_state_,
                ExecutorImpl::KernelStats* kernel_stats_);
  ~ExecutorState();

  void RunAsync(Executor::DoneCallback done);

 private:
  // Use `TaggedNode` types defined by `PropagatorStateType`.
  typedef typename PropagatorStateType::TaggedNode TaggedNode;
  typedef
      typename PropagatorStateType::TaggedNodeReadyQueue TaggedNodeReadyQueue;
  typedef typename PropagatorStateType::TaggedNodeSeq TaggedNodeSeq;

  struct AsyncState;

  // Process a ready node in current thread.
  void Process(const TaggedNode& node, int64_t scheduled_nsec);

  void ProcessInline(TaggedNodeReadyQueue* inline_ready,
                     int64_t scheduled_nsec);

  absl::Status ProcessSync(const NodeItem& item,
                           OpKernelContext::Params* params,
                           EntryVector* outputs, NodeExecStatsInterface* stats);
  void ProcessAsync(const NodeItem& item, const OpKernelContext::Params& params,
                    const TaggedNode& tagged_node, Entry* first_input,
                    NodeExecStatsInterface* stats,
                    activity_watcher::ActivityId activity_id);
  void ProcessNoop(NodeExecStatsInterface* stats);
  void ProcessConstTensor(const NodeItem& item, EntryVector* outputs,
                          NodeExecStatsInterface* stats);

  // Before invoking item->kernel, fills in its "inputs".
  absl::Status PrepareInputs(const NodeItem& item, Entry* first_input,
                             TensorValueVec* inputs,
                             AllocatorAttributeVec* input_alloc_attrs,
                             bool* is_input_dead);

  // After item->kernel computation is done, processes its outputs.
  absl::Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                              Entry* outputs, NodeExecStatsInterface* stats);

  // Called after each node finishes. Takes ownership of "stats". Returns true
  // if execution has completed.
  //
  // This method will clear `*ready` before returning.
  bool NodeDone(const absl::Status& s, TaggedNodeSeq* ready,
                NodeExecStatsInterface* stats,
                TaggedNodeReadyQueue* inline_ready);

  // Schedule all the expensive nodes in '*ready', and put all the inexpensive
  // nodes in 'ready' into 'inline_ready'.
  //
  // This method will clear `*ready` before returning.
  //
  // REQUIRES: `!ready->empty()`.
  void ScheduleReady(TaggedNodeSeq* ready, TaggedNodeReadyQueue* inline_ready);

  // A wrapper for runner_ to keep track of the pending queue length. Op
  // execution should dispatch work using this function instead of using runner_
  // directly.
  template <typename Closure>
  void RunTask(Closure&& c, int sample_rate = 0);

  // Clean up when this executor is done.
  void Finish();
  void ScheduleFinish();

  // Contains the device context assigned by the device at the beginning of a
  // step.
  DeviceContext* device_context_ = nullptr;

  const bool vlog_;  // true if VLOG_IS_ON(1). Used to check vlog cheaply.

  // true if LogMemory::IsEnabled(). Used to check memory enabled cheaply.
  const bool log_memory_;

  int64_t step_id_;
  int64_t trace_id_;  // for profiler.
  int64_t start_time_usecs_ = 0;
  // The deadline for the session to complete by. Empty if unspecified.
  absl::optional<absl::Time> deadline_;

  // Maximum number of kernels that can be scheduled inline. If lots of kernels
  // are ready at the same time, scheduling them in one thread can be very slow.
  // TODO(fishx): Make it configurable if necessary.
  static constexpr uint64 kInlineScheduleReadyThreshold = 500;

  // Not owned.
  RendezvousInterface* rendezvous_;
  CollectiveExecutor* collective_executor_ = nullptr;
  const ConfigProto* const session_config_;
  SessionState* session_state_;
  string session_handle_;
  const SessionMetadata* session_metadata_ = nullptr;
  TensorStore* tensor_store_;
  // Step-local container.
  ScopedStepContainer* step_container_;
  StepStatsCollectorInterface* const stats_collector_;
  const tsl::tracing::EventCollector* const event_collector_;
  Context context_;

  // QUESTION: Make it a checkpoint::TensorSliceReaderCacheWrapper
  // instead of a pointer?  (avoids having to delete).
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
  CallFrameInterface* call_frame_;
  const ImmutableExecutorState& immutable_state_;
  ExecutorImpl::KernelStats* const kernel_stats_;
  CancellationManager* cancellation_manager_;
  tsl::CoordinationServiceAgent* coordination_service_agent_;
  absl::optional<ManagedStackTrace> stack_trace_ = absl::nullopt;
  // If not null, use this device to schedule intra-op operation
  std::unique_ptr<DeviceBase> user_device_;
  Executor::Args::Runner runner_;
  bool sync_on_finish_;
  const bool run_all_kernels_inline_;

  PropagatorStateType propagator_;

  // Invoked when the execution finishes.
  Executor::DoneCallback done_cb_;

  std::atomic_int_fast32_t num_outstanding_ops_;

  // Available via OpKernelContext to every OpKernel invocation.
  mutex num_deferred_ops_mu_;
  int64_t num_deferred_ops_ TF_GUARDED_BY(num_deferred_ops_mu_) = 0;
  bool finish_when_deferred_ops_done_ TF_GUARDED_BY(num_deferred_ops_mu_) =
      false;

  mutex mu_;
  absl::Status status_ TF_GUARDED_BY(mu_);
};

template <class PropagatorStateType>
ExecutorState<PropagatorStateType>::ExecutorState(
    const Executor::Args& args, const ImmutableExecutorState& immutable_state,
    ExecutorImpl::KernelStats* kernel_stats)
    : vlog_(VLOG_IS_ON(1)),
      log_memory_(LogMemory::IsEnabled()),
      step_id_(args.step_id),
      trace_id_(args.function_trace_id ? *args.function_trace_id : step_id_),
      start_time_usecs_(args.start_time_usecs),
      deadline_(args.deadline),
      rendezvous_(args.rendezvous),
      collective_executor_(args.collective_executor),
      session_config_(args.session_config),
      session_state_(args.session_state),
      session_handle_(args.session_handle),
      session_metadata_(immutable_state.params().session_metadata),
      tensor_store_(args.tensor_store),
      step_container_(args.step_container),
      stats_collector_(args.stats_collector),
      event_collector_(tsl::tracing::GetEventCollector(
          tsl::tracing::EventCategory::kCompute)),
      context_(ContextKind::kThread),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame),
      immutable_state_(immutable_state),
      kernel_stats_(kernel_stats),
      cancellation_manager_(args.cancellation_manager),
      coordination_service_agent_(args.coordination_service_agent),
      stack_trace_(args.stack_trace),
      runner_(args.runner),
      sync_on_finish_(args.sync_on_finish),
      run_all_kernels_inline_(args.run_all_kernels_inline),
      propagator_(immutable_state, step_id_, vlog_),
      num_outstanding_ops_(0) {
  if (args.user_intra_op_threadpool != nullptr) {
    Device* device = immutable_state_.params().device;
    user_device_ = RenamedDevice::NewRenamedDevice(
        device->name(), device, false, false, args.user_intra_op_threadpool);
  }
}

template <class PropagatorStateType>
ExecutorState<PropagatorStateType>::~ExecutorState() {
  if (device_context_) {
    device_context_->Unref();
  }
  delete slice_reader_cache_;
}

template <class PropagatorStateType>
template <typename Closure>
void ExecutorState<PropagatorStateType>::RunTask(Closure&& c, int sample_rate) {
  // Align the atomic variables at 64 bytes to avoid false-sharing, assuming the
  // cacheline size is 64 bytes or smaller.
  alignas(64) static std::atomic<int64_t> num_enqueue_ops{0};
  alignas(64) static std::atomic<int64_t> num_dequeue_ops{0};

  auto n_enqueues = num_enqueue_ops.fetch_add(1, std::memory_order_relaxed);
  // Sample the queue length on at least every 16 enqueue operations. This
  // amortizes the cost of metric updates across 16 operations.
  if (n_enqueues % std::max(16, sample_rate) == 0) {
    auto n_dequeues = num_dequeue_ops.load(std::memory_order_relaxed);
    metrics::UpdateGraphPendingQueueLength(n_enqueues - n_dequeues);
  }

  // mutable is needed because std::forward<Closure> in the lambda body may move
  // the Closure `c`.
  runner_([c = std::forward<Closure>(c)]() mutable {
    num_dequeue_ops.fetch_add(1, std::memory_order_relaxed);
    std::forward<Closure>(c)();
  });
}

template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::RunAsync(Executor::DoneCallback done) {
  TaggedNodeSeq ready;

  // Ask the device to fill in the device context map.
  Device* device = immutable_state_.params().device;
  const absl::Status get_context_status =
      device->TryGetDeviceContext(&device_context_);
  if (!get_context_status.ok()) {
    delete this;
    done(get_context_status);
    return;
  }

  // Initialize the ready queue.
  ready.reserve(immutable_state_.root_nodes().size());
  propagator_.ActivateRoots(immutable_state_.root_nodes(), &ready);
  num_outstanding_ops_ = ready.size();
  if (ready.empty()) {
    delete this;
    done(absl::OkStatus());
  } else {
    done_cb_ = std::move(done);
    // Schedule to run all the ready ops in thread pool.
    ScheduleReady(&ready, nullptr);
  }
}

// State kept alive for executing an asynchronous node in another
// thread.  NOTE: We need to make a copy of p.input and p.input_alloc_attrs for
// asynchronous kernels because OpKernelContext methods like input_type(i) needs
// the param points to valid input type vector. It's not an issue for
// sync kernels because these vectors are kept on the stack.
template <class PropagatorStateType>
struct ExecutorState<PropagatorStateType>::AsyncState {
  AsyncState(const OpKernelContext::Params& p, const TaggedNode& _tagged_node,
             const NodeItem* _item, Entry* _first_input,
             NodeExecStatsInterface* _stats)
      : saved_inputs(p.inputs.begin(), p.inputs.end()),
        saved_input_alloc_attrs(p.input_alloc_attrs.begin(),
                                p.input_alloc_attrs.end()),
        params(p),
        tagged_node(_tagged_node),
        item(_item),
        first_input(_first_input),
        // ParamsButClearingEigenGPUDevice does equivalent of
        //   params.eigen_gpu_device = nullptr;
        ctx(ParamsButClearingEigenGPUDevice(&params), item->num_outputs),
        stats(_stats) {
    params.inputs = saved_inputs;
    params.input_alloc_attrs = saved_input_alloc_attrs;
  }

  TensorValueVec saved_inputs;
  AllocatorAttributeVec saved_input_alloc_attrs;
  OpKernelContext::Params params;
  TaggedNode tagged_node;
  const NodeItem* item;
  Entry* first_input;
  OpKernelContext ctx;
  NodeExecStatsInterface* stats;

 private:
  OpKernelContext::Params* ParamsButClearingEigenGPUDevice(
      OpKernelContext::Params* p) {
    // Ensure OpKernelContext constructor will make a new eigen GPU device if
    // necessary.
    p->eigen_gpu_device = nullptr;  // Force allocation
    return p;
  }
};

// Returns true if `item` might be traced by the given trace and event
// collectors. Returns false only if `item` definitely will not be traced.
bool MightTrace(const tsl::tracing::EventCollector* event_collector,
                bool is_expensive) {
  // Tracing will only be enabled if either `event_collector` is non null,
  // or `trace_collector` is non-null and enabled for this particular kernel.
  // Although `profiler::TraceMe`, `profiler::ScopedAnnotation`, and
  // `tsl::tracing::ScopedRegion` check subsets of these properties internally
  // in their constructors, the cost of passing the necessary arguments to them
  // can be significant, so we avoid constructing them in the common case (when
  // we know they will not be used).
  if (event_collector != nullptr) {
    return true;
  }

  if (tsl::profiler::ScopedAnnotation::IsEnabled()) return true;

  return tsl::profiler::TraceMe::Active(
      tsl::profiler::GetTFTraceMeLevel(is_expensive));
}

template <class PropagatorStateType>
absl::Status ExecutorState<PropagatorStateType>::ProcessSync(
    const NodeItem& item, OpKernelContext::Params* params, EntryVector* outputs,
    NodeExecStatsInterface* stats) {
  absl::Status s;
  OpKernelContext ctx(params, item.num_outputs);
  nodestats::SetOpStart(stats);

  OpKernel* op_kernel = item.kernel;
  Device* device = immutable_state_.params().device;
  const bool is_expensive = kernel_stats_->IsExpensive(item);

  if (TF_PREDICT_FALSE(MightTrace(event_collector_, is_expensive))) {
    tsl::tracing::ScopedRegion region(tsl::tracing::EventCategory::kCompute,
                                      op_kernel->name_view());
    profiler::AnnotatedTraceMe activity(
        [op_kernel, &ctx] {
          return op_kernel->TraceString(
              ctx, /*verbose=*/tsl::profiler::TfOpDetailsEnabled());
        },
        tsl::profiler::GetTFTraceMeLevel(is_expensive));
    device->Compute(op_kernel, &ctx);
  } else if (kernel_stats_->HasExpensiveMarker(item)) {
    KernelTimer timer;
    device->Compute(op_kernel, &ctx);
    // For expensive kernels, always update the cost estimate. For inexpensive
    // kernels, update the cost estimate with ~1/16 probability. This assumes
    // that the last 4 bits of the CPU cycle count is uniformly distributed.
    constexpr int kKernelExecutionTrackingInvocationSkipCount = 16;
    if (is_expensive ||
        timer.start_cycles % kKernelExecutionTrackingInvocationSkipCount == 0) {
      kernel_stats_->UpdateCostEstimate(item, timer.ElapsedCycles());
    }
  } else {
    device->Compute(op_kernel, &ctx);
  }
  nodestats::SetOpEnd(stats);
  if (outputs->size() < item.num_outputs) outputs->resize(item.num_outputs);
  s = ProcessOutputs(item, &ctx, outputs->data(), stats);
  nodestats::SetMemory(stats, &ctx);
  return s;
}

template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ProcessAsync(
    const NodeItem& item, const OpKernelContext::Params& params,
    const TaggedNode& tagged_node, Entry* first_input,
    NodeExecStatsInterface* stats, activity_watcher::ActivityId activity_id) {
  AsyncOpKernel* async_kernel = item.kernel->AsAsync();
  DCHECK(async_kernel != nullptr);
  AsyncState* state =
      new AsyncState(params, tagged_node, &item, first_input, stats);

  nodestats::SetOpStart(stats);

  {
    // Always trace async ops.
    profiler::AnnotatedTraceMe activity(
        [async_kernel, state] {
          return async_kernel->TraceString(
              state->ctx, /*verbose=*/tsl::profiler::TfOpDetailsEnabled());
        },
        tsl::profiler::GetTFTraceMeLevel(/*is_expensive=*/false));

    // Trace async op start.
    tsl::profiler::TraceMeProducer producer(
        [&] {
          return tsl::profiler::TraceMeEncode(
              "ExecutorState::ProcessAsync::Start",
              {{"name", async_kernel->name()},
               {"kernel_type", async_kernel->type_string()},
               {"step_id", step_id_}});
        },
        tsl::profiler::ContextType::kTfExecutor);

    auto done = [this, state, activity_id, ctx_id = producer.GetContextId()]() {
      // Trace async op done.
      tsl::profiler::TraceMeConsumer consumer(
          [&] {
            return profiler::TraceMeEncode(
                "ExecutorState::ProcessAsync::Done",
                {{"name", state->item->kernel->name()},
                 {"kernel_type", state->item->kernel->type_string()},
                 {"step_id", step_id_}});
          },
          tsl::profiler::ContextType::kTfExecutor, ctx_id);

      Device* device = immutable_state_.params().device;
      NodeExecStatsInterface* stats = state->stats;  // Shorthand
      Entry* first_input = state->first_input;       // Shorthand

      nodestats::SetOpEnd(stats);
      EntryVector outputs(state->item->num_outputs);
      absl::Status s =
          ProcessOutputs(*state->item, &state->ctx, outputs.data(), stats);
      nodestats::SetMemory(stats, &state->ctx);
      if (vlog_) {
        VLOG(2) << "Async kernel done: " << state->item->node_id << " step "
                << step_id_ << " "
                << SummarizeNodeDef(state->item->kernel->def())
                << (state->tagged_node.get_is_dead() ? " is dead" : "")
                << " device: " << device->name();
      }

      // Clears inputs.
      const int num_inputs = state->item->num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->ClearVal();
      }
      propagator_.MaybeMarkCompleted(state->tagged_node);
      activity_watcher::ActivityEnd(activity_id);
      TaggedNodeSeq ready;
      if (s.ok()) {
        propagator_.PropagateOutputs(state->tagged_node, &outputs, &ready);
      }
      outputs.clear();
      const bool completed = NodeDone(s, &ready, stats, nullptr);
      delete state;
      if (completed) ScheduleFinish();
    };

    immutable_state_.params().device->ComputeAsync(async_kernel, &state->ctx,
                                                   std::move(done));
  }
}

template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ProcessNoop(
    NodeExecStatsInterface* stats) {
  nodestats::SetOpStart(stats);
  nodestats::SetOpEnd(stats);
}

template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ProcessConstTensor(
    const NodeItem& item, EntryVector* outputs, NodeExecStatsInterface* stats) {
  nodestats::SetOpStart(stats);
  nodestats::SetOpEnd(stats);
  Entry& output = (*outputs)[0];
  output.state = Entry::State::HAS_CONST_TENSOR;
  output.const_tensor = item.const_tensor;
  output.alloc_attr = item.output_attrs()[0];
}

template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::Process(const TaggedNode& tagged_node,
                                                 int64_t scheduled_nsec) {
  tsl::profiler::TraceMe traceme("ExecutorState::Process Scheduled",
                                 tsl::profiler::TraceMeLevel::kVerbose);
  TaggedNodeReadyQueue inline_ready;
  inline_ready.push_back(tagged_node);
  return ProcessInline(&inline_ready, scheduled_nsec);
}

template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ProcessInline(
    TaggedNodeReadyQueue* inline_ready, int64_t scheduled_nsec) {
  WithContext wc(context_);
  auto ready = std::make_unique<TaggedNodeSeq>();

  // Parameters passed to OpKernel::Compute.
  auto inputs = std::make_unique<TensorValueVec>();

  AllocatorAttributeVec input_alloc_attrs;

  auto params = std::make_unique<OpKernelContext::Params>();

  params->step_id = step_id_;
  // Override device's threadpool if user provides an intra_op_threadpool
  Device* device = immutable_state_.params().device;
  if (user_device_) {
    params->device = user_device_.get();
  } else {
    params->device = device;
  }
  params->start_time_usecs = start_time_usecs_;
  params->deadline = deadline_;
  params->log_memory = log_memory_;
  params->rendezvous = rendezvous_;
  params->collective_executor = collective_executor_;
  params->session_config = session_config_;
  params->session_state = session_state_;
  params->session_handle = session_handle_;
  params->session_metadata = session_metadata_;
  params->tensor_store = tensor_store_;
  params->cancellation_manager = cancellation_manager_;
  params->coordination_service_agent = coordination_service_agent_;
  params->stack_trace = stack_trace_;
  params->call_frame = call_frame_;
  params->function_library = immutable_state_.params().function_library;
  params->resource_manager = device->resource_manager();
  params->step_container = step_container_;
  params->slice_reader_cache = slice_reader_cache_;
  params->runner = &runner_;
  params->run_all_kernels_inline = run_all_kernels_inline_;
  params->stats_collector = stats_collector_;
  params->inc_num_deferred_ops_function = [this]() {
    mutex_lock lock(num_deferred_ops_mu_);
    num_deferred_ops_++;
  };
  params->dec_num_deferred_ops_function = [this]() {
    bool finish_when_deferred_ops_done = false;
    {
      mutex_lock lock(num_deferred_ops_mu_);
      num_deferred_ops_--;
      if (num_deferred_ops_ == 0) {
        finish_when_deferred_ops_done = finish_when_deferred_ops_done_;
      }
    }
    // Invoke Finish if the graph processing has completed. Finish is always
    // called exactly once per ExecutorState, either here if there are any
    // deferred ops, or in ScheduleFinish if there aren't any deferred ops.
    if (finish_when_deferred_ops_done) Finish();
  };

  // Set the device_context for this device, if it exists.
  params->op_device_context = device_context_;

  absl::Status s;
  NodeExecStatsInterface* stats = nullptr;

  EntryVector outputs(1);

  bool completed = false;
  int64_t last_iter_num = -1;
  std::unique_ptr<tsl::profiler::TraceMeConsumer> iteration_scope;
  while (!inline_ready->empty()) {
    TaggedNode tagged_node = inline_ready->front();

    int64_t current_iter_num = tagged_node.get_iter_num();
    if (current_iter_num != last_iter_num) {
      iteration_scope = std::make_unique<tsl::profiler::TraceMeConsumer>(
          // From TraceMeProducer in DirectSession::RunInternal,
          // GraphMgr::ExecuteAsync, or FunctionLibraryRuntime::Run.
          [&] {
            // NOTE: This tracing uses the iteration number from the first
            // tagged node that executes during this call to `Process()`. In
            // principle, subsequent nodes could have different values of
            // `iter_num` that will not be traced.
            return profiler::TraceMeEncode(
                "ExecutorState::Process",
                {{"id", step_id_}, {"iter_num", tagged_node.get_iter_num()}});
          },
          tsl::profiler::ContextType::kTfExecutor, trace_id_,
          tsl::profiler::TraceMeLevel::kInfo);
      last_iter_num = current_iter_num;
    }
    inline_ready->pop_front();
    const NodeItem& item = tagged_node.get_node_item();
    const int id = item.node_id;

    propagator_.MaybeMarkStarted(tagged_node);
    const activity_watcher::ActivityId activity_id =
        activity_watcher::ActivityStart(
            [&]() {
              return std::make_unique<activity_watcher::Activity>(
                  "ExecutorState::Process",
                  activity_watcher::ActivityCategory::kMisc,
                  activity_watcher::Activity::Attributes{
                      {"node_name", item.kernel->def().name()},
                      {"op", item.kernel->def().op()},
                      {"iter_num", absl::StrCat(tagged_node.get_iter_num())},
                      {"step_id", absl::StrCat(params->step_id)},
                      {"node_id", absl::StrCat(id)},
                      {"device", device->name()},
                      {"inputs",
                       absl::StrJoin(item.kernel->def().input(), "; ")},
                      {"original_node_names",
                       absl::StrJoin(item.kernel->def()
                                         .experimental_debug_info()
                                         .original_node_names(),
                                     "; ")},
                      {"original_func_names",
                       absl::StrJoin(item.kernel->def()
                                         .experimental_debug_info()
                                         .original_func_names(),
                                     "; ")},
                  });
            },
            /*level=*/2);

    params->track_allocations = false;
    stats = nullptr;
    if (stats_collector_ && !tagged_node.get_is_dead()) {
      stats = stats_collector_->CreateNodeExecStats(&item.kernel->def());
      // Track allocations if and only if we are collecting statistics, and
      // `stats` object is expecting allocations to be tracked.
      params->track_allocations = stats ? stats->TrackAllocations() : false;
      nodestats::SetScheduled(stats, scheduled_nsec);
      nodestats::SetAllStart(stats);
    }

    if (vlog_) {
      VLOG(1) << "Process node: " << id << " step " << params->step_id << " "
              << SummarizeNodeDef(item.kernel->def())
              << (tagged_node.get_is_dead() ? " is dead" : "")
              << " device: " << device->name();
    }

    Entry* first_input = propagator_.GetInputTensors(tagged_node);

    // Only execute this node if it is not dead or it is a send/recv
    // transfer node. For transfer nodes, we need to propagate the "dead"
    // bit even when the node is dead.
    bool launched_asynchronously = false;
    if (tagged_node.get_is_dead() && !item.is_transfer_node) {
      if (outputs.size() < item.num_outputs) outputs.resize(item.num_outputs);
    } else if (TF_PREDICT_FALSE(item.is_noop)) {
      ProcessNoop(stats);
    } else if (item.const_tensor != nullptr && !params->track_allocations) {
      ProcessConstTensor(item, &outputs, stats);
    } else {
      // Prepares inputs.
      bool is_input_dead = false;
      s = PrepareInputs(item, first_input, inputs.get(), &input_alloc_attrs,
                        &is_input_dead);
      if (!s.ok()) {
        // Clear inputs.
        const int num_inputs = item.num_inputs;
        for (int i = 0; i < num_inputs; ++i) {
          (first_input + i)->ClearVal();
        }
        propagator_.MaybeMarkCompleted(tagged_node);
        activity_watcher::ActivityEnd(activity_id);
        // Continue to process the nodes in 'inline_ready'.
        completed = NodeDone(s, ready.get(), stats, inline_ready);
        continue;
      }

      // Set up compute params.
      params->op_kernel = item.kernel;
      params->frame_iter = propagator_.GetFrameAndIter(tagged_node);
      params->is_input_dead = is_input_dead;
      params->output_attr_array = item.output_attrs();
      params->forward_from_array = item.forward_from();
      params->outputs_required_array = item.outputs_required.get();
      params->inputs = *inputs;
      params->input_alloc_attrs = input_alloc_attrs;

      if (item.kernel_is_async) {
        ProcessAsync(item, *params, tagged_node, first_input, stats,
                     activity_id);
        launched_asynchronously = true;
      } else {
        s = ProcessSync(item, params.get(), &outputs, stats);
      }
    }

    if (!launched_asynchronously) {
      if (vlog_) {
        VLOG(2) << "Synchronous kernel done: " << id << " step "
                << params->step_id << " "
                << SummarizeNodeDef(item.kernel->def())
                << (tagged_node.get_is_dead() ? " is dead: " : "")
                << " device: " << device->name();
      }

      // Clears inputs.
      const int num_inputs = item.num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->ClearVal();
      }
      propagator_.MaybeMarkCompleted(tagged_node);
      activity_watcher::ActivityEnd(activity_id);
      // Propagates outputs.
      if (s.ok()) {
        propagator_.PropagateOutputs(tagged_node, &outputs, ready.get());
      }

      // Clear outputs without deallocating the `outputs` vector.
      const int num_outputs = item.num_outputs;
      for (int i = 0; i < num_outputs; ++i) {
        outputs[i].ClearVal();
      }

      if (stats) {
        scheduled_nsec = nodestats::NowInNsec();
      }
      // Postprocess.
      completed = NodeDone(s, ready.get(), stats, inline_ready);
    }
  }  // while !inline_ready.empty()

  // This thread of computation is done if completed = true.
  if (completed) ScheduleFinish();
}

template <class PropagatorStateType>
absl::Status ExecutorState<PropagatorStateType>::PrepareInputs(
    const NodeItem& item, Entry* first_input, TensorValueVec* inputs,
    AllocatorAttributeVec* input_alloc_attrs, bool* is_input_dead) {
  inputs->resize(item.num_inputs);
  input_alloc_attrs->resize(item.num_inputs);

  *is_input_dead = false;

  for (int i = 0; i < item.num_inputs; ++i) {
    const bool expect_ref = TF_PREDICT_FALSE(item.is_any_input_ref_typed) &&
                            IsRefType(item.input_type(i));
    Entry* entry = first_input + i;
    (*input_alloc_attrs)[i] = entry->alloc_attr;

    // i-th input.
    TensorValue* inp = &(*inputs)[i];

    switch (entry->state) {
      case Entry::State::NO_VALUE: {
        // Only merge and transfer nodes can have no-value inputs.
        inp->mutex_if_ref = nullptr;
        if (item.is_merge) {
          inp->tensor = nullptr;
        } else {
          DCHECK(item.is_transfer_node)
              << item.kernel->name() << " - input " << i;
          entry->state = Entry::State::HAS_CONST_TENSOR;
          entry->const_tensor = kEmptyTensor;
          // NOTE(mrry): This `const_cast` is necessary because `TensorValue`
          // stores a non-const `Tensor*`, and relies on the `OpKernelContext`
          // accessors making dynamic checks that prevent using an immutable
          // tensor as a mutable tensor.
          inp->tensor = const_cast<Tensor*>(kEmptyTensor);
          *is_input_dead = true;
        }
        break;
      }

      case Entry::State::HAS_VALUE: {
        if (TF_PREDICT_FALSE(expect_ref)) {
          return AttachDef(
              errors::InvalidArgument(i, "-th input expects a ref type"),
              item.kernel->def());
        }
        inp->mutex_if_ref = nullptr;
        inp->tensor = entry->val.get();
        break;
      }

      case Entry::State::HAS_CONST_TENSOR: {
        if (TF_PREDICT_FALSE(expect_ref)) {
          return AttachDef(
              errors::InvalidArgument(i, "-th input expects a ref type"),
              item.kernel->def());
        }
        // NOTE(mrry): This `const_cast` is necessary because `TensorValue`
        // stores a non-const `Tensor*`, and relies on the `OpKernelContext`
        // accessors making dynamic checks that prevent using an immutable
        // tensor as a mutable tensor.
        inp->mutex_if_ref = nullptr;
        inp->tensor = const_cast<Tensor*>(entry->const_tensor);
        break;
      }

      case Entry::State::HAS_REF_TENSOR: {
        {
          tf_shared_lock ml(*entry->ref_tensor.mu);
          if (TF_PREDICT_FALSE(!entry->ref_tensor.tensor->IsInitialized() &&
                               !item.is_initialization_op)) {
            return AttachDef(errors::FailedPrecondition(
                                 "Attempting to use uninitialized value ",
                                 item.kernel->requested_input(i)),
                             item.kernel->def());
          }
        }

        if (expect_ref) {
          inp->mutex_if_ref = entry->ref_tensor.mu;
          inp->tensor = entry->ref_tensor.tensor;
        } else {
          // Automatically deref the tensor ref when the op expects a
          // tensor but is given a ref to a tensor.  Need to deref it
          // under the mutex.
          {
            mutex* ref_mu = entry->ref_tensor.mu;
            Tensor* ref_tensor = entry->ref_tensor.tensor;
            tf_shared_lock l(*ref_mu);
            entry->val.Init(*ref_tensor);
          }
          entry->state = Entry::State::HAS_VALUE;

          inp->mutex_if_ref = nullptr;
          inp->tensor = entry->val.get();
          // The dtype of entry->ref_tensor.tensor could have been changed by
          // another operation that ran after the operation that "produced" it
          // executed, so re-validate that the type of the dereferenced tensor
          // matches the expected input type.
          if (TF_PREDICT_FALSE(item.input_type(i) != inp->tensor->dtype())) {
            return AttachDef(
                errors::InvalidArgument(
                    i, "-th input expects type ",
                    DataTypeString(item.input_type(i)),
                    " but automatically dereferenced input tensor has type ",
                    DataTypeString(inp->tensor->dtype())),
                item.kernel->def());
          }
        }
        break;
      }
    }
  }
  return absl::OkStatus();
}

template <class PropagatorStateType>
absl::Status ExecutorState<PropagatorStateType>::ProcessOutputs(
    const NodeItem& item, OpKernelContext* ctx, Entry* outputs,
    NodeExecStatsInterface* stats) {
  absl::Status s = ctx->status();
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());
    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      LOG(WARNING) << this << " Compute status: " << s;
    }
    if (s.code() == error::RESOURCE_EXHAUSTED) {
      if (stats_collector_) {
        string err =
            stats_collector_->ReportAllocsOnResourceExhausted(s.message());
        s = errors::CreateWithUpdatedMessage(s,
                                             strings::StrCat(s.message(), err));
      } else {
        s = errors::CreateWithUpdatedMessage(
            s,
            strings::StrCat(
                s.message(),
                "\nHint: If you want to see a list of allocated tensors when "
                "OOM happens, add report_tensor_allocations_upon_oom "
                "to RunOptions for current allocation info. This isn't "
                "available when running in Eager mode.\n"));
      }
    } else if (s.code() == error::UNAVAILABLE &&
               !item.is_distributed_communication) {
      s = errors::ReplaceErrorFromNonCommunicationOps(s, item.kernel->name());
    }
    return ADD_SOURCE_LOCATION(s);
  }

  for (int i = 0; i < item.num_outputs; ++i) {
    const TensorValue val = ctx->release_output(i);
    Entry* out = &outputs[i];
    DCHECK(out->state == Entry::State::NO_VALUE);

    if (val.tensor == nullptr) {
      // Unless it's a Switch or a Recv, or the executor has marked the output
      // as not required, the node must produce a tensor value at i-th output.
      if (!(item.is_recv_or_switch ||
            (item.outputs_required && !item.outputs_required[i]))) {
        s.Update(errors::Internal("Missing ", i, "-th output from ",
                                  FormatNodeDefForError(item.kernel->def())));
      }
    } else {
      // Set the allocator attributes of the output entry.
      out->alloc_attr = ctx->output_alloc_attr(i);

      // Sanity check of output tensor types. We need to inspect this safely as
      // we are in the tensor buffer.
      DataType dtype = val.dtype_safe();
      if (dtype == item.output_type(i)) {
        if (stats && val.tensor->IsInitialized()) {
          nodestats::SetOutput(stats, i, val.tensor);
        }
        if (val.is_ref()) {
          out->state = Entry::State::HAS_REF_TENSOR;
          out->ref_tensor.tensor = val.tensor;
          out->ref_tensor.mu = val.mutex_if_ref;
          if (log_memory_) {
            Tensor to_log;
            {
              // Dereference the tensor under the lock.
              tf_shared_lock l(*out->ref_tensor.mu);
              to_log = *out->ref_tensor.tensor;
            }
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, to_log);
          }
        } else {
          // NOTE that std::move is used here, so val.tensor goes to
          // uninitialized state (val.tensor->IsInitialized return false).
          out->state = Entry::State::HAS_VALUE;
          out->val.Init(std::move(*val.tensor));
          if (log_memory_) {
            LogMemory::RecordTensorOutput(ctx->op_kernel().name(),
                                          ctx->step_id(), i, *out->val);
          }
        }
      } else {
        s.Update(
            errors::Internal("Output ", i, " of type ", DataTypeString(dtype),
                             " does not match declared output type ",
                             DataTypeString(item.output_type(i)), " for node ",
                             FormatNodeDefForError(item.kernel->def())));
      }
    }
    if (!val.is_ref()) {
      // If OpKernelContext returns outputs via pass-by-value, we
      // don't need this trouble.
      delete val.tensor;
    }
  }
  return s;
}

template <class PropagatorStateType>
bool ExecutorState<PropagatorStateType>::NodeDone(
    const absl::Status& s, TaggedNodeSeq* ready, NodeExecStatsInterface* stats,
    TaggedNodeReadyQueue* inline_ready) {
  if (stats) {
    nodestats::SetAllEnd(stats);
    DCHECK_NE(stats_collector_, nullptr);
    stats->Done(immutable_state_.params().device->name());
  }

  if (TF_PREDICT_TRUE(s.ok())) {
    const size_t ready_size = ready->size();
    if (ready_size == 0) {
      return num_outstanding_ops_.fetch_sub(1) == 1;
    } else {
      // NOTE: Avoid touching the atomic counter if only one node becomes ready.
      if (ready_size > 1) {
        num_outstanding_ops_.fetch_add(ready_size - 1,
                                       std::memory_order_relaxed);
      }

      // Schedule the ready nodes in 'ready'.
      ScheduleReady(ready, inline_ready);

      return false;
    }
  } else {
    bool abort_run = false;
    absl::Status maybe_derived_s(s);

    // Some error happened. This thread of computation is done.
    {
      mutex_lock l(mu_);
      if (status_.ok()) {
        // If this is the first node to fail in this run, we are responsible for
        // aborting all other execution in the step.
        abort_run = true;

        // If execution has been cancelled, mark cancelled or aborted errors as
        // being derived. Note that the original node that fails might also
        // trigger cancellation, and here we make sure the original error is
        // exposed to users and not buried as a derived error.
        if (cancellation_manager_ && cancellation_manager_->IsCancelled() &&
            (errors::IsCancelled(s) || errors::IsAborted(s))) {
          status_ = StatusGroup::MakeDerived(s);
          maybe_derived_s = status_;
        } else {
          status_ = s;
        }
      }
    }

    if (abort_run) {
      TRACEPRINTF("StartAbort: %s", s.ToString());
      if (cancellation_manager_) {
        // Use VLOG instead of LOG(warning) because error status is expected
        // when the executor is run under the grappler optimization phase or
        // when iterating through a tf.data input pipeline.
        VLOG(1) << "[" << immutable_state_.params().device->name()
                << "] Executor start aborting: " << s;
      }

      if (rendezvous_) {
        rendezvous_->StartAbort(s);
      }
      if (cancellation_manager_) {
        cancellation_manager_->StartCancelWithStatus(maybe_derived_s);
      } else if (collective_executor_) {
        // If there's cancellation_manager_, collective ops aborts
        // collective_executor_ upon cancellation; otherwise we need to abort
        // here.
        collective_executor_->StartAbort(s);
      }
    }

    return num_outstanding_ops_.fetch_sub(1) == 1;
  }
}

template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ScheduleReady(
    TaggedNodeSeq* ready, TaggedNodeReadyQueue* inline_ready) {
  tsl::profiler::TraceMe activity(
      [&]() {
        return strings::StrCat(
            "ExecutorState::ScheduleReady#",
            "ready_size=", (ready == nullptr ? -1 : ready->size()),
            ",inline_ready_size=",
            (inline_ready == nullptr ? -1 : inline_ready->size()), "#");
      },
      tsl::profiler::GetTFTraceMeLevel(/*is_expensive=*/false));
  DCHECK(!ready->empty());

  int64_t scheduled_nsec = 0;
  if (stats_collector_) {
    scheduled_nsec = nodestats::NowInNsec();
  }

  if (run_all_kernels_inline_) {
    if (inline_ready == nullptr) {
      // Schedule all ready kernels from a single closure. This ensure that,
      // regardless of the `runner_` implementation, all kernels will run
      // sequentially on the same thread, and thread wakeup overhead and
      // executor mutex contention will be minimized.
      RunTask([this, ready = std::move(*ready), scheduled_nsec]() {
        for (auto& tagged_node : ready) {
          Process(tagged_node, scheduled_nsec);
        }
      });
    } else {
      for (auto& tagged_node : *ready) {
        inline_ready->push_back(tagged_node);
      }
    }
  } else {
    const TaggedNode* curr_expensive_node = nullptr;
    TaggedNodeSeq expensive_nodes;
    if (inline_ready == nullptr) {
      // Schedule to run all the ready ops in thread pool.
      for (auto& tagged_node : *ready) {
        RunTask([=]() { Process(tagged_node, scheduled_nsec); },
                /*sample_rate=*/ready->size());
      }
    } else {
      for (auto& tagged_node : *ready) {
        const NodeItem& item = *tagged_node.node_item;
        if (tagged_node.get_is_dead() || !kernel_stats_->IsExpensive(item)) {
          // Inline this inexpensive node.
          inline_ready->push_back(tagged_node);
        } else {
          if (curr_expensive_node) {
            expensive_nodes.push_back(*curr_expensive_node);
          }
          curr_expensive_node = &tagged_node;
        }
      }
    }
    if (curr_expensive_node) {
      if (inline_ready->empty()) {
        inline_ready->push_back(*curr_expensive_node);
      } else {
        // There are inline nodes to run already. We dispatch this expensive
        // node to other thread.
        expensive_nodes.push_back(*curr_expensive_node);
      }
    }
    if (!expensive_nodes.empty()) {
      if (expensive_nodes.size() < kInlineScheduleReadyThreshold) {
        for (auto& tagged_node : expensive_nodes) {
          RunTask(std::bind(&ExecutorState::Process, this, tagged_node,
                            scheduled_nsec),
                  /*sample_rate=*/expensive_nodes.size());
        }
      } else {
        // There are too many ready expensive nodes. Schedule them in child
        // threads.
        // TODO(fishx): Apply the same optimization to cheap ops as well since
        // executing lots of cheap ops in one thread can potentially be the
        // bottleneck as well.
        auto it = expensive_nodes.begin();
        while (it < expensive_nodes.end()) {
          auto end = it;
          std::advance(end, kInlineScheduleReadyThreshold);
          if (end > expensive_nodes.end()) {
            end = expensive_nodes.end();
          }
          TaggedNodeSeq ready_chunk{it, end};
          RunTask(
              [this, ready_chunk = std::move(ready_chunk), scheduled_nsec]() {
                tsl::profiler::TraceMe activity(
                    [&]() {
                      return strings::StrCat(
                          "ExecutorState::ScheduleReady::"
                          "ChildThreadExpensiveNodes#",
                          "ready_chunk_size=", ready_chunk.size(), "#");
                    },
                    tsl::profiler::GetTFTraceMeLevel(/*is_expensive=*/false));
                for (auto& tagged_node : ready_chunk) {
                  RunTask(std::bind(&ExecutorState::Process, this, tagged_node,
                                    scheduled_nsec),
                          /*sample_rate=*/ready_chunk.size());
                }
              });
          it = end;
        }
      }
    }
  }
  ready->clear();
}

template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::ScheduleFinish() {
  // Checks condition to decide if needs to invoke Finish(). If there are
  // in-flight deffered ops, wait for `num_deferred_ops_` reaches 0 to invoke
  // Finish(). Otherwise, invoke Finish() directly.
  // Note that it is critical that the ScheduleFinish / Finish codepath does not
  // block, otherwise we might deadlock.  See b/124523000 for details.
  {
    mutex_lock lock(num_deferred_ops_mu_);
    if (num_deferred_ops_ > 0) {
      finish_when_deferred_ops_done_ = true;
      return;
    }
  }
  // Finish is always called exactly once per ExecutorState, either here if
  // there aren't any deferred ops, or in the dec_num_deferred_ops_function if
  // there are deferred ops.
  Finish();
}

template <class PropagatorStateType>
void ExecutorState<PropagatorStateType>::Finish() {
  mu_.lock();
  auto status = status_;
  auto done_cb = std::move(done_cb_);
  auto runner = std::move(runner_);
  mu_.unlock();
  int64_t trace_id = trace_id_;
  int64_t step_id = step_id_;
  CHECK(done_cb != nullptr);
  Device* device = immutable_state_.params().device;

  if (vlog_ && !status.ok() && VLOG_IS_ON(1)) {
    // Logs verbose information about the current state of active and pending
    // nodes in the propagator.
    propagator_.DumpState();
  }

  // There are several potential race conditions below. To name a few:
  // 1. Even if the device's status is OK at the precise moment when
  // num_deferred_ops_ reaches 0, it could go bad before device->RefreshStatus()
  // is called below, caused by work enqueued onto the same device by other
  // concurrent ExecutorState objects.
  // 2. Some implementations of Device::RefreshStatus, such as
  // XlaDevice::RefreshStatus, may be inherently racy because it releases the
  // device mutex after a stream pointer is acquired and before the stream is
  // queried for status.
  // 3. It's the same for some implementations of Device::Sync, such as
  // XlaDevice::Sync.
  //
  // However, these race conditions are acceptable because a stream (and
  // therefore an XlaDevice) can only go from OK to not-OK, never the opposite,
  // which means we will at worst report errors when there isn't any, never the
  // opposite.

  // An early exit for devices don't allow sync on completion. Ops that run on
  // these devices should have used num_deferred_ops correctly to ensure the
  // device has finished all relevant work at this point.
  if (!device->AllowsSyncOnCompletion()) {
    status.Update(device->RefreshStatus());
    if (!status.ok()) {
      // In device async execution mode, it's possible for device execution to
      // lag behind ExecutorState scheduling so much that this is the first
      // place a device execution error surfaces.
      // If so, all ExecutorState::NodeDone calls have already happened with OK
      // status. This is the last defense where StartCancel must be called to
      // abort all computation still running on any device.
      // TODO(b/124523000): Always call Finish in a separate thread, so even if
      // StartCancel blocks the current thread's execution, we won't encounter
      // deadlocks caused by inter-op thread exhaustion.
      if (rendezvous_) {
        rendezvous_->StartAbort(status);
      }
      if (cancellation_manager_) {
        cancellation_manager_->StartCancelWithStatus(status);
      } else if (collective_executor_) {
        // If there's cancellation_manager_, collective ops aborts
        // collective_executor_ upon cancellation; otherwise we need to abort
        // here.
        collective_executor_->StartAbort(status);
      }
    }
    delete this;
    runner([step_id, trace_id, status, done_cb = std::move(done_cb)]() {
      tsl::profiler::TraceMeConsumer activity(
          // From TraceMeProducer in KernelAndDeviceFunc::RunAsync,
          // DirectSession::RunInternal or GraphMgr::ExecuteAsync.
          [&] {
            return tsl::profiler::TraceMeEncode("ExecutorDoneCallback",
                                                {{"id", step_id}});
          },
          tsl::profiler::ContextType::kTfExecutor, trace_id,
          tsl::profiler::TraceMeLevel::kInfo);
      done_cb(status);
    });
    return;
  }

  if (sync_on_finish_ && status.ok()) {
    // Block until the device has finished all queued operations. For
    // devices like GPUs that continue to execute Ops after their Compute
    // methods have completed, this ensures that control is not returned to
    // the user until the step (and its side-effects) has actually completed.
    device->Sync([this, step_id, trace_id, runner = std::move(runner),
                  done_cb =
                      std::move(done_cb)](const absl::Status& status) mutable {
      delete this;
      runner([step_id, trace_id, status, done_cb = std::move(done_cb)]() {
        tsl::profiler::TraceMeConsumer activity(
            // From TraceMeProducer in KernelAndDeviceFunc::RunAsync,
            // DirectSession::RunInternal or GraphMgr::ExecuteAsync.
            [&] {
              return tsl::profiler::TraceMeEncode("ExecutorDoneCallback",
                                                  {{"id", step_id}});
            },
            tsl::profiler::ContextType::kTfExecutor, trace_id,
            tsl::profiler::TraceMeLevel::kInfo);
        done_cb(status);
      });
    });
  } else {
    delete this;
    runner([step_id, trace_id, status, done_cb = std::move(done_cb)]() {
      tsl::profiler::TraceMeConsumer activity(
          // From TraceMeProducer in KernelAndDeviceFunc::RunAsync,
          // DirectSession::RunInternal or GraphMgr::ExecuteAsync.
          [&] {
            return tsl::profiler::TraceMeEncode("ExecutorDoneCallback",
                                                {{"id", step_id}});
          },
          tsl::profiler::ContextType::kTfExecutor, trace_id,
          tsl::profiler::TraceMeLevel::kInfo);
      done_cb(status);
    });
  }
}

void ExecutorImpl::RunAsyncInternal(const Args& args, DoneCallback done) {
  if (OpOrderDeterminismRequired()) {
    (new ExecutorState<OrderedPropagatorState>(args, immutable_state_,
                                               &kernel_stats_))
        ->RunAsync(std::move(done));
  } else if (immutable_state_.requires_control_flow_support()) {
    (new ExecutorState<PropagatorState>(args, immutable_state_, &kernel_stats_))
        ->RunAsync(std::move(done));
  } else {
    (new ExecutorState<SimplePropagatorState>(args, immutable_state_,
                                              &kernel_stats_))
        ->RunAsync(std::move(done));
  }
}

}  // namespace

absl::Status NewLocalExecutor(const LocalExecutorParams& params,
                              const Graph& graph, Executor** executor) {
  ExecutorImpl* impl = new ExecutorImpl(params);
  const absl::Status s = impl->Initialize(graph);
  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}

absl::Status CreateNonCachedKernel(
    Device* device, FunctionLibraryRuntime* flib,
    const std::shared_ptr<const NodeProperties>& props, int graph_def_version,
    OpKernel** kernel) {
  const auto device_type = DeviceType(device->attributes().device_type());
  auto allocator = device->GetAllocator(AllocatorAttributes());
  return CreateOpKernel(device_type, device, allocator, flib,
                        device->resource_manager(), props, graph_def_version,
                        kernel);
}

void DeleteNonCachedKernel(OpKernel* kernel) { delete kernel; }

namespace {

class DefaultExecutorRegistrar {
 public:
  DefaultExecutorRegistrar() {
    Factory* factory = new Factory;
    ExecutorFactory::Register("", factory);
    ExecutorFactory::Register("DEFAULT", factory);
  }

 private:
  class Factory : public ExecutorFactory {
    absl::Status NewExecutor(const LocalExecutorParams& params,
                             const Graph& graph,
                             std::unique_ptr<Executor>* out_executor) override {
      Executor* ret = nullptr;
      TF_RETURN_IF_ERROR(NewLocalExecutor(params, graph, &ret));
      out_executor->reset(ret);
      return absl::OkStatus();
    }
  };
};
static DefaultExecutorRegistrar registrar;

}  // namespace

}  // namespace tensorflow
