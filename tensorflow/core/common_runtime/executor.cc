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

#include <atomic>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/executor_factory.h"
#include "tensorflow/core/common_runtime/graph_view.h"
#include "tensorflow/core/common_runtime/immutable_executor_state.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/common_runtime/renamed_device.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/log_memory.h"
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
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/profile_utils/cpu_utils.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/annotated_traceme.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/tensor_slice_reader_cache.h"

namespace tensorflow {
namespace {

// 1-D, 0 element tensor.
static const Tensor* const kEmptyTensor = new Tensor;

// Helper routines for collecting step stats.
namespace nodestats {
inline int64 NowInNsec() { return EnvTime::NowNanos(); }

void SetScheduled(NodeExecStatsInterface* stats, int64 micros) {
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

class ExecutorImpl;

// Time the execution of kernels (in CPU cycles).  Used to dynamically identify
// inexpensive kernels which can be dispatched inline.
struct KernelTimer {
  uint64 start_cycles = profile_utils::CpuUtils::GetCurrentClockCycle();

  uint64 ElapsedCycles() {
    return profile_utils::CpuUtils::GetCurrentClockCycle() - start_cycles;
  }
};

typedef gtl::InlinedVector<TensorValue, 4> TensorValueVec;
typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

class ExecutorImpl : public Executor {
 public:
  explicit ExecutorImpl(const LocalExecutorParams& p) : immutable_state_(p) {}

  Status Initialize(const Graph& graph) {
    TF_RETURN_IF_ERROR(immutable_state_.Initialize(graph));
    kernel_stats_.Initialize(immutable_state_.graph_view());
    return Status::OK();
  }

  void RunAsync(const Args& args, DoneCallback done) override;

 private:
  friend class ExecutorState;

  // Stores execution time information about the kernels in an executor's graph.
  class KernelStats {
   public:
    KernelStats() = default;

    void Initialize(const GraphView& gview) {
      is_expensive_ = absl::make_unique<std::atomic<bool>[]>(gview.num_nodes());
      cost_estimates_ =
          absl::make_unique<std::atomic_uint_fast64_t[]>(gview.num_nodes());
      for (int32 i = 0; i < gview.num_nodes(); ++i) {
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

    // Updates the dynamic cost estimate, which is used to determine whether the
    // given node is expensive. The new cost estimate is a weighted average of
    // the old cost estimate and the latest cost.
    //
    // NOTE: We currently only expect updates to the cost estimate when
    // `is_expensive_[node.node_id]` is true (or at least, it *was* true, when
    // we started to execute the kernel. As a result, we expect that a kernel
    // can only ever transition from "expensive" to "inexpensive", but not vice
    // versa.
    void UpdateCostEstimate(const NodeItem& node, uint64 elapsed_cycles) {
      // N.B. Updates to `cost_estimate` are atomic but unlocked.  Simultaneous
      // updates may result in one or more updates being ignored.  This does not
      // affect correctness but may slow down the update frequency.
      std::atomic_uint_fast64_t& cost_estimate = cost_estimates_[node.node_id];
      uint64 new_estimate = (kCostDecay - 1) *
                                cost_estimate.load(std::memory_order_relaxed) /
                                kCostDecay +
                            (elapsed_cycles / kCostDecay);
      cost_estimate.store(new_estimate, std::memory_order_relaxed);
      if (new_estimate < kOpIsExpensiveThresholdCycles) {
        is_expensive_[node.node_id].store(false, std::memory_order_relaxed);
      }
    }

   private:
    // Initial time (in CPU cycles) we expect an operation to take.  Used to
    // determine whether an operation should be place in a threadpool.
    // Operations start out "expensive".
    static const uint64 kInitialCostEstimateCycles = 100 * 1000 * 1000;
    static const uint64 kOpIsExpensiveThresholdCycles = 5000;
    static const uint64 kCostDecay = 10;

    std::unique_ptr<std::atomic<bool>[]> is_expensive_;
    std::unique_ptr<std::atomic_uint_fast64_t[]> cost_estimates_;
  };

  ImmutableExecutorState immutable_state_;
  KernelStats kernel_stats_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExecutorImpl);
};

// The state associated with one invocation of ExecutorImpl::Run.
// ExecutorState dispatches nodes when they become ready and keeps
// track of how many predecessors of a node have not done (pending_).
class ExecutorState {
 public:
  ExecutorState(const Executor::Args& args,
                const ImmutableExecutorState& immutable_state_,
                ExecutorImpl::KernelStats* kernel_stats_);
  ~ExecutorState();

  void RunAsync(Executor::DoneCallback done);

 private:
  // Either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
  struct Entry {
    enum class State {
      NO_VALUE = 0,      // The default state for a newly-created Entry.
      HAS_VALUE,         // `this->val` is valid.
      HAS_CONST_TENSOR,  // `this->const_tensor` is valid.
      HAS_REF_TENSOR,    // `this->ref_tensor` is valid.
    };

    Entry() : state(State::NO_VALUE) {}
    Entry(const Entry& other)
        : state(other.state), alloc_attr(other.alloc_attr) {
      switch (state) {
        case State::NO_VALUE:
          break;
        case State::HAS_VALUE:
          val.Init(*other.val);
          break;
        case State::HAS_CONST_TENSOR:
          const_tensor = other.const_tensor;
          break;
        case State::HAS_REF_TENSOR:
          ref_tensor = other.ref_tensor;
          break;
      }
    }

    ~Entry() {
      if (state == State::HAS_VALUE) val.Destroy();
    }

    Entry& operator=(const Entry& other) {
      if (state == State::HAS_VALUE) {
        val.Destroy();
      }
      state = other.state;
      alloc_attr = other.alloc_attr;
      switch (state) {
        case State::NO_VALUE:
          break;
        case State::HAS_VALUE:
          val.Init(*other.val);
          break;
        case State::HAS_CONST_TENSOR:
          const_tensor = other.const_tensor;
          break;
        case State::HAS_REF_TENSOR:
          ref_tensor = other.ref_tensor;
          break;
      }
      return *this;
    }

    Entry& operator=(Entry&& other) {
      if (state == State::HAS_VALUE) {
        val.Destroy();
      }
      state = other.state;
      alloc_attr = other.alloc_attr;
      switch (state) {
        case State::NO_VALUE:
          break;
        case State::HAS_VALUE:
          val.Init(std::move(*other.val));
          break;
        case State::HAS_CONST_TENSOR:
          const_tensor = other.const_tensor;
          break;
        case State::HAS_REF_TENSOR:
          ref_tensor = other.ref_tensor;
          break;
      }
      return *this;
    }

    // Clears the <val> field, and sets this entry to the `NO_VALUE` state.
    void ClearVal() {
      if (state == State::HAS_VALUE) {
        val.Destroy();
      }
      state = State::NO_VALUE;
    }

    union {
      // A tensor value. Valid iff `state_ == HAS_VALUE`.
      ManualConstructor<Tensor> val;

      // A pointer to a constant tensor value. Valid iff `state_ ==
      // HAS_CONST_TENSOR`.
      const Tensor* const_tensor;

      // A tensor reference and associated mutex. Valid iff `state_ ==
      // HAS_REF_TENSOR`.
      struct {
        Tensor* tensor;
        mutex* mu;
      } ref_tensor;
    };

    // The current state of this entry, indicating which member of the above
    // union is active.
    State state;

    // The attributes of the allocator that creates the tensor.
    AllocatorAttributes alloc_attr;
  };

  // Contains the device context assigned by the device at the beginning of a
  // step.
  DeviceContext* device_context_ = nullptr;

  struct TaggedNode;
  typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;
  typedef gtl::InlinedVector<Entry, 4> EntryVector;

  struct IterationState {
    explicit IterationState(const PendingCounts* pending_counts,
                            int total_input_tensors)
        : input_tensors(new Entry[total_input_tensors]),
          outstanding_ops(0),
          outstanding_frame_count(0),
          counts(*pending_counts) {  // Initialize with copy of *pending_counts
    }

    // The state of an iteration.

    // One copy per iteration. For iteration k, i-th node's j-th input is in
    // input_tensors[k][immutable_state_.nodes[i].input_start + j]. An entry is
    // either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
    //
    // NOTE: No need to protect input_tensors[i] by any locks because it
    // is resized once. Each element of tensors_ is written once by the
    // source node of an edge and is cleared by the destination of the same
    // edge. The latter node is never run concurrently with the former node.
    Entry* input_tensors;

    // The number of outstanding ops for each iteration.
    size_t outstanding_ops;

    // The number of outstanding frames for each iteration.
    int outstanding_frame_count;
    int pending(PendingCounts::Handle h) { return counts.pending(h); }
    int decrement_pending(PendingCounts::Handle h, int v) {
      return counts.decrement_pending(h, v);
    }
    // Mark a merge node as live
    // REQUIRES: Node corresponding to "h" is a merge node
    void mark_live(PendingCounts::Handle h) { counts.mark_live(h); }
    // Mark a node to show that processing has started.
    void mark_started(PendingCounts::Handle h) { counts.mark_started(h); }
    // Mark a node to show that processing has completed.
    void mark_completed(PendingCounts::Handle h) { counts.mark_completed(h); }
    PendingCounts::NodeState node_state(PendingCounts::Handle h) {
      return counts.node_state(h);
    }

    int dead_count(PendingCounts::Handle h) { return counts.dead_count(h); }
    void increment_dead_count(PendingCounts::Handle h) {
      counts.increment_dead_count(h);
    }
    PendingCounts::AdjustResult adjust_for_activation(PendingCounts::Handle h,
                                                      bool increment_dead) {
      return counts.adjust_for_activation(h, increment_dead);
    }

    ~IterationState() { delete[] input_tensors; }

   private:
    PendingCounts counts;
  };

  struct FrameState {
    explicit FrameState(const ImmutableExecutorState& immutable_state,
                        int parallel_iters)
        : immutable_state(immutable_state),
          max_parallel_iterations(parallel_iters),
          num_outstanding_iterations(1),
          iterations(parallel_iters + 1),
          iterations_raw(iterations.data()) {}

    // A new frame is created for each loop. Execution starts at iteration 0.
    // When a value at iteration 0 passes through a NextIteration node,
    // iteration 1 is created and starts running. Note that iteration 0 may
    // still be running so multiple iterations may run in parallel. The
    // frame maintains the state of iterations in several data structures
    // such as pending_count and input_tensors. When iteration 0 completes,
    // we garbage collect the state of iteration 0.
    //
    // A frame instance is considered "done" and can be garbage collected
    // if all its inputs have entered and all its iterations are "done".
    //
    // A frame manages the live iterations of an iterative computation.
    // Iteration i is considered "done" when there are no outstanding ops,
    // frames at iteration i are done, all recvs for this iteration are
    // completed, and iteration i-1 is done. For iteration 0, we instead
    // wait for there to be no more pending inputs of the frame.
    //
    // Frames and iterations are garbage collected once they are done.
    // The state we need to keep around is highly dependent on the
    // parallelism enabled by the scheduler. We may want to have the
    // scheduler dynamically control the outstanding number of live
    // parallel frames and iterations. To reduce the state space, the
    // scheduler might want to schedule ops in inner frames first and
    // lower iterations first.
    //
    // This frame state is mostly initialized lazily on demand so we
    // don't introduce unnecessary overhead.

    // The immutable state of the executor the frame is in.
    const ImmutableExecutorState& immutable_state;

    // The name of this frame, which is the concatenation of its parent
    // frame name, the iteration of the parent frame when this frame was
    // created, and the value of the attr 'frame_name'.
    string frame_name;

    // The unique id for this frame. Generated by fingerprinting
    // frame_name.
    uint64 frame_id;

    // The iteration id of its parent frame when this frame is created.
    // -1 if there is no parent frame. The frame_name/parent_iter pair
    // uniquely identifies this FrameState.
    int64 parent_iter = -1;

    // The FrameState of its parent frame.
    FrameState* parent_frame = nullptr;

    // The maximum allowed number of parallel iterations.
    const int max_parallel_iterations;

    // The number of inputs this frame is still waiting.
    int num_pending_inputs = 0;

    // The highest iteration number we have reached so far in this frame.
    int64 iteration_count TF_GUARDED_BY(mu) = 0;

    // The number of outstanding iterations.
    int num_outstanding_iterations TF_GUARDED_BY(mu) = 1;

   private:
    // The active iteration states of this frame.
    gtl::InlinedVector<IterationState*, 12> iterations;
    IterationState** const iterations_raw TF_GUARDED_BY(mu);
    IterationState* iterations_first TF_GUARDED_BY(mu);

   public:
    // The NextIteration nodes to enter a new iteration. If the number of
    // outstanding iterations reaches the limit, we will defer the start of
    // the next iteration until the number of outstanding iterations falls
    // below the limit.
    std::vector<std::pair<const NodeItem*, Entry>> next_iter_roots
        TF_GUARDED_BY(mu);

    // The values of the loop invariants for this loop. They are added into
    // this list as they "enter" the frame. When a loop invariant enters,
    // we make it available to all active iterations. When the frame starts
    // a new iteration, we make all the current loop invariants available
    // to the new iteration.
    std::vector<std::pair<const NodeItem*, Entry>> inv_values TF_GUARDED_BY(mu);

    // The list of dead exit node items for the current highest iteration. We
    // will only "execute" the dead exits of the final iteration.
    std::vector<const NodeItem*> dead_exits TF_GUARDED_BY(mu);

    // Static information specific to this frame.
    PendingCounts* pending_counts = nullptr;
    int total_input_tensors = 0;
    std::vector<const NodeItem*>* nodes = nullptr;

    // Lock ordering: ExecutorState.mu_ < mu;
    // during structured traversal: parent_frame->mu < mu.
    mutex mu;

    void InitializeFrameInfo(const string& enter_name) {
      const ImmutableExecutorState::FrameInfo* finfo =
          immutable_state.get_frame_info(enter_name);
      DCHECK_NE(finfo, nullptr);
      pending_counts = finfo->pending_counts.get();
      total_input_tensors = finfo->total_inputs;
      num_pending_inputs = finfo->input_count;
      nodes = finfo->nodes.get();
    }

    inline IterationState* GetIteration(int64 iter)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu) {
      if (TF_PREDICT_TRUE(iter == 0)) {
        return iterations_first;
      } else {
        size_t index = iter % (max_parallel_iterations + 1);
        return iterations_raw[index];
      }
    }

    inline void SetIteration(int64 iter, IterationState* state)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu) {
      size_t index = iter % (max_parallel_iterations + 1);
      DCHECK(state == nullptr || iterations[index] == nullptr);
      iterations_raw[index] = state;
      if (index == 0) {
        iterations_first = state;
      }
    }

    // Decrement the outstanding op count and clean up the iterations in the
    // frame. Return true iff the execution of the frame is done.
    inline bool DecrementOutstandingOps(const GraphView* gview, int64 iter,
                                        TaggedNodeSeq* ready) {
      mutex_lock l(mu);
      return DecrementOutstandingOpsLocked(gview, iter, ready);
    }

    // Decrement the outstanding op count and clean up the iterations in the
    // frame. Return true iff the execution of the frame is done.
    inline bool DecrementOutstandingOpsLocked(const GraphView* gview,
                                              int64 iter, TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu) {
      IterationState* istate = GetIteration(iter);
      istate->outstanding_ops--;
      if (istate->outstanding_ops != 0) {
        return false;
      } else {
        return CleanupIterations(gview, iter, ready);
      }
    }

    // Returns true if the computation in the frame is completed.
    inline bool IsFrameDone() TF_EXCLUSIVE_LOCKS_REQUIRED(mu) {
      return (num_pending_inputs == 0 && num_outstanding_iterations == 0);
    }

    // Returns true if the iteration of the frame is completed.
    bool IsIterationDone(int64 iter) TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Increments the iteration id. If this is a new iteration, initialize it.
    void IncrementIteration(const GraphView* gview, TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate all the deferred NextIteration nodes in a new iteration.
    void ActivateNexts(const GraphView* gview, int64 iter, TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate all the current loop invariants in a new iteration.
    void ActivateLoopInvs(const GraphView* gview, int64 iter,
                          TaggedNodeSeq* ready) TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Add a new loop invariant and make it available to all active
    // iterations.
    void AddLoopInv(const NodeItem* item, const Entry& entry,
                    TaggedNodeSeq* ready) TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate the successors of a node. Contents of *outputs are left in an
    // indeterminate state after returning from this method.
    void ActivateNodes(const NodeItem* item, const bool is_dead, int64 iter,
                       EntryVector* outputs, TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Cleanup iterations of this frame starting from iteration iter.
    bool CleanupIterations(const GraphView* gview, int64 iter,
                           TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    void DumpIterationState(ExecutorState* parent) {
      mutex_lock l(mu);
      for (IterationState* iteration : iterations) {
        if (iteration) {
          LOG(WARNING) << "  Iteration:";
          parent->DumpIterationState(this, iteration);
        }
      }
    }

    ~FrameState() {
      for (size_t i = 0; i < iterations.size(); ++i) {
        delete iterations[i];
        iterations[i] = nullptr;
      }
    }

   private:
    // REQUIRES: `!item->is_any_consumer_merge_or_control_trigger`.
    void ActivateNodesFastPath(const NodeItem* item, const bool is_dead,
                               int64 iter, EntryVector* outputs,
                               TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    void ActivateNodesSlowPath(const NodeItem* item, const bool is_dead,
                               int64 iter, EntryVector* outputs,
                               TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);
  };

  // A tagged node: <frame*, iter, node*>.
  struct TaggedNode {
    const NodeItem* node_item;
    FrameState* input_frame;  // = nullptr;
    int64 input_iter;         // = -1;
    bool is_dead;             // = false;

    TaggedNode() {}

    TaggedNode(const NodeItem* node_item, FrameState* in_frame, int64 in_iter,
               bool dead)
        : node_item(node_item),
          input_frame(in_frame),
          input_iter(in_iter),
          is_dead(dead) {}
  };

  // A drop-in replacement for std::deque<TaggedNode>.  We typically don't
  // have that many nodes in the ready queue, so we just use a vector and
  // don't free up memory from the queue as we consume nodes.
  class TaggedNodeReadyQueue {
   public:
    TaggedNodeReadyQueue() : front_index_(0) {}

    void push_back(const TaggedNode& node) { ready_.push_back(node); }
    TaggedNode front() const {
      DCHECK_LT(front_index_, ready_.size());
      return ready_[front_index_];
    }
    void pop_front() {
      DCHECK_LT(front_index_, ready_.size());
      front_index_++;
      if ((front_index_ == ready_.size()) || (front_index_ > 16384)) {
        if (front_index_ == ready_.size()) {
          ready_.clear();
        } else {
          // Lots of unused entries at beginning of vector: move everything
          // down to start of vector.
          ready_.erase(ready_.begin(), ready_.begin() + front_index_);
        }
        front_index_ = 0;
      }
    }
    bool empty() const { return ready_.empty(); }
    const TaggedNode* begin() const { return ready_.begin() + front_index_; }
    const TaggedNode* end() const { return ready_.end(); }

   private:
    gtl::InlinedVector<TaggedNode, 16> ready_;
    int front_index_;
  };

  struct AsyncState;

  const bool vlog_;  // true if VLOG_IS_ON(1). Used to check vlog cheaply.

  // true if LogMemory::IsEnabled(). Used to check memory enabled cheaply.
  const bool log_memory_;

  int64 step_id_;
  // Not owned.
  RendezvousInterface* rendezvous_;
  CollectiveExecutor* collective_executor_ = nullptr;
  SessionState* session_state_;
  string session_handle_;
  const SessionMetadata* session_metadata_ = nullptr;
  TensorStore* tensor_store_;
  // Step-local container.
  ScopedStepContainer* step_container_;
  StepStatsCollectorInterface* const stats_collector_;
  const tracing::EventCollector* const event_collector_;
  Context context_;

  // QUESTION: Make it a checkpoint::TensorSliceReaderCacheWrapper
  // instead of a pointer?  (avoids having to delete).
  checkpoint::TensorSliceReaderCacheWrapper* slice_reader_cache_;
  CallFrameInterface* call_frame_;
  const ImmutableExecutorState& immutable_state_;
  ExecutorImpl::KernelStats* const kernel_stats_;
  CancellationManager* cancellation_manager_;
  // If not null, use this device to schedule intra-op operation
  std::unique_ptr<DeviceBase> user_device_;
  Executor::Args::Runner runner_;
  bool sync_on_finish_;
  const bool run_all_kernels_inline_;

  // Owned.

  // A flag that is set on error after the frame state has been
  // dumped for diagnostic purposes.
  bool dumped_on_error_ = false;

  // The root frame in which the execution of this step is started.
  FrameState* root_frame_;

  // Invoked when the execution finishes.
  Executor::DoneCallback done_cb_;

  std::atomic_int_fast32_t num_outstanding_ops_;

  // Available via OpKernelContext to every OpKernel invocation.
  mutex num_deferred_ops_mu_;
  int64 num_deferred_ops_ TF_GUARDED_BY(num_deferred_ops_mu_) = 0;
  bool finish_when_deferred_ops_done_ TF_GUARDED_BY(num_deferred_ops_mu_) =
      false;

  mutex mu_;
  Status status_ TF_GUARDED_BY(mu_);

  // Mapping from frame name to outstanding frames. A new frame is created
  // at some iteration of an active frame. So the unique key for the new
  // child frame is composed of the name of the parent frame, the iteration
  // number at which the parent frame is creating the new frame, and the
  // name of the new frame from nodedef.
  gtl::FlatMap<string, FrameState*> outstanding_frames_ TF_GUARDED_BY(mu_);

  // The unique name of a frame.
  inline string MakeFrameName(FrameState* frame, int64 iter_id,
                              const string& name) {
    return strings::StrCat(frame->frame_name, ";", iter_id, ";", name);
  }

  // Find an existing or create a new child frame in the frame 'frame' at
  // iteration 'iter'.
  void FindOrCreateChildFrame(FrameState* frame, int64 iter,
                              const NodeItem& node_item, FrameState** child);

  // Delete a frame. Called when the frame is done.
  void DeleteFrame(FrameState* frame, TaggedNodeSeq* ready);

  // Cleanup frames and iterations starting from frame/iter. Called when
  // a child frame is done.
  void CleanupFramesIterations(FrameState* frame, int64 iter,
                               TaggedNodeSeq* ready);

  // Process a ready node in current thread.
  void Process(TaggedNode node, int64 scheduled_nsec);

  Status ProcessSync(const NodeItem& item, OpKernelContext::Params* params,
                     EntryVector* outputs,
                     NodeExecStatsInterface* stats);
  void ProcessAsync(const NodeItem& item, const OpKernelContext::Params& params,
                    const TaggedNode& tagged_node, Entry* first_input,
                    NodeExecStatsInterface* stats);
  void ProcessNoop(NodeExecStatsInterface* stats);
  void ProcessConstTensor(const NodeItem& item, EntryVector* outputs,
                          NodeExecStatsInterface* stats);

  // Before invoking item->kernel, fills in its "inputs".
  Status PrepareInputs(const NodeItem& item, Entry* first_input,
                       TensorValueVec* inputs,
                       AllocatorAttributeVec* input_alloc_attrs,
                       bool* is_input_dead);

  // After item->kernel computation is done, processes its outputs.
  Status ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                        EntryVector* outputs, NodeExecStatsInterface* stats);

  // After processing the outputs, propagates the outputs to their dsts.
  // Contents of *outputs are left in an indeterminate state after
  // returning from this method.
  void PropagateOutputs(const TaggedNode& tagged_node, const NodeItem* item,
                        EntryVector* outputs, TaggedNodeSeq* ready);

  // Called after each node finishes. Takes ownership of "stats". Returns true
  // if execution has completed.
  //
  // This method will clear `*ready` before returning.
  bool NodeDone(const Status& s, TaggedNodeSeq* ready,
                NodeExecStatsInterface* stats,
                TaggedNodeReadyQueue* inline_ready);

  // Schedule all the expensive nodes in '*ready', and put all the inexpensive
  // nodes in 'ready' into 'inline_ready'.
  //
  // This method will clear `*ready` before returning.
  void ScheduleReady(TaggedNodeSeq* ready, TaggedNodeReadyQueue* inline_ready);

  // For debugging/logging only.
  inline void MaybeMarkCompleted(FrameState* frame, int64 iter,
                                 const int node_id);

  // Provide debugging output about an outstanding node in the executor.
  void DumpPendingNodeState(const int node_id, const Entry* input_vector,
                            bool show_nodes_with_no_ready_inputs);
  void DumpActiveNodeState(const int node_id, const Entry* input_vector);

  // Provide debugging output about an outstanding iteration in the executor.
  void DumpIterationState(const FrameState* frame, IterationState* iteration);

  // Provide debugging output of the state of the executor.
  void DumpState();
  const Tensor* GetTensorValueForDump(const Entry& input);

  // Clean up when this executor is done.
  void Finish();
  void ScheduleFinish();

  // A standalone routine for this expression so that we can express
  // that we don't want thread safety analysis on this reference (it's
  // safe to do without the lock because the iterations array never
  // resizes and this particular iteration's array element will not
  // be changed out from under us because the iteration is still alive).
  Entry* GetInputTensors(FrameState* input_frame,
                         int64 input_iter) const TF_NO_THREAD_SAFETY_ANALYSIS {
    return input_frame->GetIteration(input_iter)->input_tensors;
  }
};

ExecutorState::ExecutorState(const Executor::Args& args,
                             const ImmutableExecutorState& immutable_state,
                             ExecutorImpl::KernelStats* kernel_stats)
    : vlog_(VLOG_IS_ON(1)),
      log_memory_(LogMemory::IsEnabled()),
      step_id_(args.step_id),
      rendezvous_(args.rendezvous),
      collective_executor_(args.collective_executor),
      session_state_(args.session_state),
      session_handle_(args.session_handle),
      session_metadata_(immutable_state.params().session_metadata),
      tensor_store_(args.tensor_store),
      step_container_(args.step_container),
      stats_collector_(args.stats_collector),
      event_collector_(
          tracing::GetEventCollector(tracing::EventCategory::kCompute)),
      context_(ContextKind::kThread),
      slice_reader_cache_(new checkpoint::TensorSliceReaderCacheWrapper),
      call_frame_(args.call_frame),
      immutable_state_(immutable_state),
      kernel_stats_(kernel_stats),
      cancellation_manager_(args.cancellation_manager),
      runner_(args.runner),
      sync_on_finish_(args.sync_on_finish),
      run_all_kernels_inline_(args.run_all_kernels_inline),
      num_outstanding_ops_(0) {
  if (args.user_intra_op_threadpool != nullptr) {
    Device* device = immutable_state_.params().device;
    user_device_ = RenamedDevice::NewRenamedDevice(
        device->name(), device, false, false, args.user_intra_op_threadpool);
  }

  // We start the entire execution in iteration 0 of the root frame
  // so let us create the root frame and the state for iteration 0.
  // We assume root_frame_->frame_name.empty().
  root_frame_ = new FrameState(immutable_state_, 1);
  root_frame_->frame_id = 0;  // must be 0
  root_frame_->InitializeFrameInfo(root_frame_->frame_name);

  // Initialize iteration 0.
  root_frame_->SetIteration(
      0, new IterationState(root_frame_->pending_counts,
                            root_frame_->total_input_tensors));

  outstanding_frames_.insert({root_frame_->frame_name, root_frame_});
}

ExecutorState::~ExecutorState() {
  for (auto name_frame : outstanding_frames_) {
    delete name_frame.second;
  }
  if (device_context_) {
    device_context_->Unref();
  }
  delete slice_reader_cache_;
}

void ExecutorState::RunAsync(Executor::DoneCallback done) {
  TaggedNodeSeq ready;

  // Ask the device to fill in the device context map.
  Device* device = immutable_state_.params().device;
  const Status get_context_status =
      device->TryGetDeviceContext(&device_context_);
  if (!get_context_status.ok()) {
    delete this;
    done(get_context_status);
    return;
  }

  // Initialize the ready queue.
  ready.reserve(immutable_state_.root_nodes().size());
  for (const NodeItem* item : immutable_state_.root_nodes()) {
    DCHECK_EQ(item->num_inputs, 0);
    ready.push_back(TaggedNode{item, root_frame_, 0, false});
  }
  if (ready.empty()) {
    delete this;
    done(Status::OK());
  } else {
    num_outstanding_ops_ = ready.size();
    {
      mutex_lock l(root_frame_->mu);
      root_frame_->GetIteration(0)->outstanding_ops = ready.size();
    }
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
struct ExecutorState::AsyncState {
  AsyncState(const OpKernelContext::Params& p, const TaggedNode& _tagged_node,
             const NodeItem* _item, Entry* _first_input,
             NodeExecStatsInterface* _stats)
      : saved_inputs(*p.inputs),
        saved_input_alloc_attrs(*p.input_alloc_attrs),
        params(p),
        tagged_node(_tagged_node),
        item(_item),
        first_input(_first_input),
        // ParamsButClearingEigenGPUDevice does equivalent of
        //   params.eigen_gpu_device = nullptr;
        ctx(ParamsButClearingEigenGPUDevice(&params), item->num_outputs),
        stats(_stats) {
    params.inputs = &saved_inputs;
    params.input_alloc_attrs = &saved_input_alloc_attrs;
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
bool MightTrace(const tracing::EventCollector* event_collector,
                bool is_expensive) {
  // Tracing will only be enabled if either `event_collector` is non null,
  // or `trace_collector` is non-null and enabled for this particular kernel.
  // Although `profiler::TraceMe`, `profiler::ScopedAnnotation`, and
  // `tracing::ScopedRegion` check subsets of these properties internally in
  // their constructors, the cost of passing the necessary arguments to them can
  // be significant, so we avoid constructing them in the common case (when we
  // know they will not be used).
  if (event_collector != nullptr) {
    return true;
  }

  if (profiler::ScopedAnnotation::IsEnabled()) return true;

  return profiler::TraceMe::Active(profiler::GetTFTraceMeLevel(is_expensive));
}

Status ExecutorState::ProcessSync(const NodeItem& item,
                                  OpKernelContext::Params* params,
                                  EntryVector* outputs,
                                  NodeExecStatsInterface* stats) {
  Status s;
  OpKernelContext ctx(params, item.num_outputs);
  nodestats::SetOpStart(stats);

  OpKernel* op_kernel = item.kernel;
  Device* device = immutable_state_.params().device;
  const bool is_expensive = kernel_stats_->IsExpensive(item);

  if (TF_PREDICT_FALSE(MightTrace(event_collector_, is_expensive))) {
    tracing::ScopedRegion region(tracing::EventCategory::kCompute,
                                 op_kernel->name_view());
    profiler::AnnotatedTraceMe activity(
        [&] {
          return op_kernel->TraceString(
              &ctx, /*verbose=*/profiler::TfOpDetailsEnabled());
        },
        profiler::GetTFTraceMeLevel(is_expensive));
    device->Compute(op_kernel, &ctx);
    nodestats::SetOpEnd(stats);
    s = ProcessOutputs(item, &ctx, outputs, stats);
  } else {
    // In the common case, avoid creating any tracing objects.
    if (is_expensive) {
      KernelTimer timer;
      device->Compute(op_kernel, &ctx);
      kernel_stats_->UpdateCostEstimate(item, timer.ElapsedCycles());
    } else {
      device->Compute(op_kernel, &ctx);
    }
    nodestats::SetOpEnd(stats);
    s = ProcessOutputs(item, &ctx, outputs, stats);
  }
  nodestats::SetMemory(stats, &ctx);
  return s;
}

void ExecutorState::ProcessAsync(const NodeItem& item,
                                 const OpKernelContext::Params& params,
                                 const TaggedNode& tagged_node,
                                 Entry* first_input,
                                 NodeExecStatsInterface* stats) {
  AsyncOpKernel* async_kernel = item.kernel->AsAsync();
  DCHECK(async_kernel != nullptr);
  AsyncState* state =
      new AsyncState(params, tagged_node, &item, first_input, stats);

  auto done = [this, state]() {
    Device* device = immutable_state_.params().device;
    NodeExecStatsInterface* stats = state->stats;  // Shorthand
    Entry* first_input = state->first_input;       // Shorthand

    nodestats::SetOpEnd(stats);
    EntryVector outputs;
    Status s = ProcessOutputs(*state->item, &state->ctx, &outputs, stats);
    nodestats::SetMemory(stats, &state->ctx);
    if (vlog_) {
      VLOG(2) << "Async kernel done: " << state->item->node_id << " step "
              << step_id_ << " " << SummarizeNodeDef(state->item->kernel->def())
              << (state->tagged_node.is_dead ? " is dead" : "")
              << " device: " << device->name();
    }

    // Clears inputs.
    const int num_inputs = state->item->num_inputs;
    for (int i = 0; i < num_inputs; ++i) {
      (first_input + i)->ClearVal();
    }
    FrameState* input_frame = state->tagged_node.input_frame;
    const int64 input_iter = state->tagged_node.input_iter;
    MaybeMarkCompleted(input_frame, input_iter, state->item->node_id);
    TaggedNodeSeq ready;
    if (s.ok()) {
      PropagateOutputs(state->tagged_node, state->item, &outputs, &ready);
    }
    outputs.clear();
    const bool completed = NodeDone(s, &ready, stats, nullptr);
    delete state;
    if (completed) ScheduleFinish();
  };
  nodestats::SetOpStart(stats);
  {
    profiler::AnnotatedTraceMe activity(
        [&] {
          return async_kernel->TraceString(
              &state->ctx, /*verbose=*/profiler::TfOpDetailsEnabled());
        },
        profiler::GetTFTraceMeLevel(kernel_stats_->IsExpensive(item)));
    immutable_state_.params().device->ComputeAsync(async_kernel, &state->ctx,
                                                   std::move(done));
  }
}

void ExecutorState::ProcessNoop(NodeExecStatsInterface* stats) {
  nodestats::SetOpStart(stats);
  nodestats::SetOpEnd(stats);
}

void ExecutorState::ProcessConstTensor(const NodeItem& item,
                                       EntryVector* outputs,
                                       NodeExecStatsInterface* stats) {
  nodestats::SetOpStart(stats);
  nodestats::SetOpEnd(stats);
  outputs->resize(1);
  Entry& output = (*outputs)[0];
  output.state = Entry::State::HAS_CONST_TENSOR;
  output.const_tensor = item.const_tensor;
  output.alloc_attr = item.output_attrs()[0];
}

void ExecutorState::Process(TaggedNode tagged_node, int64 scheduled_nsec) {
  profiler::TraceMe activity(
      [&] {
        return absl::StrCat("ExecutorState::Process#id=", step_id_,
                            ",iter_num=", tagged_node.input_iter, "#");
      },
      2);
  WithContext wc(context_);
  TaggedNodeSeq ready;
  TaggedNodeReadyQueue inline_ready;

  // Parameters passed to OpKernel::Compute.
  TensorValueVec inputs;
  AllocatorAttributeVec input_alloc_attrs;

  OpKernelContext::Params params;
  params.step_id = step_id_;
  // Override device's threadpool if user provides an intra_op_threadpool
  Device* device = immutable_state_.params().device;
  if (user_device_) {
    params.device = user_device_.get();
  } else {
    params.device = device;
  }
  params.log_memory = log_memory_;
  params.rendezvous = rendezvous_;
  params.collective_executor = collective_executor_;
  params.session_state = session_state_;
  params.session_handle = session_handle_;
  params.session_metadata = session_metadata_;
  params.tensor_store = tensor_store_;
  params.cancellation_manager = cancellation_manager_;
  params.call_frame = call_frame_;
  params.function_library = immutable_state_.params().function_library;
  params.resource_manager = device->resource_manager();
  params.step_container = step_container_;
  params.slice_reader_cache = slice_reader_cache_;
  params.inputs = &inputs;
  params.input_alloc_attrs = &input_alloc_attrs;
  params.runner = &runner_;
  params.run_all_kernels_inline = run_all_kernels_inline_;
  params.stats_collector = stats_collector_;
  params.inc_num_deferred_ops_function = [this]() {
    mutex_lock lock(num_deferred_ops_mu_);
    num_deferred_ops_++;
  };
  params.dec_num_deferred_ops_function = [this]() {
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
  params.op_device_context = device_context_;

  Status s;
  NodeExecStatsInterface* stats = nullptr;

  EntryVector outputs;
  bool completed = false;
  inline_ready.push_back(tagged_node);
  while (!inline_ready.empty()) {
    tagged_node = inline_ready.front();
    inline_ready.pop_front();
    const NodeItem& item = *tagged_node.node_item;
    FrameState* input_frame = tagged_node.input_frame;
    const int64 input_iter = tagged_node.input_iter;
    const int id = item.node_id;

    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      mutex_lock l(input_frame->mu);
      input_frame->GetIteration(input_iter)
          ->mark_started(immutable_state_.pending_ids()[id]);
    }

    params.track_allocations = false;
    stats = nullptr;
    if (stats_collector_ && !tagged_node.is_dead) {
      stats = stats_collector_->CreateNodeExecStats(&item.kernel->def());
      // Track allocations if and only if we are collecting statistics, and
      // `stats` object is expecting allocations to be tracked.
      params.track_allocations = stats ? stats->TrackAllocations() : false;
      nodestats::SetScheduled(stats, scheduled_nsec);
      nodestats::SetAllStart(stats);
    }

    if (vlog_) {
      VLOG(1) << "Process node: " << id << " step " << params.step_id << " "
              << SummarizeNodeDef(item.kernel->def())
              << (tagged_node.is_dead ? " is dead" : "")
              << " device: " << device->name();
    }

    Entry* input_tensors = GetInputTensors(input_frame, input_iter);
    Entry* first_input = input_tensors + item.input_start;
    outputs.clear();

    // Only execute this node if it is not dead or it is a send/recv
    // transfer node. For transfer nodes, we need to propagate the "dead"
    // bit even when the node is dead.
    bool launched_asynchronously = false;
    if (tagged_node.is_dead && !item.is_transfer_node) {
      outputs.resize(item.num_outputs);
    } else if (TF_PREDICT_FALSE(item.is_noop)) {
      ProcessNoop(stats);
    } else if (item.const_tensor != nullptr && !params.track_allocations) {
      ProcessConstTensor(item, &outputs, stats);
    } else {
      // Prepares inputs.
      bool is_input_dead = false;
      s = PrepareInputs(item, first_input, &inputs, &input_alloc_attrs,
                        &is_input_dead);
      if (!s.ok()) {
        // Clear inputs.
        const int num_inputs = item.num_inputs;
        for (int i = 0; i < num_inputs; ++i) {
          (first_input + i)->ClearVal();
        }
        MaybeMarkCompleted(input_frame, input_iter, id);
        // Continue to process the nodes in 'inline_ready'.
        completed = NodeDone(s, &ready, stats, &inline_ready);
        continue;
      }

      // Set up compute params.
      params.op_kernel = item.kernel;
      params.frame_iter = FrameAndIter(input_frame->frame_id, input_iter);
      params.is_input_dead = is_input_dead;
      params.output_attr_array = item.output_attrs();
      params.forward_from_array = item.forward_from();
      params.outputs_required_array = item.outputs_required.get();

      if (item.kernel_is_async) {
        ProcessAsync(item, params, tagged_node, first_input, stats);
        launched_asynchronously = true;
      } else {
        s = ProcessSync(item, &params, &outputs, stats);
      }
    }

    if (!launched_asynchronously) {
      if (vlog_) {
        VLOG(2) << "Synchronous kernel done: " << id << " step "
                << params.step_id << " " << SummarizeNodeDef(item.kernel->def())
                << (tagged_node.is_dead ? " is dead: " : "")
                << " device: " << device->name();
      }

      // Clears inputs.
      const int num_inputs = item.num_inputs;
      for (int i = 0; i < num_inputs; ++i) {
        (first_input + i)->ClearVal();
      }
      MaybeMarkCompleted(input_frame, input_iter, id);
      // Propagates outputs.
      if (s.ok()) {
        PropagateOutputs(tagged_node, &item, &outputs, &ready);
      }
      outputs.clear();
      if (stats) {
        scheduled_nsec = nodestats::NowInNsec();
      }
      // Postprocess.
      completed = NodeDone(s, &ready, stats, &inline_ready);
    }
  }  // while !inline_ready.empty()

  // This thread of computation is done if completed = true.
  if (completed) ScheduleFinish();
}

Status ExecutorState::PrepareInputs(const NodeItem& item, Entry* first_input,
                                    TensorValueVec* inputs,
                                    AllocatorAttributeVec* input_alloc_attrs,
                                    bool* is_input_dead) {
  inputs->clear();
  inputs->resize(item.num_inputs);
  input_alloc_attrs->clear();
  input_alloc_attrs->resize(item.num_inputs);

  *is_input_dead = false;

  bool is_merge = item.is_merge;
  for (int i = 0; i < item.num_inputs; ++i) {
    const bool expect_ref = IsRefType(item.input_type(i));
    Entry* entry = first_input + i;
    (*input_alloc_attrs)[i] = entry->alloc_attr;

    // i-th input.
    TensorValue* inp = &(*inputs)[i];

    switch (entry->state) {
      case Entry::State::NO_VALUE: {
        // Only merge and transfer nodes can have no-value inputs.
        if (!is_merge) {
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
        if (expect_ref) {
          return AttachDef(
              errors::InvalidArgument(i, "-th input expects a ref type"),
              item.kernel->def());
        }
        inp->tensor = entry->val.get();
        break;
      }

      case Entry::State::HAS_CONST_TENSOR: {
        if (expect_ref) {
          return AttachDef(
              errors::InvalidArgument(i, "-th input expects a ref type"),
              item.kernel->def());
        }
        // NOTE(mrry): This `const_cast` is necessary because `TensorValue`
        // stores a non-const `Tensor*`, and relies on the `OpKernelContext`
        // accessors making dynamic checks that prevent using an immutable
        // tensor as a mutable tensor.
        inp->tensor = const_cast<Tensor*>(entry->const_tensor);
        break;
      }

      case Entry::State::HAS_REF_TENSOR: {
        {
          tf_shared_lock ml(*entry->ref_tensor.mu);
          if (!entry->ref_tensor.tensor->IsInitialized() &&
              !item.is_initialization_op) {
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

          inp->tensor = entry->val.get();
          // The dtype of entry->ref_tensor.tensor could have been changed by
          // another operation that ran after the operation that "produced" it
          // executed, so re-validate that the type of the dereferenced tensor
          // matches the expected input type.
          if (item.input_type(i) != inp->tensor->dtype()) {
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
  return Status::OK();
}

Status ExecutorState::ProcessOutputs(const NodeItem& item, OpKernelContext* ctx,
                                     EntryVector* outputs,
                                     NodeExecStatsInterface* stats) {
  DCHECK_EQ(0, outputs->size());
  outputs->resize(item.num_outputs);

  Status s = ctx->status();
  if (!s.ok()) {
    s = AttachDef(s, item.kernel->def());
    // TODO(misard) Replace with a finer-grain enabling flag once we
    // add better optional debugging support.
    if (vlog_ && VLOG_IS_ON(1)) {
      LOG(WARNING) << this << " Compute status: " << s;
      DumpState();
    }
    if (s.code() == error::RESOURCE_EXHAUSTED) {
      if (stats_collector_) {
        string err = stats_collector_->ReportAllocsOnResourceExhausted(
            s.error_message());
        s = Status(s.code(), strings::StrCat(s.error_message(), err));
      } else {
        s = Status(
            s.code(),
            strings::StrCat(
                s.error_message(),
                "\nHint: If you want to see a list of allocated tensors when "
                "OOM happens, add report_tensor_allocations_upon_oom "
                "to RunOptions for current allocation info.\n"));
      }
    }
    return s;
  }

  for (int i = 0; i < item.num_outputs; ++i) {
    const TensorValue val = ctx->release_output(i);
    if (val.tensor == nullptr) {
      // Unless it's a Switch or a Recv, or the executor has marked the output
      // as not required, the node must produce a tensor value at i-th output.
      if (!(item.is_recv_or_switch ||
            (item.outputs_required && !item.outputs_required[i]))) {
        s.Update(errors::Internal("Missing ", i, "-th output from ",
                                  FormatNodeDefForError(item.kernel->def())));
      }
    } else {
      Entry* out = &((*outputs)[i]);

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

void ExecutorState::PropagateOutputs(const TaggedNode& tagged_node,
                                     const NodeItem* item, EntryVector* outputs,
                                     TaggedNodeSeq* ready) {
  profiler::TraceMe activity(
      [&]() {
        return strings::StrCat(
            "ExecutorPropagateOutputs#", "id=", step_id_,
            ",kernel_name=", item->kernel->name_view(),
            ",num_output_edges=", item->num_output_edges,
            ",num_output_control_edges=", item->num_output_control_edges, "#");
      },
      profiler::GetTFTraceMeLevel(/*is_expensive=*/false));

  FrameState* input_frame = tagged_node.input_frame;
  const int64 input_iter = tagged_node.input_iter;
  const bool is_dead = tagged_node.is_dead;

  // Propagates outputs along out edges, and puts newly ready nodes
  // into the ready queue.
  DCHECK(ready->empty());
  bool is_frame_done = false;
  FrameState* output_frame = input_frame;
  int64 output_iter = input_iter;

  if (!item->is_enter_exit_or_next_iter) {
    // Fast path for nodes types that don't need special handling
    DCHECK_EQ(input_frame, output_frame);
    // Normal path for most nodes
    mutex_lock l(input_frame->mu);
    output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
    is_frame_done = input_frame->DecrementOutstandingOpsLocked(
        &immutable_state_.graph_view(), input_iter, ready);
  } else if (item->is_enter) {
    FindOrCreateChildFrame(input_frame, input_iter, *item, &output_frame);
    output_iter = 0;
    {
      mutex_lock l(output_frame->mu);
      if (item->is_constant_enter) {
        // Propagate to all active iterations if this is a loop invariant.
        output_frame->AddLoopInv(item, (*outputs)[0], ready);
      } else {
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
      }
      output_frame->num_pending_inputs--;
    }
    is_frame_done = input_frame->DecrementOutstandingOps(
        &immutable_state_.graph_view(), input_iter, ready);
  } else if (item->is_exit) {
    if (is_dead) {
      mutex_lock l(input_frame->mu);
      // Stop and remember this node if it is a dead exit.
      if (input_iter == input_frame->iteration_count) {
        input_frame->dead_exits.push_back(item);
      }
      is_frame_done = input_frame->DecrementOutstandingOpsLocked(
          &immutable_state_.graph_view(), input_iter, ready);
    } else {
      output_frame = input_frame->parent_frame;
      output_iter = input_frame->parent_iter;
      {
        mutex_lock l(output_frame->mu);
        output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
      }
      is_frame_done = input_frame->DecrementOutstandingOps(
          &immutable_state_.graph_view(), input_iter, ready);
    }
  } else {
    DCHECK(item->is_next_iteration);
    mutex_lock l(input_frame->mu);
    if (is_dead) {
      // Stop the deadness propagation.
      output_frame = nullptr;
    } else {
      if (input_iter == input_frame->iteration_count &&
          input_frame->num_outstanding_iterations ==
              input_frame->max_parallel_iterations) {
        // Reached the maximum for parallel iterations.
        input_frame->next_iter_roots.push_back({item, (*outputs)[0]});
        output_frame = nullptr;
      } else {
        // If this is a new iteration, start it.
        if (input_iter == input_frame->iteration_count) {
          input_frame->IncrementIteration(&immutable_state_.graph_view(),
                                          ready);
        }
        output_iter = input_iter + 1;
      }
    }
    if (output_frame != nullptr) {
      // This is the case when node is not Enter, Exit, or NextIteration.
      DCHECK(input_frame == output_frame);
      output_frame->ActivateNodes(item, is_dead, output_iter, outputs, ready);
    }
    is_frame_done = input_frame->DecrementOutstandingOpsLocked(
        &immutable_state_.graph_view(), input_iter, ready);
  }

  // At this point, this node is completely done. We also know if the
  // completion of this node makes its frame completed.
  if (is_frame_done) {
    FrameState* parent_frame = input_frame->parent_frame;
    const int64 parent_iter = input_frame->parent_iter;
    DeleteFrame(input_frame, ready);
    if (parent_frame != nullptr) {
      // The completion of frame may cause completions in its parent frame.
      // So clean things up recursively.
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}

bool ExecutorState::NodeDone(const Status& s, TaggedNodeSeq* ready,
                             NodeExecStatsInterface* stats,
                             TaggedNodeReadyQueue* inline_ready) {
  nodestats::SetAllEnd(stats);
  if (stats) {
    if (stats_collector_) {
      stats->Done(immutable_state_.params().device->name());
    } else {
      delete stats;
    }
  }

  bool abort_run = false;
  if (!s.ok()) {
    // Some error happened. This thread of computation is done.
    mutex_lock l(mu_);
    if (status_.ok()) {
      abort_run = true;

      // If execution has been cancelled, mark any new errors as being derived.
      // This ensures any errors triggered by cancellation are marked as
      // derived.
      if (cancellation_manager_ && cancellation_manager_->IsCancelled()) {
        status_ = StatusGroup::MakeDerived(s);
      } else {
        status_ = s;
      }
    }
  }
  if (abort_run) {
    TRACEPRINTF("StartAbort: %s", s.ToString().c_str());
    if (cancellation_manager_) {
      // only log when the abort happens during the actual run time.
      auto device_name = immutable_state_.params().device->name();
      // Use VLOG instead of LOG(warning) because error status is expected when
      // the executor is run under the grappler optimization phase or when
      // iterating through a tf.data input pipeline.
      VLOG(1) << "[" << device_name << "] Executor start aborting: " << s;
    }

    if (rendezvous_) {
      rendezvous_->StartAbort(s);
    }
    if (collective_executor_) {
      collective_executor_->StartAbort(s);
    }
    if (cancellation_manager_) {
      cancellation_manager_->StartCancel();
    }
  }

  bool completed = false;
  const size_t ready_size = ready->size();
  if (ready_size == 0 || !s.ok()) {
    completed = (num_outstanding_ops_.fetch_sub(1) == 1);
  } else if (ready_size > 1) {
    num_outstanding_ops_.fetch_add(ready_size - 1, std::memory_order_relaxed);
  }

  // Schedule the ready nodes in 'ready'.
  if (s.ok()) {
    ScheduleReady(ready, inline_ready);
  }
  return completed;
}

void ExecutorState::ScheduleReady(TaggedNodeSeq* ready,
                                  TaggedNodeReadyQueue* inline_ready) {
  if (ready->empty()) return;

  int64 scheduled_nsec = 0;
  if (stats_collector_) {
    scheduled_nsec = nodestats::NowInNsec();
  }

  if (run_all_kernels_inline_) {
    if (inline_ready == nullptr) {
      // Schedule all ready kernels from a single closure. This ensure that,
      // regardless of the `runner_` implementation, all kernels will run
      // sequentially on the same thread, and thread wakeup overhead and
      // executor mutex contention will be minimized.
      runner_([this, ready = std::move(*ready), scheduled_nsec]() {
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
    if (inline_ready == nullptr) {
      // Schedule to run all the ready ops in thread pool.
      for (auto& tagged_node : *ready) {
        runner_([=]() { Process(tagged_node, scheduled_nsec); });
      }
    } else {
      for (auto& tagged_node : *ready) {
        const NodeItem& item = *tagged_node.node_item;
        if (tagged_node.is_dead || !kernel_stats_->IsExpensive(item)) {
          // Inline this inexpensive node.
          inline_ready->push_back(tagged_node);
        } else {
          if (curr_expensive_node) {
            // Dispatch to another thread since there is plenty of work to
            // do for this thread.
            runner_(std::bind(&ExecutorState::Process, this,
                              *curr_expensive_node, scheduled_nsec));
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
        runner_(std::bind(&ExecutorState::Process, this, *curr_expensive_node,
                          scheduled_nsec));
      }
    }
  }
  ready->clear();
}

inline void ExecutorState::MaybeMarkCompleted(FrameState* frame, int64 iter,
                                              const int node_id) {
  // TODO(misard) Replace with a finer-grain enabling flag once we
  // add better optional debugging support.
  if (vlog_ && VLOG_IS_ON(1)) {
    mutex_lock l(frame->mu);
    frame->GetIteration(iter)->mark_completed(
        immutable_state_.pending_ids()[node_id]);
  }
}

const Tensor* ExecutorState::GetTensorValueForDump(const Entry& input) {
  switch (input.state) {
    case Entry::State::NO_VALUE:
      return kEmptyTensor;
    case Entry::State::HAS_VALUE:
      return input.val.get();
    case Entry::State::HAS_CONST_TENSOR:
      return input.const_tensor;
    case Entry::State::HAS_REF_TENSOR:
      return input.ref_tensor.tensor;
  }
}

void ExecutorState::DumpPendingNodeState(
    const int node_id, const Entry* input_vector,
    const bool show_nodes_with_no_ready_inputs) {
  const NodeItem& node_item = *immutable_state_.graph_view().node(node_id);
  const int input_base = node_item.input_start;
  if (!show_nodes_with_no_ready_inputs) {
    bool has_ready_input = false;
    for (int i = 0; i < node_item.num_inputs; ++i) {
      const Entry& input = input_vector[input_base + i];
      const Tensor* tensor = GetTensorValueForDump(input);
      if (tensor->IsInitialized()) {
        has_ready_input = true;
        break;
      }
    }
    if (!has_ready_input) {
      return;
    }
  }
  LOG(WARNING) << "    Pending Node: " << node_item.DebugString();
  for (int i = 0; i < node_item.num_inputs; ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

void ExecutorState::DumpActiveNodeState(const int node_id,
                                        const Entry* input_vector) {
  const NodeItem& node_item = *immutable_state_.graph_view().node(node_id);
  LOG(WARNING) << "    Active Node: " << node_item.DebugString();
  const int input_base = node_item.input_start;
  for (int i = 0; i < node_item.num_inputs; ++i) {
    const Entry& input = input_vector[input_base + i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "      Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(), ">");
    } else {
      LOG(WARNING) << "      Input " << i << ": not present";
    }
  }
}

void ExecutorState::DumpIterationState(const FrameState* frame,
                                       IterationState* iteration) {
  const std::vector<const NodeItem*>* nodes = frame->nodes;
  // Dump any waiting nodes that are holding on to tensors.
  for (const NodeItem* node : *nodes) {
    PendingCounts::Handle pending_id =
        immutable_state_.pending_ids()[node->node_id];
    if (iteration->node_state(pending_id) == PendingCounts::PENDING_NOTREADY ||
        iteration->node_state(pending_id) == PendingCounts::PENDING_READY) {
      DumpPendingNodeState(node->node_id, iteration->input_tensors, false);
    }
  }
  // Then the active nodes.
  for (const NodeItem* node : *nodes) {
    PendingCounts::Handle pending_id =
        immutable_state_.pending_ids()[node->node_id];
    if (iteration->node_state(pending_id) == PendingCounts::STARTED) {
      DumpActiveNodeState(node->node_id, iteration->input_tensors);
    }
  }
  // Show all input tensors in use.
  const int total_input_tensors = frame->total_input_tensors;
  size_t total_bytes = 0;
  for (int i = 0; i < total_input_tensors; ++i) {
    const Entry& input = iteration->input_tensors[i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor->IsInitialized()) {
      LOG(WARNING) << "    Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(),
                          ", bytes: ", tensor->TotalBytes(), ">");
      total_bytes += tensor->TotalBytes();
    }
  }
  LOG(WARNING) << "    Total bytes " << total_bytes;
}

void ExecutorState::DumpState() {
  mutex_lock l(mu_);
  if (!dumped_on_error_) {
    LOG(WARNING) << "Dumping state";
    for (auto& frame : outstanding_frames_) {
      LOG(WARNING) << frame.first;
      FrameState* frame_state = frame.second;
      frame_state->DumpIterationState(this);
    }
    dumped_on_error_ = true;
  }
}

void ExecutorState::ScheduleFinish() {
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

void ExecutorState::Finish() {
  mu_.lock();
  auto status = status_;
  auto done_cb = std::move(done_cb_);
  auto runner = std::move(runner_);
  mu_.unlock();
  int64 step_id = step_id_;
  CHECK(done_cb != nullptr);
  Device* device = immutable_state_.params().device;

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
      if (collective_executor_) {
        collective_executor_->StartAbort(status);
      }
      if (cancellation_manager_) {
        cancellation_manager_->StartCancel();
      }
    }
    delete this;
    runner([step_id, status, done_cb = std::move(done_cb)]() {
      profiler::TraceMe traceme(
          [&] {
            return absl::StrCat("ExecutorDoneCallback#id=", step_id, "#");
          },
          2);
      done_cb(status);
    });
    return;
  }

  if (sync_on_finish_ && status.ok()) {
    // Block until the device has finished all queued operations. For
    // devices like GPUs that continue to execute Ops after their Compute
    // methods have completed, this ensures that control is not returned to
    // the user until the step (and its side-effects) has actually completed.
    device->Sync([this, step_id, runner = std::move(runner),
                  done_cb = std::move(done_cb)](const Status& status) mutable {
      delete this;
      runner([step_id, status, done_cb = std::move(done_cb)]() {
        profiler::TraceMe traceme(
            [&] {
              return absl::StrCat("ExecutorDoneCallback#id=", step_id, "#");
            },
            2);
        done_cb(status);
      });
    });
  } else {
    delete this;
    runner([step_id, status, done_cb = std::move(done_cb)]() {
      profiler::TraceMe traceme(
          [&] {
            return absl::StrCat("ExecutorDoneCallback#id=", step_id, "#");
          },
          2);
      done_cb(status);
    });
  }
}

void ExecutorState::FindOrCreateChildFrame(FrameState* frame, int64 iter,
                                           const NodeItem& node_item,
                                           FrameState** child) {
  // Get the child frame name.
  AttrSlice attrs(node_item.kernel->def());
  const string& enter_name = GetNodeAttrString(attrs, "frame_name");
  DCHECK(!enter_name.empty()) << "Could not find \"frame_name\" attr in node "
                              << node_item.kernel->name();
  const string child_name = MakeFrameName(frame, iter, enter_name);

  {
    mutex_lock executor_lock(mu_);
    auto it = outstanding_frames_.find(child_name);
    if (it != outstanding_frames_.end()) {
      *child = it->second;
      return;
    }
  }

  // Need to create a new frame instance.
  // Note that this new frame instance is created without any locks.
  if (vlog_) VLOG(2) << "Create frame: " << child_name;

  int parallel_iters;
  bool found_parallel_iters =
      TryGetNodeAttr(attrs, "parallel_iterations", &parallel_iters);
  DCHECK(found_parallel_iters)
      << "Could not find \"parallel_iterations\" attr in node "
      << node_item.kernel->name();
  FrameState* temp = new FrameState(immutable_state_, parallel_iters);
  temp->frame_name = child_name;
  temp->frame_id = Hash64(child_name);
  temp->parent_frame = frame;
  temp->parent_iter = iter;
  temp->InitializeFrameInfo(enter_name);

  // Initialize iteration 0.
  {
    mutex_lock l(temp->mu);
    temp->SetIteration(
        0, new IterationState(temp->pending_counts, temp->total_input_tensors));
  }

  {
    mutex_lock executor_lock(mu_);
    auto it = outstanding_frames_.find(child_name);
    if (it != outstanding_frames_.end()) {
      *child = it->second;
    } else {
      mutex_lock frame_lock(frame->mu);
      frame->GetIteration(iter)->outstanding_frame_count++;
      outstanding_frames_[child_name] = temp;
      *child = temp;
      temp = nullptr;
    }
  }
  delete temp;  // Not used so delete it.
}

void ExecutorState::DeleteFrame(FrameState* frame, TaggedNodeSeq* ready) {
  // First, propagate dead_exits (if any) to the parent frame.
  FrameState* parent_frame = frame->parent_frame;
  const int64 parent_iter = frame->parent_iter;
  if (parent_frame != nullptr) {
    mutex_lock parent_frame_lock(parent_frame->mu);
    // Propagate all the dead exits to the parent frame.
    mutex_lock this_frame_lock(frame->mu);

    for (const NodeItem* item : frame->dead_exits) {
      auto parent_iter_state = parent_frame->GetIteration(parent_iter);

      auto maybe_add_to_ready = [&](const NodeItem& dst_item, bool dst_ready,
                                    bool dst_dead) {
        if (dst_ready) {
          if (dst_item.is_control_trigger) dst_dead = false;
          ready->emplace_back(&dst_item, parent_frame, parent_iter, dst_dead);
          parent_iter_state->outstanding_ops++;
        }
      };

      auto propagate_to_non_merge = [&](PendingCounts::Handle dst_pending_id) {
        parent_iter_state->increment_dead_count(dst_pending_id);
        return parent_iter_state->decrement_pending(dst_pending_id, 1) == 0;
      };

      for (const EdgeInfo& e : item->output_edges()) {
        const NodeItem& dst_item =
            *immutable_state_.graph_view().node(e.dst_id);
        const auto dst_pending_id = immutable_state_.pending_ids()[e.dst_id];

        bool dst_dead = true;
        bool dst_ready;
        // We know this is a dead input to dst.
        if (dst_item.is_merge) {
          parent_iter_state->increment_dead_count(dst_pending_id);
          const int dead_cnt = parent_iter_state->dead_count(dst_pending_id);
          dst_dead = (dead_cnt == dst_item.num_inputs);
          dst_ready =
              (parent_iter_state->pending(dst_pending_id) == 1) && dst_dead;
        } else {
          dst_ready = propagate_to_non_merge(dst_pending_id);
        }
        maybe_add_to_ready(dst_item, dst_ready, dst_dead);
      }

      for (const ControlEdgeInfo& e : item->output_control_edges()) {
        const NodeItem& dst_item =
            *immutable_state_.graph_view().node(e.dst_id);
        const auto dst_pending_id = immutable_state_.pending_ids()[e.dst_id];

        bool dst_dead;
        bool dst_ready;
        // We know this is a dead input to dst.
        if (dst_item.is_merge) {
          parent_iter_state->decrement_pending(dst_pending_id, 2);
          int count = parent_iter_state->pending(dst_pending_id);
          int dead_cnt = parent_iter_state->dead_count(dst_pending_id);
          dst_dead = (dead_cnt == dst_item.num_inputs);
          dst_ready = (count == 0) || ((count == 1) && dst_dead);
        } else {
          dst_dead = true;
          dst_ready = propagate_to_non_merge(dst_pending_id);
        }
        maybe_add_to_ready(dst_item, dst_ready, dst_dead);
      }
    }
  }

  // Delete the frame.
  const string& frame_name = frame->frame_name;
  if (vlog_) VLOG(2) << "Delete frame " << frame_name;
  {
    mutex_lock executor_lock(mu_);
    outstanding_frames_.erase(frame_name);
  }
  delete frame;
}

void ExecutorState::CleanupFramesIterations(FrameState* frame, int64 iter,
                                            TaggedNodeSeq* ready) {
  bool is_frame_done = false;
  {
    mutex_lock frame_lock(frame->mu);
    frame->GetIteration(iter)->outstanding_frame_count--;
    is_frame_done =
        frame->CleanupIterations(&immutable_state_.graph_view(), iter, ready);
  }
  if (is_frame_done) {
    FrameState* parent_frame = frame->parent_frame;
    const int64 parent_iter = frame->parent_iter;
    DeleteFrame(frame, ready);
    if (parent_frame != nullptr) {
      // The completion of frame may cause completions in its parent frame.
      // So clean things up recursively.
      CleanupFramesIterations(parent_frame, parent_iter, ready);
    }
  }
}

void ExecutorState::FrameState::ActivateNodesFastPath(const NodeItem* item,
                                                      const bool is_dead,
                                                      int64 iter,
                                                      EntryVector* outputs,
                                                      TaggedNodeSeq* ready) {
  // If we know that none of the item's edge destinations require special
  // handling (i.e. none of the nodes is a merge or control trigger node), we
  // can take a fast path that avoids accessing the destination NodeItem.
  const GraphView& gview = immutable_state.graph_view();
  IterationState* iter_state = GetIteration(iter);

// Add dst to the ready queue if it's ready
//
// NOTE(mrry): Use a macro here instead of a lambda, because this method is
// performance-critical and we need to ensure that the code is inlined.
#define MAYBE_ADD_TO_READY(dst_id, adjust_result)    \
  do {                                               \
    if (!adjust_result.any_pending) {                \
      const NodeItem* dst_item = gview.node(dst_id); \
      TaggedNode& t = ready->emplace_back();         \
      t.node_item = dst_item;                        \
      t.input_frame = this;                          \
      t.input_iter = iter;                           \
      t.is_dead = adjust_result.any_dead;            \
      iter_state->outstanding_ops++;                 \
    }                                                \
  } while (0);

  Entry* input_tensors = iter_state->input_tensors;

  for (const EdgeInfo& e : item->output_edges()) {
    const int dst_id = e.dst_id;
    const PendingCounts::Handle dst_pending_id =
        immutable_state.pending_ids()[dst_id];
    const int src_slot = e.output_slot;

    const bool increment_dead =
        (is_dead || ((*outputs)[src_slot].state == Entry::State::NO_VALUE));
    const PendingCounts::AdjustResult adjust_result =
        iter_state->adjust_for_activation(dst_pending_id, increment_dead);
    const int dst_loc = e.input_slot;
    if (e.is_last) {
      input_tensors[dst_loc] = std::move((*outputs)[src_slot]);
    } else {
      input_tensors[dst_loc] = (*outputs)[src_slot];
    }
    MAYBE_ADD_TO_READY(dst_id, adjust_result);
  }

  for (const ControlEdgeInfo& e : item->output_control_edges()) {
    const int dst_id = e.dst_id;
    const PendingCounts::Handle dst_pending_id =
        immutable_state.pending_ids()[dst_id];
    const PendingCounts::AdjustResult adjust_result =
        iter_state->adjust_for_activation(dst_pending_id, is_dead);
    MAYBE_ADD_TO_READY(dst_id, adjust_result);
  }
#undef MAYBE_ADD_TO_READY
}

void ExecutorState::FrameState::ActivateNodesSlowPath(const NodeItem* item,
                                                      const bool is_dead,
                                                      int64 iter,
                                                      EntryVector* outputs,
                                                      TaggedNodeSeq* ready) {
  // If any of the edge destinations is a merge or a control trigger node,
  // we need to read each destination NodeItem to determine what action
  // to take.
  const GraphView& gview = immutable_state.graph_view();
  IterationState* iter_state = GetIteration(iter);

  auto maybe_add_to_ready = [&](int dst_id, const NodeItem* dst_item,
                                bool dst_ready, bool dst_dead) {
    // Add dst to the ready queue if it's ready
    if (dst_ready) {
      if (dst_item->is_control_trigger) dst_dead = false;
      ready->emplace_back(dst_item, this, iter, dst_dead);
      iter_state->outstanding_ops++;
    }
  };

  Entry* input_tensors = iter_state->input_tensors;

  for (const EdgeInfo& e : item->output_edges()) {
    const int dst_id = e.dst_id;
    const NodeItem* dst_item = gview.node(dst_id);
    const PendingCounts::Handle dst_pending_id =
        immutable_state.pending_ids()[dst_id];
    const int src_slot = e.output_slot;

    bool dst_dead = false;
    bool dst_ready = false;
    bool dst_need_input = true;

    if (dst_item->is_merge) {
      // A merge node is ready if all control inputs have arrived and either
      // a) a live data input becomes available or b) all data inputs are
      // dead. For Merge, pending's LSB is set iff a live data input has
      // arrived.
      if ((*outputs)[src_slot].state != Entry::State::NO_VALUE) {
        // This is a live data input.
        int count = iter_state->pending(dst_pending_id);
        iter_state->mark_live(dst_pending_id);
        // Only the first live edge sets the input and (potentially)
        // triggers execution. The low bit of count is set if and
        // only if no live input has been used yet (mark_live clears
        // it). The node should be started if and only if this is
        // the first live input and there are no pending control
        // edges, i.e. count == 1.
        dst_ready = (count == 1);
        dst_need_input = ((count & 0x1) == 1);
      } else {
        // This is a dead data input. Note that dst_node is dead if node is
        // a dead enter. We need this to handle properly a while loop on
        // the untaken branch of a conditional.
        // TODO(yuanbyu): This is a bit hacky, but a good solution for
        // now.
        iter_state->increment_dead_count(dst_pending_id);
        const int dead_cnt = iter_state->dead_count(dst_pending_id);
        dst_dead = (dead_cnt == dst_item->num_inputs) || item->is_enter;
        dst_ready = (iter_state->pending(dst_pending_id) == 1) && dst_dead;
        dst_need_input = false;
      }
    } else {
      // Handle all other (non-merge) nodes.
      const bool increment_dead =
          (is_dead || ((*outputs)[src_slot].state == Entry::State::NO_VALUE));
      const PendingCounts::AdjustResult adjust_result =
          iter_state->adjust_for_activation(dst_pending_id, increment_dead);
      dst_dead = adjust_result.any_dead;
      dst_ready = !adjust_result.any_pending;
    }

    if (dst_need_input) {
      const int dst_loc = e.input_slot;
      if (e.is_last) {
        input_tensors[dst_loc] = std::move((*outputs)[src_slot]);
      } else {
        input_tensors[dst_loc] = (*outputs)[src_slot];
      }
    }

    maybe_add_to_ready(dst_id, dst_item, dst_ready, dst_dead);
  }

  for (const ControlEdgeInfo& e : item->output_control_edges()) {
    const int dst_id = e.dst_id;
    const NodeItem* dst_item = gview.node(dst_id);
    const PendingCounts::Handle dst_pending_id =
        immutable_state.pending_ids()[dst_id];

    bool dst_dead;
    bool dst_ready;
    if (dst_item->is_merge) {
      // A merge node is ready if all control inputs have arrived and either
      // a) a live data input becomes available or b) all data inputs are
      // dead. For Merge, pending's LSB is set iff a live data input has
      // arrived.
      iter_state->decrement_pending(dst_pending_id, 2);
      int count = iter_state->pending(dst_pending_id);
      int dead_cnt = iter_state->dead_count(dst_pending_id);
      dst_dead = (dead_cnt == dst_item->num_inputs);
      dst_ready = (count == 0) || ((count == 1) && dst_dead);
    } else {
      // Handle all other (non-merge) nodes.
      const PendingCounts::AdjustResult adjust_result =
          iter_state->adjust_for_activation(dst_pending_id, is_dead);
      dst_dead = adjust_result.any_dead;
      dst_ready = !adjust_result.any_pending;
    }
    maybe_add_to_ready(dst_id, dst_item, dst_ready, dst_dead);
  }
}

void ExecutorState::FrameState::ActivateNodes(const NodeItem* item,
                                              const bool is_dead, int64 iter,
                                              EntryVector* outputs,
                                              TaggedNodeSeq* ready) {
  if (TF_PREDICT_FALSE(item->is_any_consumer_merge_or_control_trigger)) {
    ActivateNodesSlowPath(item, is_dead, iter, outputs, ready);
  } else {
    ActivateNodesFastPath(item, is_dead, iter, outputs, ready);
  }
}

void ExecutorState::FrameState::ActivateNexts(const GraphView* gview,
                                              int64 iter,
                                              TaggedNodeSeq* ready) {
  // Propagate the deferred NextIteration nodes to the new iteration.
  for (auto& node_entry : next_iter_roots) {
    const NodeItem* item = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = entry.state == Entry::State::NO_VALUE;
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, iter, &outputs, ready);
  }
  next_iter_roots.clear();
}

void ExecutorState::FrameState::ActivateLoopInvs(const GraphView* gview,
                                                 int64 iter,
                                                 TaggedNodeSeq* ready) {
  // Propagate loop invariants to the new iteration.
  for (auto& node_entry : inv_values) {
    const NodeItem* item = node_entry.first;
    const Entry& entry = node_entry.second;
    const bool is_dead = entry.state == Entry::State::NO_VALUE;
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, iter, &outputs, ready);
  }
}

void ExecutorState::FrameState::AddLoopInv(const NodeItem* item,
                                           const Entry& entry,
                                           TaggedNodeSeq* ready) {
  // Store this value.
  inv_values.push_back({item, entry});

  // Make this value available to all iterations.
  const bool is_dead = entry.state == Entry::State::NO_VALUE;
  for (int i = 0; i <= iteration_count; ++i) {
    EntryVector outputs{entry};
    ActivateNodes(item, is_dead, i, &outputs, ready);
  }
}

bool ExecutorState::FrameState::IsIterationDone(int64 iter) {
  IterationState* iter_state = GetIteration(iter);
  if (iter_state->outstanding_ops == 0 &&
      iter_state->outstanding_frame_count == 0) {
    if (iter == 0) {
      // The enclosing frame has no pending input.
      return num_pending_inputs == 0;
    } else {
      // The preceding iteration is deleted (and therefore done).
      return (GetIteration(iter - 1) == nullptr);
    }
  }
  return false;
}

void ExecutorState::FrameState::IncrementIteration(const GraphView* gview,
                                                   TaggedNodeSeq* ready) {
  iteration_count++;
  const int64 next_iter = iteration_count;

  // Initialize the next iteration.
  IterationState* iter_state =
      new IterationState(pending_counts, total_input_tensors);
  SetIteration(next_iter, iter_state);
  num_outstanding_iterations++;
  dead_exits.clear();

  // Activate the successors of the deferred roots in the new iteration.
  ActivateNexts(gview, next_iter, ready);

  // Activate the loop invariants in the new iteration.
  ActivateLoopInvs(gview, next_iter, ready);
}

bool ExecutorState::FrameState::CleanupIterations(const GraphView* gview,
                                                  int64 iter,
                                                  TaggedNodeSeq* ready) {
  int64 curr_iter = iter;
  while (curr_iter <= iteration_count && IsIterationDone(curr_iter)) {
    // Delete the iteration curr_iter.
    delete GetIteration(curr_iter);
    SetIteration(curr_iter, nullptr);
    --num_outstanding_iterations;
    ++curr_iter;

    // When one iteration is completed, we check for deferred iteration,
    // and start it if there is one.
    if (!next_iter_roots.empty()) {
      IncrementIteration(gview, ready);
    }
  }
  return IsFrameDone();
}

void ExecutorImpl::RunAsync(const Args& args, DoneCallback done) {
  (new ExecutorState(args, immutable_state_, &kernel_stats_))
      ->RunAsync(std::move(done));
}

}  // namespace

Status NewLocalExecutor(const LocalExecutorParams& params, const Graph& graph,
                        Executor** executor) {
  ExecutorImpl* impl = new ExecutorImpl(params);
  const Status s = impl->Initialize(graph);
  if (s.ok()) {
    *executor = impl;
  } else {
    delete impl;
  }
  return s;
}

Status CreateNonCachedKernel(Device* device, FunctionLibraryRuntime* flib,
                             const std::shared_ptr<const NodeProperties>& props,
                             int graph_def_version, OpKernel** kernel) {
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
    Status NewExecutor(const LocalExecutorParams& params, const Graph& graph,
                       std::unique_ptr<Executor>* out_executor) override {
      Executor* ret = nullptr;
      TF_RETURN_IF_ERROR(NewLocalExecutor(params, std::move(graph), &ret));
      out_executor->reset(ret);
      return Status::OK();
    }
  };
};
static DefaultExecutorRegistrar registrar;

}  // namespace

}  // namespace tensorflow
