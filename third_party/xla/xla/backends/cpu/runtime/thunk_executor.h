/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_THUNK_EXECUTOR_H_
#define XLA_BACKENDS_CPU_RUNTIME_THUNK_EXECUTOR_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <queue>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

namespace internal {
// Clang does not allow defining a nested struct with member initializer, as
// a workaround we define a struct in internal namespace and create an alias.
struct ThunkExecutorOptions {
  enum class ReadyQueueType { kFifo, kLifo, kPriority };

  // If all thunks in a sequence use buffers of size less than or equal to the
  // given threshold, we mark execution as sequential, as concurrency overheads
  // will likely dominate the overall execution time.
  size_t execute_sequential_buffer_threshold = 512;

  // If thunk sequence length is less than or equal to the given threshold, we
  // mark execution as sequential, as concurrency overheads will likely dominate
  // the overall execution time.
  size_t execute_sequential_num_thunks_threshold = 8;

  // The type of a queue for ready thunks.
  ReadyQueueType ready_queue_type = ReadyQueueType::kFifo;
};
}  // namespace internal

// A dataflow-style (run when ready) executor for a ThunkSequence that depends
// on buffer uses to build a DAG defining execution order. At run time executes
// thunks concurrently in a given thread pool.
class ThunkExecutor {
 public:
  using BufferUses = Thunk::BufferUses;
  using ResourceUses = Thunk::ResourceUses;
  using ExecuteEvent = Thunk::ExecuteEvent;
  using Options = internal::ThunkExecutorOptions;

  // Nodes identified by their index in the captured ThunkSequence.
  using NodeId = int32_t;

  static constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::min();

  ThunkExecutor(ThunkExecutor&&) = default;
  ThunkExecutor& operator=(ThunkExecutor&&) = default;

  static absl::StatusOr<ThunkExecutor> Create(
      ThunkSequence thunk_sequence, const Options& options = Options());

  // We store all `in_edges` and `out_edges` referenced by the `NodeDef` inside
  // large vectors to optimize for data locality on a hot path.
  using NodesEdges = std::vector<NodeId>;

  // NodeDef defines an execution order for all thunks in a sequence.
  struct NodeDef {
    NodeId id = kInvalidNodeId;
    int64_t priority = 0;
    absl::Span<const NodeId> in_edges;
    absl::Span<const NodeId> out_edges;
  };

  // A NodeDef builder to collect all in-edges and out-edges before constructing
  // a NodeDef. We use it at ThunkExecutor creation time when we don't know how
  // many in-edges and out-edges we have in total.
  struct NodeDefBuilder {
    NodeId id = kInvalidNodeId;
    int64_t priority = 0;
    std::vector<NodeId> in_edges;
    std::vector<NodeId> out_edges;
  };

  // Executes the thunk sequence using the prepared dataflow graph. Executor
  // uses runner to execute ready tasks concurrently. If runner is not provided,
  // executes all tasks in the caller thread.
  //
  // Returned execute event becomes ready when all thunks completed execution.
  // If any of the thunks failed, the event will be in error state.
  tsl::AsyncValueRef<ExecuteEvent> Execute(const Thunk::ExecuteParams& params);

  const ThunkSequence& thunk_sequence() const { return thunk_sequence_; }

  absl::Span<const NodeDef> nodes_defs() const { return nodes_defs_; }
  const NodeDef& node_def(NodeId id) const { return nodes_defs_[id]; }

  absl::Span<const NodeId> source() const { return source_; }
  absl::Span<const NodeId> sink() const { return sink_; }

  BufferUses buffer_uses() const { return thunk_sequence_.buffer_uses(); }
  ResourceUses resource_uses() const { return thunk_sequence_.resource_uses(); }

  std::string ToString() const;

  bool is_sequential() const { return is_sequential_; }

  // A ready queue that executes nodes in FIFO order.
  class FifoReadyQueue {
   public:
    explicit FifoReadyQueue(absl::Span<const NodeId> ready_nodes);

    void Push(NodeId id);

    NodeId Pop();
    FifoReadyQueue PopHalf();

    size_t Size() const;
    bool Empty() const;

    FifoReadyQueue CreateEmptyReadyQueue() const;

   private:
    absl::InlinedVector<NodeId, 8> queue_;
    size_t head_ = 0;
  };

  // A ready queue that executes nodes in LIFO order.
  class LifoReadyQueue {
   public:
    explicit LifoReadyQueue(absl::Span<const NodeId> ready_nodes);

    void Push(NodeId id);

    NodeId Pop();
    LifoReadyQueue PopHalf();

    size_t Size() const;
    bool Empty() const;

    LifoReadyQueue CreateEmptyReadyQueue() const;

   private:
    absl::InlinedVector<NodeId, 8> queue_;
  };

  // A ready queue that executes nodes sorted by NodeDef priority.
  class PriorityReadyQueue {
   public:
    PriorityReadyQueue(absl::Span<const NodeDef> nodes_defs,
                       absl::Span<const NodeId> ready_nodes);

    void Push(NodeId id);

    NodeId Pop();
    PriorityReadyQueue PopHalf();

    size_t Size() const;
    bool Empty() const;

    PriorityReadyQueue CreateEmptyReadyQueue() const;

   private:
    struct Compare {
      bool operator()(NodeId a, NodeId b) const {
        return nodes_defs[a].priority < nodes_defs[b].priority;
      }
      absl::Span<const NodeDef> nodes_defs;
    };

    using InlinedPriorityQueue =
        std::priority_queue<NodeId, absl::InlinedVector<NodeId, 8>, Compare>;

    absl::Span<const NodeDef> nodes_defs_;
    InlinedPriorityQueue queue_;
  };

 private:
  // Align all atomic counters to a cache line boundary to avoid false
  // sharing between multiple worker threads.
  static constexpr size_t kAtomicAlignment =
#if defined(__cpp_lib_hardware_interference_size)
      std::hardware_destructive_interference_size;
#else
      64;
#endif

  // A struct to keep the state of a running ThunkExecutor.
  struct ExecuteState {
    // At run time NodeDef instantiated as a Node with an atomic counter that
    // drops to zero when all `in_edges` are ready.
    struct Node {
      explicit Node(const NodeDef& node_def);

      alignas(kAtomicAlignment) std::atomic<int64_t> counter;
      absl::Span<const NodeId> out_edges;
    };

    static_assert(std::is_trivially_destructible_v<Node>,
                  "Node must be trivially destructible");

    // We use indirection via NodeStorage to be able to allocate uninitialized
    // memory and do not pay the cost of default initializing all nodes.
    struct NodeStorage {
      alignas(Node) std::byte data[sizeof(Node)];
    };

    ExecuteState(ThunkExecutor* executor, Thunk::TaskRunner* runner);

    Node& node(NodeId id) {
      DCHECK_LT(id, nodes.size()) << "Node id is out of bounds";
      return *reinterpret_cast<Node*>(&nodes.data()[id]);
    }

    ThunkExecutor* executor;
    Thunk::TaskRunner* runner;

    // Note: using alignas(Node) here instead of in NodeStorage does not work:
    // `nodes` would be aligned, but not its elements.
    absl::FixedArray<NodeStorage> nodes;
    tsl::AsyncValueRef<ExecuteEvent> execute_event;

    // Once the number of pending sink nodes drops to zero, the execution is
    // completed and we set `execute_event` as concrete or error.
    alignas(kAtomicAlignment) std::atomic<int64_t> pending_sink_nodes;

    // We store the first error from failed thunks in `abort_status` and at the
    // end of execution the executor forwards it via the `execute_event`.
    alignas(kAtomicAlignment) std::atomic<bool> abort;
    absl::Mutex abort_mutex;
    absl::Status abort_status ABSL_GUARDED_BY(abort_mutex);
  };

  ThunkExecutor(ThunkSequence thunk_sequence, NodesEdges nodes_in_edges,
                NodesEdges nodes_out_edges, std::vector<NodeDef> nodes_defs,
                const Options& options);

  // Executes given `thunk` with `params` and adds tracing annotation to capture
  // start and end events for profiling.
  static tsl::AsyncValueRef<ExecuteEvent> TracedExecute(
      Thunk& thunk, const Thunk::ExecuteParams& params);

  // Executes thunks sequentially starting from the first thunk in the
  // sequence.
  tsl::AsyncValueRef<ExecuteEvent> ExecuteSequential(
      const Thunk::ExecuteParams& params);

  // Resumes sequential thunk execution starting from the given index.
  using ThunkIterator = typename ThunkSequence::iterator;
  void ResumeExecuteSequential(ThunkIterator it,
                               const Thunk::ExecuteParams& params,
                               tsl::AsyncValueRef<ExecuteEvent> event);

  // Executes nodes in the ready queue with given thunk parameters.
  template <typename ReadyQueue>
  void Execute(ExecuteState* state, const Thunk::ExecuteParams& params,
               ReadyQueue ready_queue, Thunk::ExecuteSession::Lock lock);

  // Splits ready queue starting from `start_index` into ThunkExecutor tasks and
  // offloads them to the task runner.
  template <typename ReadyQueue>
  void SplitReadyQueue(ExecuteState* state, const Thunk::ExecuteParams& params,
                       ReadyQueue& ready_queue, int64_t split_threshold);

  // Processes out edges of a completed `node` and updates `ready_queue` with
  // nodes that are ready to execute. If `node_event` is in error state, aborts
  // the execution and records the error status to forward it to the caller.
  template <typename ReadyQueue>
  void ProcessOutEdges(ExecuteState* state,
                       tsl::AsyncValuePtr<Thunk::ExecuteEvent> node_event,
                       ExecuteState::Node& node, ReadyQueue& ready_queue);

  // Converts a vector of NodeDefBuilder to a tuple of NodesEdges and a vector
  // of NodeDef.
  static std::tuple<NodesEdges, NodesEdges, std::vector<NodeDef>>
  CreateNodeDefs(std::vector<NodeDefBuilder> builders);

  // Runs a transitive reduction on the NodeDefBuilder graph to remove redundant
  // edges, and updates nodes priorities. Returns the number of removed edges.
  //
  // See: https://en.wikipedia.org/wiki/Transitive_reduction
  static int64_t RunTransitiveReductionAndUpdatePriorities(
      absl::Span<NodeDefBuilder> builders);

  ThunkSequence thunk_sequence_;
  Options options_;

  int64_t num_thunks_;

  NodesEdges nodes_in_edges_;   // `in_edges` referenced by `nodes_defs_`
  NodesEdges nodes_out_edges_;  // `out_edges` referenced by `nodes_defs_`
  std::vector<NodeDef> nodes_defs_;

  std::vector<NodeId> source_;
  std::vector<NodeId> sink_;

  // If NodeDef graph dependency structure is sequential and does not have any
  // opportunities for executing thunks concurrently, we skip the expensive
  // async execution and simply run thunks in the `thunk_sequence_` one by one.
  bool is_sequential_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_THUNK_EXECUTOR_H_
