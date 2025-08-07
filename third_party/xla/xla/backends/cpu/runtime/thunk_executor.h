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
#include <new>
#include <queue>
#include <string>
#include <type_traits>
#include <utility>
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
#include "xla/runtime/execution_graph.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla::cpu {

// A dataflow-style (run when ready) executor for a ThunkSequence that depends
// on buffer uses to build a DAG defining execution order. At run time executes
// thunks concurrently in a given thread pool.
class ThunkExecutor {
 public:
  using BufferUses = Thunk::BufferUses;
  using ResourceUses = Thunk::ResourceUses;
  using ExecuteEvent = Thunk::ExecuteEvent;

  ThunkExecutor(ThunkExecutor&&) = default;
  ThunkExecutor& operator=(ThunkExecutor&&) = default;

  struct Options {
    enum class ReadyQueueType { kFifo, kLifo, kPriority };

    // If all thunks in a sequence use buffers of size less than or equal to the
    // given threshold, we mark execution as sequential, as concurrency
    // overheads will likely dominate the overall execution time.
    size_t execute_sequential_buffer_threshold = 512;

    // If thunk sequence length is less than or equal to the given threshold, we
    // mark execution as sequential, as concurrency overheads will likely
    // dominate the overall execution time.
    size_t execute_sequential_num_thunks_threshold = 8;

    // The type of a queue for ready thunks.
    ReadyQueueType ready_queue_type = ReadyQueueType::kFifo;

    // Flag denoting whether the executor is nested within another executor.
    bool is_nested_executor = true;
  };

  static absl::StatusOr<ThunkExecutor> Create(ThunkSequence thunk_sequence,
                                              const Options& options);

  static absl::StatusOr<ThunkExecutor> Create(ThunkSequence thunk_sequence) {
    return Create(std::move(thunk_sequence), Options());
  }

  // Executes the thunk sequence using the prepared dataflow graph. Executor
  // uses runner to execute ready tasks concurrently. If runner is not provided,
  // executes all tasks in the caller thread.
  //
  // Returned execute event becomes ready when all thunks completed execution.
  // If any of the thunks failed, the event will be in error state.
  tsl::AsyncValueRef<ExecuteEvent> Execute(const Thunk::ExecuteParams& params);

  const ThunkSequence& thunk_sequence() const { return thunk_sequence_; }

  BufferUses buffer_uses() const { return thunk_sequence_.buffer_uses(); }
  ResourceUses resource_uses() const { return thunk_sequence_.resource_uses(); }

  std::string ToString() const;

  bool is_sequential() const { return is_sequential_; }

  // We use underlying execution graph nodes to index into the thunk sequence.
  using NodeId = ExecutionGraph::NodeId;
  using NodeDef = ExecutionGraph::NodeDef;
  using NodeEdge = ExecutionGraph::NodeEdge;

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
      absl::Span<const NodeEdge> out_edges;
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

    // Once the number of pending nodes drops to zero, the execution is
    // completed and we set `execute_event` as concrete or error.
    alignas(kAtomicAlignment) std::atomic<int64_t> pending_nodes;

    // We store the first error from failed thunks in `abort_status` and at the
    // end of execution the executor forwards it via the `execute_event`.
    alignas(kAtomicAlignment) std::atomic<bool> abort;
    absl::Mutex abort_mutex;
    absl::Status abort_status ABSL_GUARDED_BY(abort_mutex);
  };

  ThunkExecutor(ThunkSequence thunk_sequence, ExecutionGraph execution_graph,
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

  // Processes out edges of a scheduled `node` and updates `ready_queue` with
  // nodes that are ready to execute. Returns true if `node` had any scheduling
  // edges, and pending nodes counter was incremented, and must be dropped when
  // `node` is completed.
  template <typename ReadyQueue>
  bool ProcessScheduledOutEdges(ExecuteState* state, ExecuteState::Node& node,
                                ReadyQueue& ready_queue);

  // Processes out edges of a completed `node` and updates `ready_queue` with
  // nodes that are ready to execute. If `node_event` is in error state, aborts
  // the execution and records the error status to forward it to the caller. We
  // can combine processing of scheduling edges with processing of completed
  // edges, if thunk completes execution in the caller thread.
  template <bool process_scheduling_edges, typename ReadyQueue>
  void ProcessCompletedOutEdges(
      ExecuteState* state, tsl::AsyncValuePtr<Thunk::ExecuteEvent> node_event,
      ExecuteState::Node& node, ReadyQueue& ready_queue,
      bool drop_pending_nodes);

  ThunkSequence thunk_sequence_;
  ExecutionGraph execution_graph_;
  Options options_;

  int64_t num_thunks_;

  // In addition to the execution graph sequential ordering property, we use
  // heuristics to use sequential execution for sequences of small thunks where
  // async execution overhead will likely dominate the overall execution time.
  bool is_sequential_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_THUNK_EXECUTOR_H_
