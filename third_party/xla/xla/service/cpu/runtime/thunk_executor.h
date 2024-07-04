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

#ifndef XLA_SERVICE_CPU_RUNTIME_THUNK_EXECUTOR_H_
#define XLA_SERVICE_CPU_RUNTIME_THUNK_EXECUTOR_H_

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/service/cpu/runtime/thunk.h"
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

  // Nodes identified by their index in the captured ThunkSequence.
  using NodeId = int64_t;

  static constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::min();

  ThunkExecutor(ThunkExecutor&&) = default;
  ThunkExecutor& operator=(ThunkExecutor&&) = default;

  static absl::StatusOr<ThunkExecutor> Create(ThunkSequence thunk_sequence);

  // NodeDef defines an execution order for all thunks in a sequence.
  struct NodeDef {
    NodeId id = kInvalidNodeId;
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

  absl::Span<const NodeDef> nodes_defs() const { return nodes_defs_; }
  const NodeDef& node_def(NodeId id) const { return nodes_defs_[id]; }

  absl::Span<const NodeId> source() const { return source_; }
  absl::Span<const NodeId> sink() const { return sink_; }

  BufferUses buffer_uses() const { return thunk_sequence_.buffer_uses(); }
  ResourceUses resource_uses() const { return thunk_sequence_.resource_uses(); }

  std::string ToString() const;

  bool is_sequential() const { return is_sequential_; }

 private:
  using ReadyQueue = absl::InlinedVector<NodeId, 8>;

  ThunkExecutor(ThunkSequence thunk_sequence, std::vector<NodeDef> nodes_defs);

  // At run time NodeDef instantiated as a Node with an atomic counter that
  // drops to zero when all in_edges are ready.
  struct Node {
    NodeId id = kInvalidNodeId;
    std::atomic<int64_t>* counter = nullptr;
    const std::vector<NodeId>* out_edges = nullptr;
  };

  // A struct to keep the state of a running executor.
  struct ExecuteState {
    ExecuteState(ThunkExecutor* executor, Thunk::TaskRunner* runner);

    ThunkExecutor* executor;
    Thunk::TaskRunner* runner;

    // Containers for nodes' pending counters and nodes themselves.
    absl::FixedArray<std::atomic<int64_t>> counters;
    absl::InlinedVector<Node, 32> nodes;

    // We store the first error from failed thunks in `abort_status` and at the
    // end of execution the executor forwards it via the `execute_event`.
    std::atomic<bool> abort;
    absl::Mutex abort_mutex;
    absl::Status abort_status ABSL_GUARDED_BY(abort_mutex);

    // Once the number of pending sink nodes drops to zero, the execution is
    // completed and we set `execute_event` as concrete or error.
    std::atomic<int64_t> pending_sink_nodes;
    tsl::AsyncValueRef<ExecuteEvent> execute_event;
  };

  // Executes thunks sequentially starting from the first thunk in the sequence.
  tsl::AsyncValueRef<ExecuteEvent> ExecuteSequential(
      const Thunk::ExecuteParams& params);

  // Resumes sequential thunk execution starting from the given index.
  void ResumeExecuteSequential(int64_t index,
                               const Thunk::ExecuteParams& params,
                               tsl::AsyncValueRef<ExecuteEvent> event);

  // Executes nodes in the ready queue with given thunk parameters.
  void Execute(ExecuteState* state, const Thunk::ExecuteParams& params,
               ReadyQueue ready_queue);

  // Processes out edges of a completed `node` and updates `ready_queue` with
  // nodes that are ready to execute. If `event` is in error state, aborts the
  // execution and records the error status to forward it to the caller.
  void ProcessOutEdges(ExecuteState* state,
                       tsl::AsyncValuePtr<Thunk::ExecuteEvent> node_event,
                       Node& node, ReadyQueue& ready_queue);

  // Runs a transitive reduction on the NodeDef graph to remove redundant edges.
  // Returns the number of removed edges.
  //
  // See: https://en.wikipedia.org/wiki/Transitive_reduction
  int64_t TransitiveReduction();

  ThunkSequence thunk_sequence_;
  std::vector<NodeDef> nodes_defs_;

  std::vector<NodeId> source_;
  std::vector<NodeId> sink_;

  // If NodeDef graph dependency structure is sequential and does not have any
  // opportunities for executing thunks concurrently, we skip the expensive
  // async execution and simply run thunks in the `thunk_sequence_` one by one.
  bool is_sequential_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_THUNK_EXECUTOR_H_
