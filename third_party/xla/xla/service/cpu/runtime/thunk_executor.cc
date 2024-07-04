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

#include "xla/service/cpu/runtime/thunk_executor.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/cpu/runtime/resource_use.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

ThunkExecutor::ThunkExecutor(ThunkSequence thunk_sequence,
                             std::vector<NodeDef> nodes_defs)
    : thunk_sequence_(std::move(thunk_sequence)),
      nodes_defs_(std::move(nodes_defs)),
      is_sequential_(true) {
  for (NodeId i = 0; i < nodes_defs_.size(); ++i) {
    // Mark nodes with empty in-edges as source nodes.
    if (nodes_defs_[i].in_edges.empty()) {
      source_.push_back(i);
    }

    // Mark nodes with empty out-edges as sink nodes.
    if (nodes_defs_[i].out_edges.empty()) {
      sink_.push_back(i);
    }
  }

  // Erase redundant edges between nodes.
  int64_t num_erased_edges = TransitiveReduction();

  // Check if constructed execution DAG is sequential: every node depends on the
  // completion of the previous node.
  for (NodeId i = 1; i < nodes_defs_.size() && is_sequential_; ++i) {
    is_sequential_ &= (absl::c_count(nodes_defs_[i].in_edges, i - 1) != 0);
  }

  VLOG(2) << absl::StreamFormat(
      "Constructed ThunkExecutor with %d nodes: #source_nodes=%d "
      "#sink_nodes=%d, #erased_edges=%d, is_sequential=%v",
      nodes_defs_.size(), source_.size(), sink_.size(), num_erased_edges,
      is_sequential_);

  // Sanity check that all vectors are empty or all vectors are non-empty.
  DCHECK((!source_.empty() && !sink_.empty() && !thunk_sequence_.empty()) ||
         (source_.empty() && sink_.empty() && thunk_sequence_.empty()));
}

absl::StatusOr<ThunkExecutor> ThunkExecutor::Create(
    ThunkSequence thunk_sequence) {
  std::vector<NodeDef> defs(thunk_sequence.size());

  std::vector<BufferUse::ReadWriteSet> buffer_rwsets(thunk_sequence.size());
  std::vector<ResourceUse::ReadWriteSet> resource_rwsets(thunk_sequence.size());

  // TODO(ezhulenev): This is very inefficient O(N^2) complexity algorithm
  // that will create a lot of redundant edges. We can do much better by
  // stopping traversal once we prove that we already have dependencies on the
  // most recent updates that touch the whole buffer slice.

  for (NodeId i = 0; i < thunk_sequence.size(); ++i) {
    defs[i].id = i;

    Thunk& thunk = *thunk_sequence[i];
    buffer_rwsets[i].AddAll(thunk.buffer_uses());
    resource_rwsets[i].AddAll(thunk.resource_uses());

    for (NodeId j = i - 1; j >= 0; --j) {
      // Check if node `i` must be executed after node `j`.
      if (buffer_rwsets[j].HasConflicts(buffer_rwsets[i]) ||
          resource_rwsets[j].HasConflicts(resource_rwsets[i])) {
        defs[j].out_edges.push_back(i);
        defs[i].in_edges.push_back(j);
      }
    }
  }

  return ThunkExecutor(std::move(thunk_sequence), std::move(defs));
}

ThunkExecutor::ExecuteState::ExecuteState(ThunkExecutor* executor,
                                          Thunk::TaskRunner* runner)
    : executor(executor),
      runner(runner),
      counters(executor->nodes_defs().size()),
      nodes(executor->nodes_defs().size()),
      abort(false),
      pending_sink_nodes(executor->sink().size()),
      execute_event(tsl::MakeConstructedAsyncValueRef<ExecuteEvent>()) {
  DCHECK(runner == nullptr || static_cast<bool>(*runner))
      << "`runner` must be nullptr or a valid TaskRunner";

  for (NodeId id = 0; id < nodes.size(); ++id) {
    const NodeDef& node_def = executor->node_def(id);
    counters[id].store(node_def.in_edges.size(), std::memory_order_release);
    nodes[id] = Node{id, &counters[id], &node_def.out_edges};
  }
}

tsl::AsyncValueRef<ThunkExecutor::ExecuteEvent> ThunkExecutor::Execute(
    const Thunk::ExecuteParams& params) {
  // Short-circuit execution of trivial thunk sequences.
  if (ABSL_PREDICT_FALSE(thunk_sequence_.empty())) {
    return Thunk::OkExecuteEvent();
  }
  if (ABSL_PREDICT_FALSE(thunk_sequence_.size() == 1)) {
    return thunk_sequence_[0]->Execute(params);
  }

  // If thunk sequence dependencies form a sequential execution graph, we skip
  // expensive async execution and simply run thunks one by one.
  if (is_sequential_) {
    return ExecuteSequential(params);
  }

  // Create async execution state on heap and kick-off execution.
  auto state = std::make_unique<ExecuteState>(this, params.task_runner);
  Execute(state.get(), params, ReadyQueue(source_.begin(), source_.end()));

  // Move execute state to the execute event callback to ensure that it is kept
  // alive while thunk executor has pending tasks.
  auto execute_event = state->execute_event;
  execute_event.AndThen([state = std::move(state)] {
    CHECK_EQ(state->pending_sink_nodes.load(std::memory_order_acquire), 0)
        << "All sink nodes must be completed before execute_event is marked "
           "available.";
  });

  return execute_event;
}

tsl::AsyncValueRef<ThunkExecutor::ExecuteEvent>
ThunkExecutor::ExecuteSequential(const Thunk::ExecuteParams& params) {
  for (int64_t i = 0; i < thunk_sequence_.size(); ++i) {
    Thunk& thunk = *thunk_sequence_[i];
    auto execute_event = thunk.Execute(params);

    // If thunk execution is not completed yet, attach a continuation to
    // resume sequential execution starting from the next thunk.
    if (ABSL_PREDICT_FALSE(!execute_event.IsAvailable())) {
      auto event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();
      execute_event.AndThen([this, &params, i, event](absl::Status status) {
        if (ABSL_PREDICT_FALSE(!status.ok())) {
          event.SetError(std::move(status));
        } else {
          ResumeExecuteSequential(i + 1, params, std::move(event));
        }
      });
      return event;
    }

    // Abort execution if any of the thunks failed.
    if (ABSL_PREDICT_FALSE(execute_event.IsError())) {
      return execute_event;
    }
  }

  // If we got to the end of the sequence it means that all thunks have
  // succeeded.
  return Thunk::OkExecuteEvent();
}

void ThunkExecutor::ResumeExecuteSequential(
    int64_t index, const Thunk::ExecuteParams& params,
    tsl::AsyncValueRef<ExecuteEvent> event) {
  for (int64_t i = index; i < thunk_sequence_.size(); ++i) {
    Thunk& thunk = *thunk_sequence_[i];
    auto execute_event = thunk.Execute(params);

    // If thunk execution is not completed yet, attach a continuation to
    // resume sequential execution starting from the next thunk.
    if (ABSL_PREDICT_FALSE(!execute_event.IsAvailable())) {
      execute_event.AndThen(
          [this, &params, i, event = std::move(event)](absl::Status status) {
            if (ABSL_PREDICT_FALSE(!status.ok())) {
              event.SetError(std::move(status));
            } else {
              ResumeExecuteSequential(i + 1, params, std::move(event));
            }
          });
      return;
    }

    // Abort execution if any of the thunks failed.
    if (ABSL_PREDICT_FALSE(execute_event.IsError())) {
      event.SetError(execute_event.GetError());
      return;
    }
  }

  // If we got to the end of the sequence it means that all thunks have
  // succeeded.
  event.SetStateConcrete();
}

void ThunkExecutor::Execute(ExecuteState* state,
                            const Thunk::ExecuteParams& params,
                            ReadyQueue ready_queue) {
  tsl::profiler::TraceMe trace("ThunkExecutor::Execute");
  CHECK(!ready_queue.empty()) << "Ready queue must not be empty";  // Crash Ok

  bool has_runner = state->runner != nullptr;

  for (int64_t i = 0; i < ready_queue.size(); ++i) {
    NodeId id = ready_queue[i];
    Node& node = state->nodes[id];

    int64_t cnt = node.counter->load(std::memory_order_acquire);
    CHECK_EQ(cnt, 0) << "Node counter must be 0";  // Crash Ok

    // TODO(ezhulenev): Benchmark other strategies of work distribution, i.e. we
    // can offload only second half of the ready queue if it grows above some
    // threshold. Also we might want to add a limit on the number of concurrent
    // tasks processing the same execute session.

    // We use recursive work splitting to push the tail of the ready queue to
    // the task runner. Recursive work splitting creates a more uniform work
    // distribution across the task runner threads and avoids a situation when
    // we have a long tail of work that is processed by a single thread.
    if (ABSL_PREDICT_FALSE(has_runner && i < ready_queue.size() - 1)) {
      int64_t start_index = i + 1;
      int64_t end_index = ready_queue.size();

      // Execute [mid_index, end_index) nodes in the task runner.
      while (end_index - start_index > 1) {
        int64_t mid_index = (start_index + end_index) / 2;
        (*state->runner)([&params, state,
                          ready_queue = ReadyQueue(
                              ready_queue.begin() + mid_index,
                              ready_queue.begin() + end_index)]() mutable {
          state->executor->Execute(state, params, std::move(ready_queue));
        });
        end_index = mid_index;
      }

      // Execute the last remaining ready node in the task runner.
      (*state->runner)(
          [&params, state,
           ready_queue = ReadyQueue({ready_queue[start_index]})]() mutable {
            state->executor->Execute(state, params, std::move(ready_queue));
          });

      // Erase ready nodes passed to the task runner.
      ready_queue.erase(ready_queue.begin() + i + 1, ready_queue.end());
    }

    // Execute thunk for the given node id. If execution is aborted, we keep
    // processing the nodes DAG without executing thunks.
    Thunk& thunk = *state->executor->thunk_sequence_[id];
    auto execute_event = state->abort.load(std::memory_order_relaxed)
                             ? Thunk::OkExecuteEvent()
                             : thunk.Execute(params);

    if (ABSL_PREDICT_TRUE(execute_event.IsAvailable())) {
      // If thunk execution is completed, process out edges in the current
      // thread and keep working on the ready queue.
      ProcessOutEdges(state, execute_event.AsPtr(), node, ready_queue);

    } else {
      // If thunk execution is not completed yet, attach a continuation to the
      // event and resume execution on the continuation thread (ready queue
      // processing will continue on a thread that marked event completed).
      execute_event.AndThen(
          [&params, &node, state, execute_event = execute_event.AsPtr()] {
            ReadyQueue ready_queue;
            state->executor->ProcessOutEdges(state, execute_event, node,
                                             ready_queue);
            // If ready queue is empty it might mean that we have completed an
            // execution and destroyed the `state`.
            if (ABSL_PREDICT_TRUE(!ready_queue.empty())) {
              state->executor->Execute(state, params, std::move(ready_queue));
            }
          });
    }
  }
}

void ThunkExecutor::ProcessOutEdges(
    ExecuteState* state, tsl::AsyncValuePtr<Thunk::ExecuteEvent> node_event,
    Node& node, ReadyQueue& ready_queue) {
  // If thunk execution failed, mark execution as aborted and record the error.
  // We still continue processing the nodes DAG to eventually mark sink nodes
  // completed as it's easier than to add a special abort handling logic.
  if (ABSL_PREDICT_FALSE(node_event.IsError())) {
    absl::MutexLock lock(&state->abort_mutex);
    state->abort = true;
    state->abort_status.Update(node_event.GetError());
  }

  // Load `is_sink` before dropping node counters because otherwise it might
  // race with NodeDef destructor.
  bool is_sink = node.out_edges->empty();

  // Append ready nodes to the back of the ready queue.
  for (NodeId out_edge : *node.out_edges) {
    Node& out_node = state->nodes[out_edge];

    int64_t cnt = out_node.counter->fetch_sub(1, std::memory_order_release);
    CHECK_GE(cnt, 1) << "Node counter can't drop below 0";  // Crash Ok
    if (cnt == 1) ready_queue.push_back(out_edge);
  }

  // Drop the pending sink nodes counter if the node is a sink.
  if (ABSL_PREDICT_FALSE(is_sink)) {
    // Check if it was the last sink node and thunk executor is done. We update
    // the counter using `std::memory_order_acq_rel` to ensure that the
    // remaining memory writes are visible to the consumer of execute event.
    bool is_done =
        state->pending_sink_nodes.fetch_sub(1, std::memory_order_acq_rel) == 1;
    if (ABSL_PREDICT_TRUE(!is_done)) return;

    // In the unlikely event of an execution error during thunk execution,
    // forward it to the caller via the execute event.
    if (ABSL_PREDICT_FALSE(state->abort.load(std::memory_order_relaxed))) {
      auto take_error = [&] {
        absl::MutexLock lock(&state->abort_mutex);
        CHECK(!state->abort_status.ok())  // Crash Ok
            << "Abort status must be set if execution is aborted";
        return std::move(state->abort_status);
      };
      state->execute_event.SetError(take_error());
    } else {
      state->execute_event.SetStateConcrete();
    }
  }
}

// Erases edge from `from` node to `to` node if it exists.
//
// TODO(ezhulenev): Out and In-edges are sorted in increasing and decreasing
// order respectively. We can use binary search to speed up this function.
static int64_t EraseEdge(ThunkExecutor::NodeDef& from,
                         ThunkExecutor::NodeDef& to) {
  auto out_edge_it = absl::c_find(from.out_edges, to.id);
  auto in_edge_it = absl::c_find(to.in_edges, from.id);

  bool has_out_edge = out_edge_it != from.out_edges.end();
  bool has_in_edge = in_edge_it != to.in_edges.end();

  DCHECK_EQ(has_out_edge, has_in_edge) << "Edges must be symmetric";

  if (has_out_edge && has_in_edge) {
    from.out_edges.erase(out_edge_it);
    to.in_edges.erase(in_edge_it);
    return 1;
  }

  return 0;
}

int64_t ThunkExecutor::TransitiveReduction() {
  int64_t num_erased_edges = 0;

  // Keep workspace for DFS traversal between iterations.
  std::vector<int64_t> stack;
  std::vector<bool> visited;

  auto add_to_stack = [&](int64_t node_id) {
    if (!visited[node_id]) {
      stack.push_back(node_id);
      visited[node_id] = true;
    }
  };

  // For each node we do a DFS traversal and delete redundant edges that
  // connect source node with the node reachable via DFS.
  for (int64_t i = 0; i < nodes_defs_.size(); ++i) {
    NodeDef& source_node = nodes_defs_[i];

    // Clear DFS workspace from previous iteration.
    stack.clear();
    visited.assign(nodes_defs_.size(), false);

    // Initialize stack with nodes reachable via immediate out nodes. We don't
    // need to add source node and immediate out nodes to the visited set
    // because graph is acyclic and we don't visit them again.
    for (int64_t out_id : source_node.out_edges) {
      NodeDef& out_node = nodes_defs_[out_id];
      for (int64_t start_id : out_node.out_edges) add_to_stack(start_id);
    }

    // Traverse the graph and delete redundant edges.
    while (!stack.empty()) {
      int64_t node_id = stack.back();
      stack.pop_back();

      NodeDef& node = nodes_defs_[node_id];
      num_erased_edges += EraseEdge(source_node, node);

      for (int64_t out_id : node.out_edges) add_to_stack(out_id);
    }
  }

  return num_erased_edges;
}

std::string ThunkExecutor::ToString() const {
  std::string str = absl::StrFormat(
      "ThunkExecutor: #thunks=%d #source_nodes=%d #sink_nodes=%d",
      thunk_sequence_.size(), source_.size(), sink_.size());

  // Collect names of `in_edges`.
  std::vector<std::vector<std::string>> in_edges(thunk_sequence_.size());
  for (const auto& node_def : nodes_defs_) {
    for (NodeId in_edge : node_def.in_edges) {
      in_edges[node_def.id].push_back(thunk_sequence_[in_edge]->info().op_name);
    }
  }

  // Print thunks with a list of their dependencies;
  for (NodeId i = 0; i < thunk_sequence_.size(); ++i) {
    const Thunk& thunk = *thunk_sequence_[i];
    bool is_source = absl::c_find(source_, i) != source_.end();
    bool is_sink = absl::c_find(sink_, i) != sink_.end();
    absl::StrAppendFormat(
        &str,
        "\n thunk #%05d: op_name=%s, dependencies=[%s], source=%v, sink=%v", i,
        thunk.info().op_name, absl::StrJoin(in_edges[i], ", "), is_source,
        is_sink);
  }

  return str;
}

}  // namespace xla::cpu
