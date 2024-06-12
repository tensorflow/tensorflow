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
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

ThunkExecutor::ThunkExecutor(ThunkSequence thunk_sequence,
                             std::vector<NodeDef> nodes_defs)
    : thunk_sequence_(std::move(thunk_sequence)),
      nodes_defs_(std::move(nodes_defs)) {
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

  // Sanity check that all vectors are empty or all vectors are non-empty.
  DCHECK((!source_.empty() && !sink_.empty() && !thunk_sequence_.empty()) ||
         (source_.empty() && sink_.empty() && thunk_sequence_.empty()));
}

absl::StatusOr<ThunkExecutor> ThunkExecutor::Create(
    ThunkSequence thunk_sequence) {
  std::vector<NodeDef> defs(thunk_sequence.size());

  std::vector<BufferUse::ReadWriteSet> rwsets(thunk_sequence.size());
  std::vector<Thunk::BufferUses> buffer_uses(thunk_sequence.size());

  // TODO(ezhulenev): This is very inefficient O(N^2) complexity algorithm
  // that will create a lot of redundant edges. We can do much better by
  // stopping traversal once we prove that we already have dependencies on the
  // most recent updates that touch the whole buffer slice.

  for (NodeId i = 0; i < thunk_sequence.size(); ++i) {
    defs[i].id = i;

    Thunk& thunk = *thunk_sequence[i];
    rwsets[i].AddAll(thunk.buffer_uses());

    for (NodeId j = i - 1; j >= 0; --j) {
      // Node `i` must be executed after node `j`.
      if (rwsets[j].HasConflicts(rwsets[i])) {
        defs[j].out_edges.push_back(i);
        defs[i].in_edges.push_back(j);
      }
    }
  }

  return ThunkExecutor(std::move(thunk_sequence), std::move(defs));
}

ThunkExecutor::ExecuteState::ExecuteState(ThunkExecutor* executor,
                                          TaskRunner runner)
    : executor(executor),
      runner(std::move(runner)),
      abort(false),
      counters(executor->nodes_defs().size()),
      nodes(executor->nodes_defs().size()),
      done(executor->sink().size()) {
  for (NodeId id = 0; id < nodes.size(); ++id) {
    const NodeDef& node_def = executor->node_def(id);
    counters[id].store(node_def.in_edges.size(), std::memory_order_release);
    nodes[id] = Node{id, &counters[id], &node_def.out_edges};
  }
}

absl::Status ThunkExecutor::Execute(const Thunk::ExecuteParams& params,
                                    TaskRunner runner) {
  ReadyQueue ready_queue(source_.begin(), source_.end());
  if (ready_queue.empty()) return absl::OkStatus();

  auto state = std::make_unique<ExecuteState>(this, std::move(runner));
  Execute(state.get(), params, std::move(ready_queue));

  tsl::profiler::TraceMe trace("ThunkExecutor::Execute (wait for done)");
  state->done.Wait();

  // If execution was aborted, forward error to the caller.
  if (state->abort.load(std::memory_order_relaxed)) {
    absl::MutexLock lock(&state->abort_mutex);
    CHECK(!state->abort_status.ok())  // Crash Ok
        << "Abort status must be set if execution is aborted";
    return state->abort_status;
  }

  return absl::OkStatus();
}

void ThunkExecutor::Execute(ExecuteState* state,
                            const Thunk::ExecuteParams& params,
                            ReadyQueue ready_queue) {
  tsl::profiler::TraceMe trace("ThunkExecutor::Execute");
  if (ready_queue.empty()) return;  // Nothing to execute.

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

    // Push the tail of the ready queue to the task runner.
    if (has_runner && i < ready_queue.size() - 1) {
      ReadyQueue tail(ready_queue.begin() + i + 1, ready_queue.end());
      ready_queue.erase(ready_queue.begin() + i + 1, ready_queue.end());
      state->runner([&params, state, tail = std::move(tail)]() mutable {
        state->executor->Execute(state, params, std::move(tail));
      });
    }

    // Execute thunk for the given node id. If execution is aborted, we keep
    // processing the nodes DAG without executing thunks.
    Thunk& thunk = *state->executor->thunk_sequence_[id];
    auto execute_event = state->abort.load(std::memory_order_relaxed)
                             ? Thunk::OkExecuteEvent()
                             : thunk.Execute(params);

    // If thunk execution is not completed yet, attach a continuation to the
    // event and resume execution on the continuation thread (ready queue
    // processing will continue on a thread that marked event completed).
    if (ABSL_PREDICT_FALSE(!execute_event.IsAvailable())) {
      execute_event.AndThen([&, state, execute_event = execute_event.AsPtr()] {
        ReadyQueue ready_queue;
        ProcessOutEdges(state, execute_event, node, ready_queue);
        Execute(state, params, std::move(ready_queue));
      });
    } else {
      // If thunk execution is completed, process out edges in the current
      // thread and keep working on the ready queue.
      ProcessOutEdges(state, execute_event.AsPtr(), node, ready_queue);
    }
  }
}

void ThunkExecutor::ProcessOutEdges(
    ExecuteState* state, tsl::AsyncValuePtr<Thunk::ExecuteEvent> execute_event,
    Node& node, ReadyQueue& ready_queue) {
  // If thunk execution failed, mark execution as aborted and record the error.
  // We still continue processing the nodes DAG to eventually mark sink nodes
  // completed as it's easier than to add a special abort handling logic.
  if (ABSL_PREDICT_FALSE(execute_event.IsError())) {
    absl::MutexLock lock(&state->abort_mutex);
    state->abort = true;
    state->abort_status.Update(execute_event.GetError());
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

  // Drop done counter if the node is a sink.
  if (ABSL_PREDICT_FALSE(is_sink)) state->done.DecrementCount();
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
