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

#include "xla/backends/cpu/runtime/thunk_executor.h"

#include <sys/types.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/resource_use.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/numbers.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

// If XLA:CPU compiled with `-DXLA_CPU_USE_BLOCKING_THUNK_EXECUTOR` we'll run
// all thunks sequentially and block on the completion of all thunks, which is
// helpful for debugging and gives more readable Xprof traces.
//
// WARNING: This option is UNSAFE and can lead to deadlocks. It should be used
// only for debugging purposes.
static constexpr bool UseBlockingThunkExecutor() {
#if defined(XLA_CPU_USE_BLOCKING_THUNK_EXECUTOR)
  return true;
#else
  return false;
#endif  // XLA_CPU_USE_BLOCKING_THUNK_EXECUTOR
}

namespace {

// An adaptor from Thunk to ExecutionGraph::Operation for building an execution
// graph from a thunk sequence.
struct ThunkOperation : public ExecutionGraph::Operation {
  ThunkOperation(Thunk::BufferUses buffers, Thunk::ResourceUses resources)
      : buffers(std::move(buffers)), resources(std::move(resources)) {}

  absl::Span<const BufferUse> BufferUses() const final { return buffers; }
  absl::Span<const ResourceUse> ResourceUses() const final { return resources; }

 private:
  Thunk::BufferUses buffers;
  Thunk::ResourceUses resources;
};

}  // namespace

// Converts a ThunkSequence to a vector of ThunkOperations.
static std::vector<ThunkOperation> CreateThunkOperations(
    const ThunkSequence& thunk_sequence) {
  std::vector<ThunkOperation> operations;
  operations.reserve(thunk_sequence.size());
  for (const auto& thunk : thunk_sequence) {
    operations.emplace_back(thunk->buffer_uses(), thunk->resource_uses());
  }
  return operations;
}

ThunkExecutor::ThunkExecutor(ThunkSequence thunk_sequence,
                             ExecutionGraph execution_graph,
                             const ThunkExecutor::Options& options)
    : thunk_sequence_(std::move(thunk_sequence)),
      execution_graph_(std::move(execution_graph)),
      options_(options),
      num_thunks_(thunk_sequence_.size()),
      is_sequential_(execution_graph_.is_sequential()) {
  // Prefer sequential execution if all thunks use small buffers.
  auto uses_small_buffers = [&](const std::unique_ptr<Thunk>& thunk) {
    return absl::c_all_of(thunk->buffer_uses(), [&](const BufferUse& use) {
      return use.slice().size() <= options.execute_sequential_buffer_threshold;
    });
  };

  bool small_buffers = absl::c_all_of(thunk_sequence_, uses_small_buffers);
  is_sequential_ |= small_buffers;

  // Prefer sequential execution for small thunk sequences.
  is_sequential_ |=
      thunk_sequence_.size() <= options.execute_sequential_num_thunks_threshold;

  // Force sequential execution if we are running in blocking mode as it makes
  // Xprof traces easier to read.
  is_sequential_ |= UseBlockingThunkExecutor();

  VLOG(2) << absl::StreamFormat(
      "Constructed ThunkExecutor with %d thunks: #source_nodes=%d "
      "#sink_nodes=%d, is_sequential=%v, small_buffers=%v",
      num_thunks_, execution_graph_.source().size(),
      execution_graph_.sink().size(), is_sequential_, small_buffers);

  VLOG(6) << "ThunkExecutor execution graph:\n" << ToString();
}

absl::StatusOr<ThunkExecutor> ThunkExecutor::Create(
    ThunkSequence thunk_sequence, const ThunkExecutor::Options& options) {
  // Construct an execution graph for the given thunk sequence.
  TF_ASSIGN_OR_RETURN(ExecutionGraph execution_graph,
                      ExecutionGraph::Create<ThunkOperation>(
                          CreateThunkOperations(thunk_sequence)));

  return ThunkExecutor(std::move(thunk_sequence), std::move(execution_graph),
                       options);
}

ThunkExecutor::ExecuteState::Node::Node(const NodeDef& node_def)
    : counter(node_def.in_edges.size()), out_edges(node_def.out_edges) {}

ThunkExecutor::ExecuteState::ExecuteState(ThunkExecutor* executor,
                                          Thunk::TaskRunner* runner)
    : executor(executor),
      runner(runner),
      nodes(executor->execution_graph_.nodes_defs().size()),
      execute_event(tsl::MakeConstructedAsyncValueRef<ExecuteEvent>()),
      pending_sink_nodes(executor->execution_graph_.sink().size()),
      abort(false) {
  NodeStorage* node = nodes.data();
  for (const NodeDef& node_def : executor->execution_graph_.nodes_defs()) {
    new (node++) Node(node_def);
  }
}

// Executes given `thunk` and adds tracing annotation to record the execution
// start and end events for profiling.
tsl::AsyncValueRef<Thunk::ExecuteEvent> ThunkExecutor::TracedExecute(
    Thunk& thunk, const Thunk::ExecuteParams& params) {
  // If profiler is not active avoid overheads of calling AndThen below.
  if (ABSL_PREDICT_TRUE(!tsl::profiler::TraceMe::Active())) {
    return thunk.Execute(params);
  }

  // Create a producer traceme to capture the start event.
  tsl::profiler::TraceMeProducer producer([&] { return thunk.TraceMeEncode(); },
                                          tsl::profiler::ContextType::kGeneric);

  auto execute_event = thunk.Execute(params);

  // When thunk execution completes, create a consumer traceme to capture the
  // end event.
  execute_event.AndThen([context_id = producer.GetContextId(), &thunk] {
    tsl::profiler::TraceMeConsumer(
        [&] { return absl::StrFormat("end: %s", thunk.info().op_name); },
        tsl::profiler::ContextType::kGeneric, context_id);
  });

  return execute_event;
}

tsl::AsyncValueRef<ThunkExecutor::ExecuteEvent> ThunkExecutor::Execute(
    const Thunk::ExecuteParams& params) {
  // Short-circuit execution of empty thunk sequence.
  if (ABSL_PREDICT_FALSE(num_thunks_ == 0)) {
    return Thunk::OkExecuteEventSingleton();
  }

  // Short-circuit execution of single thunk sequence.
  if (ABSL_PREDICT_FALSE(num_thunks_ == 1)) {
    return TracedExecute(*thunk_sequence_[0], params);
  }

  // When we choose sequential execution strategy (we rely on heuristics and
  // a cost model to make the decision), we skip expensive async execution and
  // simply run thunks one by one. This minimizes runtime overheads from small
  // XLA programs with many cheap operations.
  if (is_sequential_) {
    return ExecuteSequential(params);
  }

  // Create async execution state on heap and kick-off execution.
  auto state = std::make_unique<ExecuteState>(this, params.task_runner);

  // When we kick-off execution we don't have to grab the session lock, as the
  // main thread is not counted towards the number of concurrent workers limit.
  // This also works for thunks with nested thunk executors (i.e., WhileThunk),
  // as launching nested thunk sequence must not reduce the available
  // concurrency for the other thunks executing in parallel.
  auto execute = [&](auto ready_queue) {
    Execute(state.get(), params, std::move(ready_queue), /*lock=*/nullptr);
  };

  switch (options_.ready_queue_type) {
    case Options::ReadyQueueType::kFifo:
      execute(FifoReadyQueue(execution_graph_.source()));
      break;
    case Options::ReadyQueueType::kLifo:
      execute(LifoReadyQueue(execution_graph_.source()));
      break;
    case Options::ReadyQueueType::kPriority:
      execute(PriorityReadyQueue(execution_graph_.nodes_defs(),
                                 execution_graph_.source()));
      break;
  }

  // If execution already completed (all kernels executed in the caller thread),
  // immediately return the result to avoid wasteful reference counting below.
  if (ABSL_PREDICT_TRUE(state->execute_event.IsAvailable())) {
    return std::move(state->execute_event);
  }

  // Move execute state to the execute event callback to ensure that it is kept
  // alive while thunk executor has pending tasks.
  tsl::AsyncValueRef<ExecuteEvent> execute_event = state->execute_event;
  execute_event.AndThen([state = std::move(state)] {
    auto cnt = state->pending_sink_nodes.load(std::memory_order_acquire);
    DCHECK_EQ(cnt, 0)
        << "All sink nodes must be completed before execute_event is marked "
           "available.";
  });

  return execute_event;
}

// We deliberately opt-out from the cognitive complexity check, as this
// function is on a hot path, any any attempt to split it leads to measurable
// regressions in microbenchmarks.
tsl::AsyncValueRef<ThunkExecutor::ExecuteEvent>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
ThunkExecutor::ExecuteSequential(const Thunk::ExecuteParams& params) {
  if constexpr (UseBlockingThunkExecutor()) {
    VLOG(2) << absl::StreamFormat(
        "ThunkExecutor::ExecuteSequential: execute %d thunks in blocking mode",
        num_thunks_);
  }

  for (auto it = thunk_sequence_.begin(); it != thunk_sequence_.end(); ++it) {
    // Record thunk execution start time in blocking mode.
    uint64_t start_us;
    if constexpr (UseBlockingThunkExecutor()) {
      start_us = tsl::Env::Default()->NowMicros();
    }

    Thunk& thunk = **it;
    auto execute_event = TracedExecute(thunk, params);

    // Log thunk execution time in blocking mode.
    if constexpr (UseBlockingThunkExecutor()) {
      tsl::BlockUntilReady(execute_event);
      VLOG(2) << absl::StreamFormat(
          "  thunk[%d] took %s (op_name: %s)",
          std::distance(thunk_sequence_.begin(), it),
          tsl::strings::HumanReadableElapsedTime(
              (tsl::Env::Default()->NowMicros() - start_us) / 1000000.0),
          thunk.info().op_name);
    }

    // Fast path for thunks executed inline and returned OkExecuteEvent.
    if (ABSL_PREDICT_TRUE(thunk.IsOkExecuteEvent(execute_event))) {
      continue;
    }

    // If thunk execution is not completed yet, attach a continuation to
    // resume sequential execution starting from the next thunk.
    if (ABSL_PREDICT_FALSE(!execute_event.IsAvailable())) {
      auto event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();
      execute_event.AndThen([this, &params, it, event](absl::Status status) {
        Thunk::TaskRunner* runner = params.task_runner;

        if (ABSL_PREDICT_FALSE(!status.ok())) {
          event.SetError(std::move(status));
        } else if (ABSL_PREDICT_TRUE(!runner || runner->current_worker_id())) {
          // Resume execution in the current thread if we are already running
          // on a thread managed by the task runner.
          ResumeExecuteSequential(it + 1, params, std::move(event));
        } else {
          // Resume execution in the task runner to avoid thread "leaks".
          (*runner)([this, &params, it, event = std::move(event)] {
            ResumeExecuteSequential(it + 1, params, std::move(event));
          });
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
  return Thunk::OkExecuteEventSingleton();
}

void ThunkExecutor::ResumeExecuteSequential(
    ThunkIterator it, const Thunk::ExecuteParams& params,
    tsl::AsyncValueRef<ExecuteEvent> event) {
  for (; it != thunk_sequence_.end(); ++it) {
    Thunk& thunk = **it;
    auto execute_event = TracedExecute(thunk, params);

    // Fast path for thunks executed inline and returned OkExecuteEvent.
    if (ABSL_PREDICT_TRUE(thunk.IsOkExecuteEvent(execute_event))) {
      continue;
    }

    // If thunk execution is not completed yet, attach a continuation to
    // resume sequential execution starting from the next thunk.
    if (ABSL_PREDICT_FALSE(!execute_event.IsAvailable())) {
      execute_event.AndThen([this, &params, it,
                             event = std::move(event)](absl::Status status) {
        Thunk::TaskRunner* runner = params.task_runner;

        if (ABSL_PREDICT_FALSE(!status.ok())) {
          event.SetError(std::move(status));
        } else if (ABSL_PREDICT_TRUE(!runner || runner->current_worker_id())) {
          // Resume execution in the current thread if we are already
          // running on a thread managed by the task runner.
          ResumeExecuteSequential(it + 1, params, std::move(event));
        } else {
          // Resume execution in the task runner to avoid thread "leaks".
          (*runner)([this, &params, it, event = std::move(event)] {
            ResumeExecuteSequential(it + 1, params, std::move(event));
          });
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

// We deliberately opt-out from the cognitive complexity check, as this
// function is on a hot path, any any attempt to split it leads to measurable
// regressions in microbenchmarks.
template <typename ReadyQueue>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void ThunkExecutor::Execute(ExecuteState* state,
                            const Thunk::ExecuteParams& params,
                            ReadyQueue ready_queue,
                            Thunk::ExecuteSession::Lock lock) {
  DCHECK(!ready_queue.Empty()) << "Ready queue must not be empty";

  tsl::profiler::TraceMe trace("ThunkExecutor::Execute");
  bool has_runner = state->runner != nullptr;
  bool has_lock = static_cast<bool>(lock);

  // Threshold for splitting ready queue into separate thunk executor tasks.
  int64_t split_threshold = params.session.split_threshold();

  while (!ready_queue.Empty()) {
    // If we had and execution lock passed to us by the caller, we must not
    // lose it in the middle of the loop (donate to one of the callbacks).
    DCHECK_EQ(static_cast<bool>(lock), has_lock)
        << "Execute session lock must not be lost in the middle of the loop";

    NodeId id = ready_queue.Pop();
    ExecuteState::Node& node = state->node(id);

    int64_t cnt = node.counter.load(std::memory_order_acquire);
    DCHECK_EQ(cnt, 0) << "Node counter must be 0";

    // If we have multiple ready thunks, split the ready queue and offload
    // thunks processing to the task runner.
    int64_t num_ready_thunks = ready_queue.Size();
    if (ABSL_PREDICT_FALSE(has_runner && num_ready_thunks > split_threshold)) {
      SplitReadyQueue(state, params, ready_queue, split_threshold);
    }

    // Execute thunk for the given node id. If execution is aborted, we keep
    // processing the nodes DAG without executing thunks.
    Thunk& thunk = *state->executor->thunk_sequence_[id];
    tsl::AsyncValueRef<ExecuteEvent> execute_event =
        ABSL_PREDICT_FALSE(state->abort.load(std::memory_order_relaxed))
            ? Thunk::OkExecuteEventSingleton()
            : TracedExecute(thunk, params);

    if (ABSL_PREDICT_TRUE(execute_event.IsAvailable())) {
      // If thunk execution is completed, process out edges in the current
      // thread and keep working on the ready queue.
      ProcessOutEdges(state, execute_event.AsPtr(), node, ready_queue);

    } else {
      // If thunk execution is not completed yet, attach a continuation to the
      // event and resume execution on the continuation thread (ready queue
      // processing will continue on a thread that marked event completed).
      //
      // We unconditionally join the execute session by passing the session
      // lock to the callback, because having a pending execute event means
      // that we have at least one more thread that is processing the same
      // execute session. If we happen to process the last thunk in the ready
      // queue, we will forward the lock that we already hold (note that the
      // lock might be empty, if `Execute` was called by the main thread).
      execute_event.AndThen(
          [&params, &node, state, execute_event = execute_event.AsPtr(),
           ready_queue = ready_queue.CreateEmptyReadyQueue(),
           lock = ready_queue.Empty() ? std::move(lock)
                                      : params.session.Join()]() mutable {
            state->executor->ProcessOutEdges(state, execute_event, node,
                                             ready_queue);

            // If ready queue is empty, it might mean that we have completed an
            // execution and destroyed the `state`, so we make sure we don't
            // touch `state` if we don't have to.
            if (ABSL_PREDICT_FALSE(ready_queue.Empty())) {
              return;
            }

            Thunk::TaskRunner* runner = state->runner;
            if (ABSL_PREDICT_TRUE(!runner || runner->current_worker_id())) {
              // Resume execution in the current thread if we are already
              // running on a thread managed by the task runner.
              state->executor->Execute(state, params, std::move(ready_queue),
                                       std::move(lock));
            } else {
              // Resume execution in the task runner to avoid thread "leaks".
              (*runner)([state, &params, ready_queue = std::move(ready_queue),
                         lock = std::move(lock)] {
                state->executor->Execute(state, params, std::move(ready_queue),
                                         std::move(lock));
              });
            }
          });
    }
  }
}

template <typename ReadyQueue>
inline ABSL_ATTRIBUTE_ALWAYS_INLINE void ThunkExecutor::SplitReadyQueue(
    ExecuteState* state, const Thunk::ExecuteParams& params,
    ReadyQueue& ready_queue, int64_t split_threshold) {
  DCHECK(state->runner) << "TaskRunner must be set";

  // We use recursive work splitting to push the tail of the ready queue to
  // the task runner. Recursive work splitting creates a more uniform work
  // distribution across the task runner threads and avoids a situation when
  // we have a long tail of work that is processed by a single thread.
  while (ready_queue.Size() > split_threshold) {
    // Try to acquire a lock to offload ready thunks to the task runner. If
    // we can't get a lock, we will keep processing the ready queue in the
    // current thread as it means that we have enough concurrent workers
    // processing the same execute session.
    Thunk::ExecuteSession::Lock task_runner_lock = params.session.TryJoin();
    if (!task_runner_lock) {
      break;
    }

    // Execute half of the ready queue nodes in the task runner.
    (*state->runner)([&params, state, ready_queue = ready_queue.PopHalf(),
                      lock = std::move(task_runner_lock)] {
      state->executor->Execute(state, params, std::move(ready_queue),
                               std::move(lock));
    });
  }
}

template <typename ReadyQueue>
void ThunkExecutor::ProcessOutEdges(
    ExecuteState* state, tsl::AsyncValuePtr<Thunk::ExecuteEvent> node_event,
    ExecuteState::Node& node, ReadyQueue& ready_queue) {
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
  bool is_sink = node.out_edges.empty();

  // Append ready nodes to the back of the ready queue.
  for (NodeId out_edge : node.out_edges) {
    ExecuteState::Node& out_node = state->node(out_edge);

    int64_t cnt = out_node.counter.fetch_sub(1, std::memory_order_release);
    DCHECK_GE(cnt, 1) << "Node counter can't drop below 0";
    if (cnt == 1) ready_queue.Push(out_edge);
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
        DCHECK(!state->abort_status.ok())
            << "Abort status must be set if execution is aborted";
        return std::move(state->abort_status);
      };
      state->execute_event.SetError(take_error());
    } else {
      state->execute_event.SetStateConcrete();
    }
  }
}

std::string ThunkExecutor::ToString() const {
  std::string str = absl::StrFormat(
      "ThunkExecutor: #thunks=%d #source_nodes=%d #sink_nodes=%d", num_thunks_,
      execution_graph_.source().size(), execution_graph_.sink().size());

  // Collect names of `in_edges`.
  std::vector<std::vector<std::string>> in_edges(num_thunks_);
  for (const auto& node_def : execution_graph_.nodes_defs()) {
    for (NodeId in_edge : node_def.in_edges) {
      in_edges[node_def.id].push_back(thunk_sequence_[in_edge]->info().op_name);
    }
  }

  absl::Span<const NodeId> source = execution_graph_.source();
  absl::Span<const NodeId> sink = execution_graph_.sink();

  // Print thunks with a list of their dependencies;
  for (NodeId i = 0; i < num_thunks_; ++i) {
    const Thunk& thunk = *thunk_sequence_[i];
    bool is_source = absl::c_find(source, i) != source.end();
    bool is_sink = absl::c_find(sink, i) != sink.end();
    absl::StrAppendFormat(
        &str,
        "\n thunk #%05d: op_name=%s, kind=%s, dependencies=[%s], "
        "source=%v, sink=%v, priority=%d",
        i, thunk.info().op_name, Thunk::KindToString(thunk.kind()),
        absl::StrJoin(in_edges[i], ", "), is_source, is_sink,
        execution_graph_.priority(i));
  }

  return str;
}

ThunkExecutor::FifoReadyQueue::FifoReadyQueue(
    absl::Span<const NodeId> ready_nodes)
    : queue_(ready_nodes.begin(), ready_nodes.end()) {}

void ThunkExecutor::FifoReadyQueue::Push(NodeId id) { queue_.push_back(id); }

ThunkExecutor::NodeId ThunkExecutor::FifoReadyQueue::Pop() {
  DCHECK(!Empty()) << "Queue must not be empty";
  return queue_[head_++];
}

ThunkExecutor::FifoReadyQueue ThunkExecutor::FifoReadyQueue::PopHalf() {
  DCHECK(!Empty()) << "Queue must not be empty";
  auto mid = queue_.begin() + head_ + Size() / 2;
  FifoReadyQueue popped(absl::MakeConstSpan(&*mid, queue_.end() - mid));
  queue_.resize(mid - queue_.begin());
  return popped;
}

size_t ThunkExecutor::FifoReadyQueue::Size() const {
  return queue_.size() - head_;
}

bool ThunkExecutor::FifoReadyQueue::Empty() const {
  return head_ == queue_.size();
}

ThunkExecutor::FifoReadyQueue
ThunkExecutor::FifoReadyQueue::CreateEmptyReadyQueue() const {
  return FifoReadyQueue(absl::Span<const NodeId>());
}

ThunkExecutor::LifoReadyQueue::LifoReadyQueue(
    absl::Span<const NodeId> ready_nodes)
    : queue_(ready_nodes.begin(), ready_nodes.end()) {}

void ThunkExecutor::LifoReadyQueue::Push(NodeId id) { queue_.push_back(id); }

ThunkExecutor::NodeId ThunkExecutor::LifoReadyQueue::Pop() {
  DCHECK(!Empty()) << "Queue must not be empty";
  NodeId id = queue_.back();
  queue_.pop_back();
  return id;
}

ThunkExecutor::LifoReadyQueue ThunkExecutor::LifoReadyQueue::PopHalf() {
  DCHECK(!Empty()) << "Queue must not be empty";
  auto mid = Size() / 2 + 1;
  LifoReadyQueue popped(
      absl::MakeConstSpan(queue_.begin(), queue_.begin() + mid));

  std::move(queue_.begin() + mid, queue_.end(), queue_.begin());
  queue_.resize(queue_.size() - mid);
  return popped;
}

size_t ThunkExecutor::LifoReadyQueue::Size() const { return queue_.size(); }

bool ThunkExecutor::LifoReadyQueue::Empty() const { return queue_.empty(); }

ThunkExecutor::LifoReadyQueue
ThunkExecutor::LifoReadyQueue::CreateEmptyReadyQueue() const {
  return LifoReadyQueue(absl::Span<const NodeId>());
}

ThunkExecutor::PriorityReadyQueue::PriorityReadyQueue(
    absl::Span<const NodeDef> nodes_defs, absl::Span<const NodeId> ready_nodes)
    : nodes_defs_(nodes_defs),
      queue_(ready_nodes.begin(), ready_nodes.end(), Compare{nodes_defs}) {}

void ThunkExecutor::PriorityReadyQueue::Push(NodeId id) { queue_.push(id); }

ThunkExecutor::NodeId ThunkExecutor::PriorityReadyQueue::Pop() {
  DCHECK(!Empty()) << "Queue must not be empty";
  NodeId id = queue_.top();
  queue_.pop();
  return id;
}

ThunkExecutor::PriorityReadyQueue ThunkExecutor::PriorityReadyQueue::PopHalf() {
  DCHECK(!Empty()) << "Queue must not be empty";
  int64_t keep_top_nodes = queue_.size() / 2;

  // First pop nodes with highest priority from the queue.
  PriorityReadyQueue popped(nodes_defs_, {});
  while (keep_top_nodes-- > 0) {
    popped.queue_.push(queue_.top());
    queue_.pop();
  }

  // Swap popped nodes with remaining nodes, to return to the caller nodes with
  // smaller priorities, and keep higher priority nodes in the queue.
  popped.queue_.swap(queue_);

  return popped;
}

size_t ThunkExecutor::PriorityReadyQueue::Size() const { return queue_.size(); }

bool ThunkExecutor::PriorityReadyQueue::Empty() const { return queue_.empty(); }

ThunkExecutor::PriorityReadyQueue
ThunkExecutor::PriorityReadyQueue::CreateEmptyReadyQueue() const {
  return PriorityReadyQueue(nodes_defs_, {});
}

}  // namespace xla::cpu
