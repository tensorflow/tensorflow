/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ASYNC_EXECUTION_H_
#define XLA_BACKENDS_GPU_RUNTIME_ASYNC_EXECUTION_H_

#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/object_pool.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

// AsyncExecution is a helper class to manage async execution of XLA thunks.
//
// XLA has several asynchronous operations (mostly communication operations,
// however it also supports asynchronous computations) that are decomposed into
// a pair of start and done operations:
//
// Example: asynchronous all reduce
//
//   %start = all-reduce-start(...), stream_id=comm-stream
//   ... compute kernels launched on compute stream
//   %done = all-reduce-done(%start)
//
// `%start` operation launches an all-reduce on a dedicated communication
// stream, and `%done` operation synchronizes it with a compute stream. By
// decomposing operation into start/done pairs XLA makes them "asynchronous" in
// a sense that they can run concurrently with the "main" compute stream.
//
// Simply synchronizing compute stream with a communication stream is not going
// to work, because it can create false dependencies between multiple async
// operations:
//
//   %start0 = all-reduce-start(...), stream_id=comm-stream
//   %start1 = all-reduce-start(...), stream_id=comm-stream
//   ... compute kernels launched on compute stream
//   %done0 = all-reduce-done(%start0)
//   %done1 = all-reduce-done(%start1)
//
// If `all-reduce-done` would simply synchronize compute stream with
// `comm-stream` it would create a false dependency between two unrelated all
// reduce operations.
//
// Async execution scope is always started by a `start` Thunk, and start thunk
// id is used throughout the runtime as async execution id.
class AsyncExecution {
 public:
  using EventPool = ObjectPool<std::unique_ptr<se::Event>>;

  // We need to know the thunk that starts an async execution, as we use its id
  // as a key in the execution scoped state and its profile annotation for
  // logging.
  explicit AsyncExecution(const Thunk* start_thunk);

  // An RAII guard that automatically records an event on a given async stream
  // when it goes out of scope.
  class ExecutionGuard {
   public:
    ~ExecutionGuard();

   private:
    friend class AsyncExecution;
    ExecutionGuard(se::Event* event, se::Stream* async_stream);
    se::Event* event_;          // owned by ExecutionScopedState
    se::Stream* async_stream_;  // owned by GpuExecutable
  };

  // Initializes async execution state on the given executor. Borrows an event
  // from the pool (creating one if needed) and stores it in the execution
  // scoped state. When the state is destroyed, the event is returned to the
  // pool.
  absl::Status Initialize(Thunk::ExecutionScopedState* state,
                          se::StreamExecutor* executor);

  // Starts an async execution on `async_stream`: creates a dependency from
  // `stream` to `async_stream` which guarantees that async operations launched
  // on `async_stream` will observe all prior operations launched on `stream`.
  // Start can be called at most once. Before the next Start can be called,
  // the async execution has to be completed with Done.
  absl::StatusOr<ExecutionGuard> Start(RunId run_id,
                                       Thunk::ExecutionScopedState* state,
                                       se::Stream* stream,
                                       se::Stream* async_stream);

  // Completes an async execution on `stream` by synchronizing with the event
  // recorded by the corresponding Start call.
  absl::Status Done(Thunk::ExecutionScopedState* state, se::Stream* stream);

 private:
  // Returns or creates an event pool for the given executor.
  EventPool& GetOrCreatePool(se::StreamExecutor* executor);

  const Thunk* start_thunk_;

  absl::Mutex mu_;
  absl::node_hash_map<se::StreamExecutor*, EventPool> event_pools_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ASYNC_EXECUTION_H_
