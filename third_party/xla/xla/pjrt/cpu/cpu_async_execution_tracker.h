/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_CPU_CPU_ASYNC_EXECUTION_TRACKER_H_
#define XLA_PJRT_CPU_CPU_ASYNC_EXECUTION_TRACKER_H_

#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {

class CpuAsyncExecutionTracker;

// RAII wrapper for an async execution. It reports the completion of the async
// execution to the tracker and acts as a helper to set the state of the execute
// event only if it is not set yet. Typically, either `SetStateConcrete()` or
// `SetError()` should be called before the destruction of
// `CpuScopedAsyncExecution`.
//
// Not thread-safe.
class CpuScopedAsyncExecution {
 public:
  // Opaque key that uniquely identifies an async execution for tracking
  // purposes.
  using Key = const void*;

  CpuScopedAsyncExecution(CpuAsyncExecutionTracker* tracker, int32_t launch_id,
                          Key key);
  CpuScopedAsyncExecution(CpuScopedAsyncExecution&& other);
  ~CpuScopedAsyncExecution();

  CpuScopedAsyncExecution(const CpuScopedAsyncExecution&) = delete;

  // Sets the state of the execution to a ready state. No-op if the execute
  // event is already set.
  void SetStateConcrete();

  // Sets the state of the execution to an error. No-op if the execute event is
  // already set.
  void SetError(absl::Status error);

 private:
  CpuAsyncExecutionTracker* tracker_;
  int32_t launch_id_;
  Key key_;
};

// Tracks async executions that have not finished yet. Upon destruction, the
// tracker will wait for all async executions to finish to help graceful
// teardown of the runtime state.
//
// Thread-safe.
class CpuAsyncExecutionTracker {
 public:
  using Key = CpuScopedAsyncExecution::Key;

  // Registers a new execution dispatched to a device.
  CpuScopedAsyncExecution NewAsyncExecution(
      int32_t launch_id, tsl::AsyncValueRef<CpuEvent> execute_event);

  // Sets the state of any executions with `launch_id` to an error. Returns true
  // if it succeeds to set the state. Returns false if all executions have been
  // removed or their execute event is already set.
  bool SetError(int32_t launch_id, absl::Status error);

  // Below is used by `CpuScopedAsyncExecution`.

  // Sets the state of the execute event to an error. Returns true if it
  // succeeds to set the state. Returns false if the execution has been removed
  // or the execute event is already set.
  void SetError(int32_t launch_id, Key key, absl::Status error);

  // Sets the state of the execute event to a ready state. Returns true if it
  // succeeds to set the state. Returns false if the execution has been removed
  // or the execute event is already set.
  void SetStateConcrete(int32_t launch_id, Key key);

  // Removes the execution from the tracker without setting the state of the
  // execute event.
  void RemoveAsyncExecution(int32_t launch_id, Key key);

 private:
  absl::Mutex mu_;

  // Maps launch_id to the execute event of async executions that have no
  // execute state set yet. The map value tracks multiple execute events from
  // async executions that use the same `launch_id`.
  absl::flat_hash_map<int32_t,
                      absl::flat_hash_map<Key, tsl::AsyncValueRef<CpuEvent>>>
      executions_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla

#endif  // XLA_PJRT_CPU_CPU_ASYNC_EXECUTION_TRACKER_H_
