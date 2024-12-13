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

#ifndef XLA_BACKENDS_CPU_RUNTIME_XNNPACK_PARALLEL_LOOP_RUNNER_H_
#define XLA_BACKENDS_CPU_RUNTIME_XNNPACK_PARALLEL_LOOP_RUNNER_H_

#include <cstddef>
#include <functional>

#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace xla::cpu {

// Parallel loop runner uses underlying Eigen ThreadPoolDevice to execute
// parallel loops providing implicit synchronization: the next parallel loop
// starts execution only after all tasks from the previous loop are completed.
//
// Scheduled parallel loops execute asynchronously without blocking the caller
// thread. It is the user's responsibility to ensure that all values captured by
// the task are valid until the task is completed.
//
// Parallel loop runner is an implementation of the `pthreadpool` API adaptor
// for XLA:CPU runtime.
//
// WARNING: ParallelLoopRunner is not thread-safe, and must be externally
// synchronized by the user.
class ParallelLoopRunner {
 public:
  explicit ParallelLoopRunner(Eigen::ThreadPoolDevice* device);

  // Takes ownership of the runner and returns a done event. After the done
  // event is transferred to the caller, it is illegal to schedule more parallel
  // loops on the moved-from runner.
  static tsl::AsyncValueRef<tsl::Chain> TakeDoneEvent(
      ParallelLoopRunner&& runner);

  using Task1D = std::function<void(size_t offset, size_t extent)>;

  // This function implements a parallel version of a following loop:
  //
  //   for (size_t i = 0; i < range; i += tile)
  //     task(i, std::min(range - i, tile));
  void Parallelize(size_t range, size_t tile, Task1D task);

  tsl::AsyncValueRef<tsl::Chain> done_event() const { return done_event_; }
  Eigen::ThreadPoolDevice* device() const { return device_; }

 private:
  // Async value that signals completion of the last scheduled parallel loop.
  tsl::AsyncValueRef<tsl::Chain> done_event_;

  Eigen::ThreadPoolDevice* device_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_PARALLEL_LOOP_RUNNER_H_
