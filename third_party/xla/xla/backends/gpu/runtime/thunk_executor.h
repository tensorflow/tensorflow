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

#ifndef XLA_BACKENDS_GPU_RUNTIME_THUNK_EXECUTOR_H_
#define XLA_BACKENDS_GPU_RUNTIME_THUNK_EXECUTOR_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

// The thunk executor is responsible for executing a thunk sequence on the
// underlying GPU device. An XLA:GPU executable itself is essentially a sequence
// of thunks, where each individual thunk can have a nested sequence of its own
// (e.g. for control flow thunks like WhileThunk). The thunk executor is an
// interpreter that executes thunks and optionally can track execution progress.
class ThunkExecutor {
 public:
  // Forward declaration. See definition below.
  class ScopedProgressTracker;

  explicit ThunkExecutor(ThunkSequence thunks);

  ThunkExecutor(ThunkExecutor&&) = default;
  ThunkExecutor& operator=(ThunkExecutor&&) = default;

  // Thunk execution lifecycle operations.
  absl::Status Prepare(const Thunk::PrepareParams& params);
  absl::Status Initialize(const Thunk::InitializeParams& params);
  absl::Status ExecuteOnStream(const Thunk::ExecuteParams& params);

  absl::Status WalkNested(
      absl::FunctionRef<absl::Status(const Thunk*)> callback) const;

  const ThunkSequence& thunks() const { return thunks_; }
  ThunkSequence& thunks() { return thunks_; }

 private:
  ThunkSequence thunks_;
};

//===----------------------------------------------------------------------===//
// Tracking Thunk execution progress.
//===----------------------------------------------------------------------===//

// Sometimes XLA:GPU executable can get stuck on device because of buggy
// kernels or problems with the GPU itself, i.e. a kernel can go into an
// infinite loop and never complete its execution, or a collective kernel can
// wait for data to arrive forever, because the communication channel dropped
// the packet and it will never arrive. Such kernels will lead to XLA getting
// stuck at some point later when XLA hits the device queue limit. The Thunk
// that will hit the queue limit and will be visible in the stack trace is not
// the kernel that caused the hang to happen. XLA can optionally add progress
// tracking to all thunks via recording events after every `Thunk` execution.
//
// We rely on the fact that `GpuExecutable` itself and all `Thunks` with nested
// control flow use `ThunkExecutor` for executing nested thunks, to track the
// progress of the GPU executable itself. We rely on one scoped progress tracker
// which is installed for the current thread using a thread-local mechanism, and
// it is guaranteed to work, because thunk sequence recording is
// single-threaded.
class ThunkExecutor::ScopedProgressTracker {
 public:
  // Thunk execution record: the thunk's global index, the wall-clock time it
  // was launched, and its human-readable name.
  struct ThunkExecution {
    size_t index;
    absl::Time executed;
    absl::string_view name;
  };

  ~ScopedProgressTracker();

  ScopedProgressTracker(ScopedProgressTracker&&) = default;
  ScopedProgressTracker& operator=(ScopedProgressTracker&&) = default;

  size_t num_thunks() const {
    absl::MutexLock lock(&events_->mu);
    return events_->map.size();
  }

  // Returns the last `n` thunks that successfully completed execution.
  std::vector<ThunkExecution> LastCompletedThunks(size_t n);

  // Returns the first `n` thunks that were executed but didn't complete
  // execution. These are the deadlock suspects.
  std::vector<ThunkExecution> FirstPendingThunks(size_t n);

  // Returns the last `n` thunks that were executed but didn't complete
  // execution. These tell how far XLA executable got before deadlock.
  std::vector<ThunkExecution> LastPendingThunks(size_t n);

 private:
  friend class ThunkExecutor;
  friend absl::StatusOr<ThunkExecutor::ScopedProgressTracker>
  InstallProgressTracker(se::StreamExecutor*, const ThunkExecutor&);

  // We use global indexing across all nested thunks in a sequence, not only the
  // top-level thunks executed by `ThunkExecutor`. This is different from the
  // index that is used by `ThunkExecutor` progress v-logging. We track the
  // time when a thunk was executed, but not when it completed.
  struct ThunkProgress {
    size_t index;
    absl::Time executed;
    std::unique_ptr<se::Event> event;
  };

  struct ThunkEvents {
    explicit ThunkEvents(absl::flat_hash_map<const Thunk*, ThunkProgress> map)
        : map(std::move(map)) {}

    absl::Mutex mu;
    absl::flat_hash_map<const Thunk*, ThunkProgress> map ABSL_GUARDED_BY(mu);
  };

  // Each thread can have at most one progress tracker installed and it is
  // automatically removed when the scoped progress tracker goes out of scope.
  static thread_local ThunkEvents* installed_progress_tracker;

  explicit ScopedProgressTracker(
      absl::flat_hash_map<const Thunk*, ThunkProgress> progress_map);

  // Collects up to `n` thunks matching `status`, sorted by executed time.
  // If `most_recent_first` is true, returns most recently executed thunks
  // first (descending order); otherwise returns earliest executed thunks first
  // (ascending order). Lazily checks event status to avoid polling all thunks.
  std::vector<ThunkExecution> CollectThunks(se::Event::Status status,
                                            bool most_recent_first, size_t n);

  std::unique_ptr<ThunkEvents> events_;
};

// Installs a progress tracker for the given sequential thunk in the current
// thread. The progress tracker will be automatically deactivated when it is
// destroyed. It is an error to install a progress tracker twice;
// trying to do so will lead to a process crash.
absl::StatusOr<ThunkExecutor::ScopedProgressTracker> InstallProgressTracker(
    se::StreamExecutor* stream_executor, const ThunkExecutor& executor);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_EXECUTOR_H_
