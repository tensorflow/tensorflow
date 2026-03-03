/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SEQUENTIAL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_SEQUENTIAL_THUNK_H_

#include <cstddef>
#include <memory>
#include <string>
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
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

// A thunk that wraps a list of sub-thunks. Executing this thunk executes all
// the sub-thunks sequentially. This is useful to implement instructions that
// require multiple kernel launches or library calls.
class SequentialThunk : public Thunk {
 public:
  // Forward declaration. See definition below.
  class ScopedProgressTracker;

  SequentialThunk(ThunkInfo thunk_info, ThunkSequence thunks);
  SequentialThunk(const SequentialThunk&) = delete;
  SequentialThunk& operator=(const SequentialThunk&) = delete;

  ThunkSequence& thunks() { return thunks_; }
  const ThunkSequence& thunks() const { return thunks_; }
  std::string ToString(int indent) const override;

  absl::Status Prepare(const PrepareParams& params) override;
  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  absl::Status WalkNested(Walker callback) override;
  absl::Status TransformNested(Transformer callback) override;

  absl::StatusOr<ThunkProto> ToProto() const override;

  static absl::StatusOr<std::unique_ptr<SequentialThunk>> FromProto(
      ThunkInfo thunk_info, const SequentialThunkProto& thunk_proto,
      const Deserializer& deserializer);

  // Converts a Thunk into a SequentialThunk. If the input is already a
  // SequentialThunk, the returned value is the downcasted input.
  //
  // The new thunk, if created, will use a default-initialized ThunkInfo.
  static std::unique_ptr<SequentialThunk> FromThunk(
      std::unique_ptr<Thunk> thunk);

 private:
  // The list of sub-thunks.
  ThunkSequence thunks_;
};

//===----------------------------------------------------------------------===//
// Tracking XLA execution progress.
//===----------------------------------------------------------------------===//

// Some times XLA:GPU executable can get stuck on device because of buggy
// kernels or problems with the GPU itself, i.e. a kernel can go into infinite
// loop and never complete its execution, or a collective kernel can wait for
// data to arrive forever, because the communication channel dropped the packet
// and it will never arrive. Such kernels will lead to XLA getting stuck at some
// point later when XLA will hit the device queue limit. The Thunk that will
// hit the queue limit and will be visible in the stack trace is not the kernel
// that caused the hang to happen. XLA can optionally add progress tracking to
// all thunks via recording events after every `Thunk::ExecuteOnStream`.
//
// We rely on the fact that `GpuExecutable` itself and all `Thunks` with nested
// control flow use `SequentialThunk` for execution (we can call sequential
// thunk and executor, and this is a good refactoring to be consistent with
// commands execution) to track the progress of GPU executable itself. We rely
// one scoped progress tracker which is installed for the current thread using
// thread local mechanism, and it is guaranteed to work, because thunk sequence
// recording is single-threaded.
class SequentialThunk::ScopedProgressTracker {
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
  // execution. These tells how far XLA executable got before deadlock.
  std::vector<ThunkExecution> LastPendingThunks(size_t n);

 private:
  friend class SequentialThunk;
  friend absl::StatusOr<SequentialThunk::ScopedProgressTracker>
  InstallProgressTracker(se::StreamExecutor*, const SequentialThunk&);

  // We use global indexing across all nested thunks in a sequence, not only the
  // top-level thunks executed by `SequentialThunk`. This is different from the
  // index that is used by `SequentialThunk` progress v-logging. We track the
  // time when thunk was executed, but not when it completed.
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
  // automatically removed when scoped progress tracker gets out of scope.
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
// thread. Progress tracker will be automatically deactivated when the progress
// tracker will be destroyed. It is an error to install progress tracker twice,
// trying to do so will lead to process crash.
absl::StatusOr<SequentialThunk::ScopedProgressTracker> InstallProgressTracker(
    se::StreamExecutor* stream_executor, const SequentialThunk& thunk);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_SEQUENTIAL_THUNK_H_
