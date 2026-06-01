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
=============================================================================*/

#ifndef XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_BUFFER_H_
#define XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_BUFFER_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

// A cache for traced command buffers that will re-trace on change in buffer
// allocations that are relevant for `buffers` passed to constructor. We use a
// very simple most-recently-used cache of traced command buffers as in practice
// subsequent calls to XLA executable tend to reuse the same allocations.
class TracedCommandBuffer : public CommandState {
 public:
  explicit TracedCommandBuffer(const Command* trace_cmd,
                               Command::BufferUses buffers,
                               int64_t capacity = 16);

  // Returns cached command buffer traced using the same buffer addresses or
  // traces and caches a new command buffer using user provided callback.
  absl::StatusOr<se::CommandBuffer*> GetOrTraceCommandBuffer(
      const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
      se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
      se::StreamPriority priority = se::StreamPriority::Default);

 private:
  std::vector<BufferAllocation::Index> allocs_indices_;

  enum class State {
    kEmpty,
    kTracing,
    kOccupied,
  };

  struct Entry {
    std::vector<se::DeviceAddressBase> recorded_allocs;
    std::unique_ptr<se::CommandBuffer> command_buffer;
    State state = State::kEmpty;
  };
  const Command* trace_cmd_;
  int64_t capacity_;
  absl::Mutex mutex_;
  absl::CondVar cv_;
  std::vector<Entry> entries_ ABSL_GUARDED_BY(mutex_);

  // Cache manipulation functions.
  Entry& ShiftRight(size_t i) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  int FindEvictableSlot() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  int FindMatchingEntry(absl::Span<const se::DeviceAddressBase> allocs) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  int FindReservedTracingEntry(absl::Span<const se::DeviceAddressBase> allocs)
      const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
  std::unique_ptr<se::CommandBuffer> ReserveSlot(
      int idx, absl::Span<const se::DeviceAddressBase> allocs)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_TRACED_COMMAND_BUFFER_H_
