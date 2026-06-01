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

#include "xla/backends/gpu/runtime/traced_command_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"

namespace xla::gpu {

TracedCommandBuffer::TracedCommandBuffer(const Command* trace_cmd,
                                         Command::BufferUses buffers,
                                         int64_t capacity)
    : trace_cmd_(trace_cmd), capacity_(capacity), entries_(capacity) {
  CHECK_GT(capacity, 0) << "capacity must be larger than 0";  // NOLINT
  // Collect unique buffer allocation indices in a set first and convert to
  // vector as flat hash set iteration has measurable overheads.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices;
  for (auto& buffer : buffers) {
    allocs_indices.insert(buffer.slice().index());
  }
  allocs_indices_.assign(allocs_indices.begin(), allocs_indices.end());
}

// Shifts the entry at index `i` to the right and returns the entry at index 0.
TracedCommandBuffer::Entry& TracedCommandBuffer::ShiftRight(size_t i) {
  if (i == 0) {
    return entries_[0];
  }

  Entry entry = std::move(entries_[i]);
  do {
    entries_[i] = std::move(entries_[i - 1]);
  } while (--i > 0);

  return entries_[0] = std::move(entry);
}

// Returns the index of the first empty slot or the least recently used occupied
// slot.
int TracedCommandBuffer::FindEvictableSlot() const {
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i].state == State::kEmpty) {
      return static_cast<int>(i);
    }
  }
  for (int i = static_cast<int>(entries_.size()) - 1; i >= 0; --i) {
    if (entries_[i].state == State::kOccupied) {
      return i;
    }
  }
  return -1;
}

// Returns the index of the matching entry if it exists.
int TracedCommandBuffer::FindMatchingEntry(
    absl::Span<const se::DeviceAddressBase> allocs) const {
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i].state != State::kEmpty &&
        absl::c_equal(entries_[i].recorded_allocs, allocs)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// Returns the index of the reserved tracing entry if it exists.
int TracedCommandBuffer::FindReservedTracingEntry(
    absl::Span<const se::DeviceAddressBase> allocs) const {
  for (size_t i = 0; i < entries_.size(); ++i) {
    if (entries_[i].state == State::kTracing &&
        absl::c_equal(entries_[i].recorded_allocs, allocs)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

// Reserves a slot for tracing and returns any evicted buffer to be destroyed.
std::unique_ptr<se::CommandBuffer> TracedCommandBuffer::ReserveSlot(
    int idx, absl::Span<const se::DeviceAddressBase> allocs) {
  Entry& slot = entries_[idx];
  std::unique_ptr<se::CommandBuffer> local_eviction = nullptr;
  if (slot.state == State::kOccupied) {
    local_eviction = std::move(slot.command_buffer);
  }
  slot.command_buffer.reset();
  slot.recorded_allocs.assign(allocs.begin(), allocs.end());
  slot.state = State::kTracing;
  return local_eviction;
}

absl::StatusOr<se::CommandBuffer*> TracedCommandBuffer::GetOrTraceCommandBuffer(
    const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
    se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
    se::StreamPriority priority) {
  // Collect memory addresses for relevant allocations.
  absl::InlinedVector<se::DeviceAddressBase, 4> allocs;
  allocs.reserve(allocs_indices_.size());
  for (auto& index : allocs_indices_) {
    allocs.emplace_back(buffer_allocation->GetDeviceAddress(index));
  }

  // Keep track of any evicted buffer under lock and destroy it outside the
  // lock and after tracing is complete to avoid context thrashing.
  std::unique_ptr<se::CommandBuffer> local_eviction = nullptr;

  while (true) {
    absl::MutexLock lock(&mutex_);

    int match_idx = FindMatchingEntry(allocs);
    // Cache hit, retrieve and update position or wait for tracing to complete.
    if (match_idx != -1) {
      Entry& entry = entries_[match_idx];
      if (entry.state == State::kOccupied) {
        VLOG(6) << "Command buffer trace cache hit for command "
                << trace_cmd_->ToString(0);
        return ShiftRight(match_idx).command_buffer.get();
      }
      if (entry.state == State::kTracing) {
        cv_.Wait(&mutex_);
        continue;
      }
    }

    // Select eviction target on cache miss.
    int evict_idx = FindEvictableSlot();
    if (evict_idx == -1) {
      // No eviction targets available, wait for tracing to complete.
      cv_.Wait(&mutex_);
      continue;
    }

    // Reserve cache slot and extract evicted buffer.
    local_eviction = ReserveSlot(evict_idx, allocs);
    break;
  }

  // Trace completely outside of lock.
  absl::StatusOr<std::unique_ptr<se::CommandBuffer>> trace_result =
      se::TraceCommandBufferFactory::Create(executor, stream, trace);

  absl::Status priority_status = absl::OkStatus();
  if (trace_result.ok() && priority != se::StreamPriority::Default) {
    priority_status = (*trace_result)->SetPriority(priority);
  }

  {
    absl::MutexLock lock(&mutex_);
    // Look up the reserved slot again, it may have changed since we released
    // the lock.
    int found_idx = FindReservedTracingEntry(allocs);

    if (ABSL_PREDICT_TRUE(found_idx != -1)) {
      if (trace_result.ok() && priority_status.ok()) {
        entries_[found_idx].command_buffer = std::move(*trace_result);
        entries_[found_idx].state = State::kOccupied;
        se::CommandBuffer* cmd_buf = entries_[found_idx].command_buffer.get();
        ShiftRight(found_idx);
        cv_.SignalAll();
        VLOG(6) << "Command buffer trace cache created new item for command "
                << trace_cmd_->ToString(0);
        return cmd_buf;
      }

      // Failed to trace, so clear the slot and signal any waiters.
      entries_[found_idx].state = State::kEmpty;
      entries_[found_idx].recorded_allocs.clear();
      entries_[found_idx].command_buffer.reset();
      cv_.SignalAll();
      return trace_result.status().ok() ? priority_status
                                        : trace_result.status();
    }
  }

  // Should be unreachable.
  return absl::InternalError(
      "TracedCommandBuffer cache entry was lost during tracing");
}

}  // namespace xla::gpu
