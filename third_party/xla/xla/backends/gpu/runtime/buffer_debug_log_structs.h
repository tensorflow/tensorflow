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

#ifndef XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_STRUCTS_H_
#define XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_STRUCTS_H_

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/buffer_debug_log.pb.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

TSL_LIB_GTL_DEFINE_INT_TYPE(BufferDebugLogEntryId, uint32_t)

struct BufferDebugLogEntry {
  // An ID that uniquely identifies a log entry within a HLO module execution.
  BufferDebugLogEntryId entry_id;
  uint32_t value;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const BufferDebugLogEntry& entry) {
    absl::Format(&sink, "{entry_id: %v, value: %u}", entry.entry_id.value(),
                 entry.value);
  }

  bool operator==(const BufferDebugLogEntry& other) const {
    return std::tie(entry_id, value) == std::tie(other.entry_id, other.value);
  }

  bool operator!=(const BufferDebugLogEntry& other) const {
    return !(*this == other);
  }
};

// The struct layout must match on both host and device.
static_assert(_Alignof(BufferDebugLogEntry) == _Alignof(uint32_t));
static_assert(sizeof(BufferDebugLogEntry) == sizeof(uint32_t) * 2);
static_assert(offsetof(BufferDebugLogEntry, entry_id) == 0);
static_assert(offsetof(BufferDebugLogEntry, value) == sizeof(uint32_t));

struct BufferDebugFloatCheckEntry {
  // An ID that uniquely identifies a log entry within a HLO module execution.
  BufferDebugLogEntryId entry_id;
  uint32_t nan_count;
  uint32_t inf_count;
  uint32_t zero_count;

  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            const BufferDebugFloatCheckEntry& entry) {
    absl::Format(&sink,
                 "{entry_id: %v, nan_count: %u, inf_count: %u, zero_count: %u}",
                 entry.entry_id.value(), entry.nan_count, entry.inf_count,
                 entry.zero_count);
  }

  bool operator==(const BufferDebugFloatCheckEntry& other) const {
    return std::tie(entry_id, nan_count, inf_count, zero_count) ==
           std::tie(other.entry_id, other.nan_count, other.inf_count,
                    other.zero_count);
  }

  bool operator!=(const BufferDebugFloatCheckEntry& other) const {
    return !(*this == other);
  }
};

// The struct layout must match on both host and device.
static_assert(_Alignof(BufferDebugFloatCheckEntry) == _Alignof(uint32_t));
static_assert(sizeof(BufferDebugFloatCheckEntry) == sizeof(uint32_t) * 4);
static_assert(offsetof(BufferDebugFloatCheckEntry, entry_id) == 0);
static_assert(offsetof(BufferDebugFloatCheckEntry, nan_count) ==
              sizeof(uint32_t));

struct BufferDebugLogHeader {
  // The first entry in `BufferDebugLogEntry` following the header that has not
  // been written to. May be bigger than `capacity` if the log was truncated.
  uint32_t write_idx;
  // The number of `BufferDebugLogEntry` structs the log can hold.
  uint32_t capacity;
};

// The struct layout must match on both host and device.
static_assert(_Alignof(BufferDebugLogHeader) == _Alignof(uint32_t));
static_assert(sizeof(BufferDebugLogHeader) == sizeof(uint32_t) * 2);
static_assert(offsetof(BufferDebugLogHeader, write_idx) == 0);
static_assert(offsetof(BufferDebugLogHeader, capacity) == sizeof(uint32_t));

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_BUFFER_DEBUG_LOG_STRUCTS_H_
