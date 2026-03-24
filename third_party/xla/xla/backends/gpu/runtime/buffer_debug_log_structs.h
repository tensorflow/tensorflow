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

// Output of float checker for a single buffer.
struct FloatCheckResult {
  uint32_t nan_count;
  uint32_t inf_count;
  uint32_t zero_count;
  // 4B of unused padding here to align the double values.
  double min_value;
  double max_value;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const FloatCheckResult& result) {
    absl::Format(&sink,
                 "{nan_count: %u, inf_count: %u, zero_count: %u, "
                 "min_value: %f, max_value: %f}",
                 result.nan_count, result.inf_count, result.zero_count,
                 result.min_value, result.max_value);
  }
};

// The struct layout must match on both host and device.
static_assert(_Alignof(FloatCheckResult) == _Alignof(double));
static_assert(sizeof(FloatCheckResult) ==
              sizeof(uint32_t) * 4 + sizeof(double) * 2);
static_assert(offsetof(FloatCheckResult, nan_count) == 0);
static_assert(offsetof(FloatCheckResult, inf_count) == sizeof(uint32_t));
static_assert(offsetof(FloatCheckResult, zero_count) == sizeof(uint32_t) * 2);
static_assert(offsetof(FloatCheckResult, min_value) == sizeof(uint32_t) * 4);
static_assert(offsetof(FloatCheckResult, max_value) ==
              sizeof(uint32_t) * 4 + sizeof(double));

// A single entry in the BufferDebugLog for float checks.
struct BufferDebugFloatCheckEntry {
  // An ID that uniquely identifies a log entry within a HLO module execution.
  BufferDebugLogEntryId entry_id;
  // 4B of unused padding here to satisfy FloatCheckResult alignment.
  FloatCheckResult result;

  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            const BufferDebugFloatCheckEntry& entry) {
    absl::Format(&sink, "{entry_id: %v, result: %v}", entry.entry_id.value(),
                 entry.result);
  }
};

// The struct layout must match on both host and device.
static_assert(_Alignof(BufferDebugFloatCheckEntry) == _Alignof(double));
static_assert(sizeof(BufferDebugFloatCheckEntry) ==
              sizeof(uint32_t) * 2 + sizeof(FloatCheckResult));
static_assert(offsetof(BufferDebugFloatCheckEntry, entry_id) == 0);
static_assert(offsetof(BufferDebugFloatCheckEntry, result) ==
              sizeof(uint32_t) * 2);

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
