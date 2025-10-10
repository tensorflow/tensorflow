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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SDC_LOG_STRUCTS_H_
#define XLA_BACKENDS_GPU_RUNTIME_SDC_LOG_STRUCTS_H_

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "xla/backends/gpu/runtime/sdc_buffer_id.h"

namespace xla::gpu {

struct SdcLogEntry {
  // An ID that uniquely identifies a thunk and its specific input or output
  // buffer.
  SdcBufferId entry_id;
  uint32_t checksum;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SdcLogEntry& entry) {
    absl::Format(&sink, "{entry_id: %v, checksum: %u}", entry.entry_id,
                 entry.checksum);
  }

  bool operator==(const SdcLogEntry& other) const {
    return std::tie(entry_id, checksum) ==
           std::tie(other.entry_id, other.checksum);
  }

  bool operator!=(const SdcLogEntry& other) const { return !(*this == other); }
};

// The struct layout must match on both host and device.
static_assert(_Alignof(SdcLogEntry) == _Alignof(uint32_t));
static_assert(sizeof(SdcLogEntry) == sizeof(uint32_t) * 2);
static_assert(offsetof(SdcLogEntry, entry_id) == 0);
static_assert(offsetof(SdcLogEntry, checksum) == sizeof(uint32_t));

struct SdcLogHeader {
  // The first entry in `SdcLogEntry` following the header that has not
  // been written to. May be bigger than `capacity` if the log was truncated.
  uint32_t write_idx;
  // The number of `SdcLogEntry` structs the log can hold.
  uint32_t capacity;
};

// The struct layout must match on both host and device.
static_assert(_Alignof(SdcLogHeader) == _Alignof(uint32_t));
static_assert(sizeof(SdcLogHeader) == sizeof(uint32_t) * 2);
static_assert(offsetof(SdcLogHeader, write_idx) == 0);
static_assert(offsetof(SdcLogHeader, capacity) == sizeof(uint32_t));

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_SDC_LOG_STRUCTS_H_
