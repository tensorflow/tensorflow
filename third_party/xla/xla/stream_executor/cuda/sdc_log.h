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

#ifndef XLA_STREAM_EXECUTOR_CUDA_SDC_LOG_H_
#define XLA_STREAM_EXECUTOR_CUDA_SDC_LOG_H_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::cuda {

struct SdcLogEntry {
  // An ID that uniquely identifies a thunk and its specific input or output
  // buffer.
  uint32_t entry_id;
  uint32_t checksum;
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

// A device memory buffer that holds a SdcLogHeader and a variable number of
// SdcLogEntry structs.
class SdcLog {
 public:
  // Uses `allocator` to allocate an empty `SdcLog` on the device with enough
  // capacity to hold `max_entries` of `SdcLogEntry` structs.
  //
  // `allocator` must be associated with the same device as `stream`, and must
  // outlive the returned `SdcLog`.
  //
  // Contents of the log can be retrieved with `SdcLog::ReadFromDevice`.
  static absl::StatusOr<SdcLog> CreateOnDevice(
      uint32_t max_entries, Stream& stream, DeviceMemoryAllocator& allocator);

  // Reads the header from the device log.
  //
  // `stream` must be associated with the same device as the one used to create
  // the log.
  absl::StatusOr<SdcLogHeader> ReadHeaderFromDevice(Stream& stream) const;

  // Reads all entries from the device log into host memory.
  //
  // Returned vector contains all initialized entries. If the log overflowed,
  // excess elements are silently discarded.
  //
  // `stream` must be associated with the same device as the one used to create
  // the log.
  absl::StatusOr<std::vector<SdcLogEntry>> ReadFromDevice(Stream& stream) const;

  // Returns a view of the `SdcLogHeader`.
  //
  // The returned `DeviceMemory` gets invalidated when the `SdcLog` is
  // destroyed.
  DeviceMemory<SdcLogHeader> GetDeviceHeader() const {
    return DeviceMemory<SdcLogHeader>(
        memory_->GetByteSlice(0, sizeof(SdcLogHeader)));
  }

  // Returns a view of the `SdcLogEntry` array.
  //
  // The returned `DeviceMemory` gets invalidated when the `SdcLog` is
  // destroyed.
  DeviceMemory<SdcLogEntry> GetDeviceEntries() const {
    return DeviceMemory<SdcLogEntry>(memory_->GetByteSlice(
        sizeof(SdcLogHeader), memory_->size() - sizeof(SdcLogHeader)));
  }

 private:
  explicit SdcLog(OwningDeviceMemory&& memory) : memory_(std::move(memory)) {}

  OwningDeviceMemory memory_;
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_SDC_LOG_H_
