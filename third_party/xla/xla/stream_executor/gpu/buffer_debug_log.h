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

#ifndef XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_LOG_H_
#define XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_LOG_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/buffer_debug_log.pb.h"
#include "xla/backends/gpu/runtime/buffer_debug_log_structs.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::gpu {

// A wrapper over a device memory buffer used to store debug info about contents
// of buffers (e.g. checksums).
//
// It holds a BufferDebugLogHeader and a variable number of BufferDebugLogEntry
// structs.
class BufferDebugLog {
 public:
  // Returns the number of bytes required to store a log with `entries`
  // entries.
  static constexpr size_t RequiredSizeForEntries(size_t entries) {
    return sizeof(xla::gpu::BufferDebugLogHeader) +
           sizeof(xla::gpu::BufferDebugLogEntry) * entries;
  }

  // Initializes an empty `BufferDebugLog` using a `log_buffer` allocated in
  // device memory.
  //
  // `log_buffer` must be allocated in memory of the same device `stream` is
  // associated with. `log_buffer` must outlive the returned `BufferDebugLog`.
  //
  // Contents of the log can be retrieved with `BufferDebugLog::ReadFromDevice`.
  //
  // Fails with `absl::StatusCode::kInvalidArgument` if `log_buffer` is too
  // small to hold any entries.
  static absl::StatusOr<BufferDebugLog> CreateOnDevice(
      Stream& stream, DeviceMemory<uint8_t> log_buffer);

  // Creates a `BufferDebugLog` from an already initialized device memory
  // buffer.
  //
  // `log_buffer` must contain an initialized `BufferDebugLogHeader`.
  static BufferDebugLog FromDeviceMemoryUnchecked(
      DeviceMemory<uint8_t> log_buffer) {
    return BufferDebugLog(log_buffer);
  }

  // Reads the header from the device log.
  //
  // `stream` must be associated with the same device as the one used to create
  // the log.
  absl::StatusOr<xla::gpu::BufferDebugLogHeader> ReadHeaderFromDevice(
      Stream& stream) const;

  // Reads all entries from the device log into host memory.
  //
  // Returned vector contains all initialized entries. If the log overflowed,
  // excess elements are silently discarded.
  //
  // `stream` must be associated with the same device as the one used to create
  // the log.
  absl::StatusOr<std::vector<xla::gpu::BufferDebugLogEntry>> ReadFromDevice(
      Stream& stream) const;

  // Returns a view of the `BufferDebugLogHeader`.
  //
  // The returned `DeviceMemory` gets invalidated when the `BufferDebugLog` is
  // destroyed.
  DeviceMemory<xla::gpu::BufferDebugLogHeader> GetDeviceHeader() const {
    return DeviceMemory<xla::gpu::BufferDebugLogHeader>(
        memory_.GetByteSlice(0, sizeof(xla::gpu::BufferDebugLogHeader)));
  }

  // Returns a view of the `BufferDebugLogEntry` array.
  //
  // The returned `DeviceMemory` gets invalidated when the `BufferDebugLog` is
  // destroyed.
  DeviceMemory<xla::gpu::BufferDebugLogEntry> GetDeviceEntries() const {
    return DeviceMemory<xla::gpu::BufferDebugLogEntry>(memory_.GetByteSlice(
        sizeof(xla::gpu::BufferDebugLogHeader),
        memory_.size() - sizeof(xla::gpu::BufferDebugLogHeader)));
  }

 private:
  explicit BufferDebugLog(DeviceMemory<uint8_t> memory) : memory_(memory) {}

  DeviceMemory<uint8_t> memory_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_BUFFER_DEBUG_LOG_H_
