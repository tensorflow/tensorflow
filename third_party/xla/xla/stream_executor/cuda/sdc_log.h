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
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/sdc_log_structs.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"

namespace stream_executor::cuda {

// A device memory buffer that holds a SdcLogHeader and a variable number of
// SdcLogEntry structs.
class SdcLog {
 public:
  // Returns the number of bytes required to store a log with `entries`
  // entries.
  static constexpr size_t RequiredSizeForEntries(size_t entries) {
    return sizeof(xla::gpu::SdcLogHeader) +
           sizeof(xla::gpu::SdcLogEntry) * entries;
  }

  // Initializes an empty `SdcLog` using a `log_buffer` allocated in device
  // memory.
  //
  // `log_buffer` must be allocated in memory of the same device `stream` is
  // associated with. `log_buffer` must outlive the returned `SdcLog`.
  //
  // Contents of the log can be retrieved with `SdcLog::ReadFromDevice`.
  //
  // Fails with `absl::StatusCode::kInvalidArgument` if `log_buffer` is too
  // small to hold any entries.
  static absl::StatusOr<SdcLog> CreateOnDevice(
      Stream& stream, DeviceMemory<uint8_t> log_buffer);

  // Reads the header from the device log.
  //
  // `stream` must be associated with the same device as the one used to create
  // the log.
  absl::StatusOr<xla::gpu::SdcLogHeader> ReadHeaderFromDevice(
      Stream& stream) const;

  // Reads all entries from the device log into host memory.
  //
  // Returned vector contains all initialized entries. If the log overflowed,
  // excess elements are silently discarded.
  //
  // `stream` must be associated with the same device as the one used to create
  // the log.
  absl::StatusOr<std::vector<xla::gpu::SdcLogEntry>> ReadFromDevice(
      Stream& stream) const;

  // Returns a view of the `SdcLogHeader`.
  //
  // The returned `DeviceMemory` gets invalidated when the `SdcLog` is
  // destroyed.
  DeviceMemory<xla::gpu::SdcLogHeader> GetDeviceHeader() const {
    return DeviceMemory<xla::gpu::SdcLogHeader>(
        memory_.GetByteSlice(0, sizeof(xla::gpu::SdcLogHeader)));
  }

  // Returns a view of the `SdcLogEntry` array.
  //
  // The returned `DeviceMemory` gets invalidated when the `SdcLog` is
  // destroyed.
  DeviceMemory<xla::gpu::SdcLogEntry> GetDeviceEntries() const {
    return DeviceMemory<xla::gpu::SdcLogEntry>(
        memory_.GetByteSlice(sizeof(xla::gpu::SdcLogHeader),
                             memory_.size() - sizeof(xla::gpu::SdcLogHeader)));
  }

 private:
  explicit SdcLog(DeviceMemory<uint8_t> memory) : memory_(memory) {}

  DeviceMemory<uint8_t> memory_;
};

}  // namespace stream_executor::cuda

#endif  // XLA_STREAM_EXECUTOR_CUDA_SDC_LOG_H_
