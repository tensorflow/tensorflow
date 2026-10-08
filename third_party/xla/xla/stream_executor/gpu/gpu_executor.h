/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/multicast_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_common.h"

namespace stream_executor::gpu {

class GpuStream;

// Intermediate implementation class for StreamExecutors that are used with
// GPUs.
class GpuExecutor : public StreamExecutorCommon {
 public:
  GpuExecutor(Platform* platform, int device_ordinal)
      : StreamExecutorCommon(platform), device_ordinal_(device_ordinal) {}

  int device_ordinal() const override { return device_ordinal_; };

  virtual absl::StatusOr<std::unique_ptr<MulticastMemory>>
  CreateMulticastMemory(uint64_t size, int num_devices) const {
    return absl::UnimplementedError(
        "CreateMulticastMemory is not implemented.");
  };

  virtual bool is_multicast_supported() const { return false; }

  // Returns the allocation range that contains the given pointer.
  virtual absl::StatusOr<DeviceAddressBase> GetAllocationRange(
      void* ptr) const {
    return absl::UnimplementedError("GetAllocationRange is not implemented.");
  }

  // Exports the given memory as a fabric handle that can be imported by another
  // host via `ImportFabricHandle`. `ptr` must have been allocated with VMM API.
  //
  // Note: The returned handle represents the entire allocation containing
  // `ptr`, rather than starting at `ptr`.
  virtual absl::StatusOr<std::string> ExportFabricHandle(void* ptr) const {
    return absl::UnimplementedError("ExportFabricHandle is not implemented.");
  }

  // Imports a fabric handle that had been exported by another host via
  // `ExportFabricHandle`. The returned memory can be free'd with `Deallocate`.
  virtual absl::StatusOr<DeviceAddressBase> ImportFabricHandle(
      absl::string_view serialized) {
    return absl::UnimplementedError("ImportFabricHandle is not implemented.");
  }

 private:
  // The device ordinal value that this executor was initialized with; recorded
  // for use in getting device metadata. Immutable post-initialization.
  int device_ordinal_;

  GpuExecutor(const GpuExecutor&) = delete;
  void operator=(const GpuExecutor&) = delete;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
