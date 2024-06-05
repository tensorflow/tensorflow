#include "xla/stream_executor/platform.h"
/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_INTERFACE_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_INTERFACE_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tpu/tpu_platform_interface.h"
#include "xla/stream_executor/tpu/tpu_topology.h"

namespace tpu {
class TpuCore;
}  // namespace tpu

namespace tensorflow {
namespace tpu {

class TpuExecutorInterface : public stream_executor::StreamExecutor {
 public:
  explicit TpuExecutorInterface(stream_executor::Platform* platform)
      : StreamExecutor(platform) {}

  class TemporaryDeviceMemory {
   public:
    virtual ~TemporaryDeviceMemory() {}
    virtual stream_executor::DeviceMemoryBase AsDeviceMemoryBase() const = 0;
  };

  virtual absl::StatusOr<std::unique_ptr<TemporaryDeviceMemory>>
  CreateTemporaryDeviceMemory(int64_t memory_space, int64_t byte_offset,
                              int64_t size) {
    LOG(FATAL) << "Unimplemented.";
  }

  virtual const TpuPlatformInterface& platform() const {
    LOG(FATAL) << "Unimplemented.";
  }

  virtual TpuPlatformInterface& platform() { LOG(FATAL) << "Unimplemented."; }

  virtual TpuCoreLocationExternal GetCoreLocationExternal() const {
    LOG(FATAL) << "Unimplemented.";
  }

  virtual absl::Status UnloadAllPrograms() { LOG(FATAL) << "Unimplemented."; }

  virtual absl::Status EnqueueCompactionOnStreamForHbm(
      stream_executor::Stream* compaction_stream) {
    LOG(FATAL) << "Unimplemented.";
  }
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_INTERFACE_H_
