/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_INTERFACE_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_INTERFACE_H_

#include <memory>

#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace tpu {
class TpuCore;
}  // namespace tpu

namespace tensorflow {
namespace tpu {

class TpuExecutorInterface
    : public stream_executor::internal::StreamExecutorInterface {
 public:
  template <typename T>
  using StatusOr = stream_executor::port::StatusOr<T>;

  class TemporaryDeviceMemory {
   public:
    virtual ~TemporaryDeviceMemory() {}
    virtual stream_executor::DeviceMemoryBase AsDeviceMemoryBase() const = 0;
  };

  virtual StatusOr<std::unique_ptr<TemporaryDeviceMemory>>
  CreateTemporaryDeviceMemory(int64 memory_space, int64 byte_offset,
                              int64 size) {
    LOG(FATAL) << "Unimplemented.";
  }

  virtual const TpuPlatformInterface& platform() const {
    LOG(FATAL) << "Unimplemented.";
  }

  virtual TpuPlatformInterface& platform() { LOG(FATAL) << "Unimplemented."; }
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_EXECUTOR_INTERFACE_H_
