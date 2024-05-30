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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_STREAM_INTERFACE_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_STREAM_INTERFACE_H_

#include "absl/status/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_common.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_interface.h"

namespace tensorflow {
namespace tpu {

class TpuStreamInterface : public stream_executor::StreamCommon {
 public:
  explicit TpuStreamInterface(stream_executor::StreamExecutor* executor)
      : StreamCommon(executor) {}
  virtual bool IsSameSharedMemoryLocation(TpuStreamInterface* other) = 0;
  virtual absl::Status EnqueueOnTpuDeviceSendRecvLocal(
      stream_executor::DeviceMemoryBase send_buffer,
      stream_executor::DeviceMemoryBase recv_buffer) = 0;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_STREAM_INTERFACE_H_
