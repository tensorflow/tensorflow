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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_STREAM_INTERFACE_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_STREAM_INTERFACE_H_

#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace tensorflow {
namespace tpu {

class TpuStreamInterface : public stream_executor::internal::StreamInterface {
 public:
  using Status = stream_executor::port::Status;

  virtual bool IsSameSharedMemoryLocation(TpuStreamInterface* other) = 0;
  virtual Status EnqueueOnTpuDeviceSendRecvLocal(
      stream_executor::DeviceMemoryBase send_buffer,
      stream_executor::DeviceMemoryBase recv_buffer) = 0;
};

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_STREAM_INTERFACE_H_
