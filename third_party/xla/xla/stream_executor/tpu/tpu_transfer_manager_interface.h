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

#ifndef XLA_STREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_INTERFACE_H_
#define XLA_STREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_INTERFACE_H_

#include <deque>

#include "xla/service/transfer_manager.h"
#include "xla/status.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tpu/noncopyable_buffer.h"

namespace xla {

class TpuTransferManagerInterface : public xla::TransferManager {
 public:
  virtual Status TransferBuffersToInfeed(
      se::StreamExecutor* executor,
      const std::deque<tensorflow::tpu::NoncopyableBuffer>& buffers) = 0;

  virtual Status LinearizeToBuffers(
      const LiteralSlice& literal, const Shape& device_shape,
      std::deque<tensorflow::tpu::NoncopyableBuffer>* buffers) = 0;

  static TpuTransferManagerInterface* GetRegisteredTpuTransferManager();
};

}  // namespace xla

#endif  // XLA_STREAM_EXECUTOR_TPU_TPU_TRANSFER_MANAGER_INTERFACE_H_
