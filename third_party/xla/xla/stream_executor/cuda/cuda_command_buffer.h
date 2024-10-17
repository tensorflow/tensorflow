/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_

#include <memory>

#include "absl/log/log.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_executor.h"

namespace stream_executor::gpu {

class CudaCommandBuffer : public GpuCommandBuffer {
 public:
  static absl::StatusOr<std::unique_ptr<CudaCommandBuffer>> Create(
      Mode mode, GpuExecutor* parent);

 private:
  using GpuCommandBuffer::GpuCommandBuffer;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_
