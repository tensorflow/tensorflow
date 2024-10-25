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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_executor.h"

namespace stream_executor::gpu {

// Implements GpuCommandBuffer for AMD GPUs.
class RocmCommandBuffer : public GpuCommandBuffer {
 public:
  // Creates a new ROCm command buffer and the underlying HIP graph.
  static absl::StatusOr<std::unique_ptr<RocmCommandBuffer>> Create(
      Mode mode, GpuExecutor* parent);

 private:
  RocmCommandBuffer(Mode mode, GpuExecutor* parent, hipGraph_t graph,
                    bool is_owned_graph)
      : GpuCommandBuffer(mode, parent, graph, is_owned_graph),
        parent_(parent) {}

  absl::StatusOr<SetIfConditionKernel*> GetSetIfConditionKernel() override;
  absl::StatusOr<SetIfElseConditionKernel*> GetSetIfElseConditionKernel()
      override;
  absl::StatusOr<SetCaseConditionKernel*> GetSetCaseConditionKernel() override;
  absl::StatusOr<SetForConditionKernel*> GetSetForConditionKernel() override;
  absl::StatusOr<SetWhileConditionKernel*> GetSetWhileConditionKernel()
      override;
  absl::StatusOr<NoOpKernel*> GetNoOpKernel() override;

  std::unique_ptr<GpuCommandBuffer> CreateNestedCommandBuffer(
      hipGraph_t graph) override;

  GpuExecutor* parent_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_
