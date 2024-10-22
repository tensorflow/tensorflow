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

// This class implements GpuCommandBuffer for Nvidia GPUs.
class CudaCommandBuffer : public GpuCommandBuffer {
 public:
  // Creates a new CUDA command buffer and the underlying CUDA graph.
  static absl::StatusOr<std::unique_ptr<CudaCommandBuffer>> Create(
      Mode mode, GpuExecutor* parent);

 private:
  CudaCommandBuffer(Mode mode, GpuExecutor* parent, CUgraph graph,
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
      CUgraph graph) override;

  // Lazy loaded auxiliary kernels required for building CUDA graphs (no-op
  // barriers, updating conditional handles, etc.).
  SetIfConditionKernel set_if_condition_kernel_;
  SetIfElseConditionKernel set_if_else_condition_kernel_;
  SetCaseConditionKernel set_case_condition_kernel_;
  SetForConditionKernel set_for_condition_kernel_;
  SetWhileConditionKernel set_while_condition_kernel_;
  NoOpKernel noop_kernel_;

  GpuExecutor* parent_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_
