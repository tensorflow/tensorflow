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

#include "xla/stream_executor/rocm/rocm_command_buffer.h"

#include <memory>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<hipGraph_t> CreateGraph() {
  VLOG(2) << "Create new HIP graph";
  hipGraph_t graph;
  TF_RETURN_IF_ERROR(ToStatus(wrap::hipGraphCreate(&graph, /*flags=*/0),
                              "Failed to create HIP graph"));
  VLOG(2) << "Created HIP graph " << graph;
  return graph;
}
}  // namespace

absl::StatusOr<std::unique_ptr<RocmCommandBuffer>> RocmCommandBuffer::Create(
    Mode mode, GpuExecutor* parent) {
  TF_ASSIGN_OR_RETURN(hipGraph_t graph, CreateGraph());
  return std::unique_ptr<RocmCommandBuffer>(
      new RocmCommandBuffer(mode, parent, graph,
                            /*is_owned_graph=*/true));
}

std::unique_ptr<GpuCommandBuffer> RocmCommandBuffer::CreateNestedCommandBuffer(
    hipGraph_t graph) {
  return std::unique_ptr<RocmCommandBuffer>(
      new RocmCommandBuffer(Mode::kNested, parent_, graph,
                            /*is_owned_graph=*/false));
}

absl::StatusOr<GpuCommandBuffer::SetIfConditionKernel*>
RocmCommandBuffer::GetSetIfConditionKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GpuCommandBuffer::SetIfElseConditionKernel*>
RocmCommandBuffer::GetSetIfElseConditionKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GpuCommandBuffer::SetCaseConditionKernel*>
RocmCommandBuffer::GetSetCaseConditionKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GpuCommandBuffer::SetForConditionKernel*>
RocmCommandBuffer::GetSetForConditionKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GpuCommandBuffer::SetWhileConditionKernel*>
RocmCommandBuffer::GetSetWhileConditionKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}

absl::StatusOr<GpuCommandBuffer::NoOpKernel*>
RocmCommandBuffer::GetNoOpKernel() {
  return absl::UnimplementedError("Conditionals are not supported on ROCM.");
}
}  // namespace stream_executor::gpu
