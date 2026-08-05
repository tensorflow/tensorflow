/* Copyright 2026 The OpenXLA Authors.

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

#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_command_buffer.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/semantic_version.h"

namespace stream_executor::gpu {

using GraphNodeHandle = GpuCommandBuffer::GraphNodeHandle;

namespace {

absl::Status CheckRuntimeVersion(const DeviceDescription& device_description) {
  if (device_description.driver_version() < SemanticVersion{12, 9, 0}) {
    return absl::UnimplementedError(
        "Moved child node require CUDA driver version >= 12.9");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateMovedChildNodeImpl(
    absl::Span<const GraphNodeHandle> dependencies,
    stream_executor::CommandBuffer* nested) {
  RETURN_IF_ERROR(CheckRuntimeVersion(stream_exec_->GetDeviceDescription()));
  auto* child_command_buffer = absl::down_cast<CudaCommandBuffer*>(nested);
  CHECK_EQ(child_command_buffer->parent_, nullptr)
      << "Nested command buffer's parent is not null";

  CUgraph child_graph = child_command_buffer->graph_;
  std::vector<CUgraphNode> deps = ToCudaGraphHandles(dependencies);

  VLOG(2) << "Create a new node by moving the child graph " << child_graph
          << " and add it to " << graph_ << "; deps(" << dependencies.size()
          << "): " << FormatGraphNodeHandles(dependencies);

  CUgraphNodeParams nodeParams{};
  nodeParams.type = CU_GRAPH_NODE_TYPE_GRAPH;
  nodeParams.graph.graph = child_graph;
  nodeParams.graph.ownership = CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE;

  CUgraphNode node_handle;
  RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddNode_v2(&node_handle, graph_, deps.data(),
                        /*dependencyData=*/nullptr, deps.size(), &nodeParams),
      "Failed to create a child graph node and add it to a CUDA graph"));

  child_command_buffer->parent_ = this;
  child_command_buffer->is_owned_graph_ = false;

  return FromCudaGraphHandle(node_handle);
}

}  // namespace stream_executor::gpu
