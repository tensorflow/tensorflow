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

#include "xla/stream_executor/cuda/cuda_command_buffer.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/cuda/command_buffer_kernels.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"  // IWYU pragma: keep
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {
absl::StatusOr<CUgraph> CreateGraph() {
  VLOG(2) << "Create new CUDA graph";
  CUgraph graph = nullptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuGraphCreate(&graph, /*flags=*/0),
                                    "Failed to create CUDA graph"));
  VLOG(2) << "Created CUDA graph " << graph;
  return graph;
}

CUdeviceptr AsDevicePtr(const DeviceMemoryBase& mem) {
  return absl::bit_cast<CUdeviceptr>(mem.opaque());
}

using GraphNodeHandle = GpuCommandBuffer::GraphNodeHandle;

// Converts a platform independent GraphNodeHandle into a CUDA specific
// CUgraphNode.
CUgraphNode ToCudaGraphHandle(GraphNodeHandle handle) {
  return absl::bit_cast<CUgraphNode>(handle);
}

// Converts a list of platform independent GraphNodeHandles into a list of
// CUDA specific CUgraphNode.
std::vector<CUgraphNode> ToCudaGraphHandles(
    absl::Span<const GraphNodeHandle> opaque_handles) {
  std::vector<CUgraphNode> handles;
  handles.reserve(opaque_handles.size());
  for (const GraphNodeHandle opaque_handle : opaque_handles) {
    handles.push_back(ToCudaGraphHandle(opaque_handle));
  }
  return handles;
}

// Converts a CUDA specific CUgraphNode into a platform independent
// GraphNodeHandle.
GraphNodeHandle FromCudaGraphHandle(CUgraphNode handle) {
  return absl::bit_cast<GraphNodeHandle>(handle);
}
}  // namespace

absl::StatusOr<std::unique_ptr<CudaCommandBuffer>> CudaCommandBuffer::Create(
    Mode mode, GpuExecutor* parent) {
  TF_ASSIGN_OR_RETURN(CUgraph graph, CreateGraph());
  return std::unique_ptr<CudaCommandBuffer>(
      new CudaCommandBuffer(mode, parent, graph,
                            /*is_owned_graph=*/true));
}

absl::StatusOr<CudaCommandBuffer::SetIfConditionKernel*>
CudaCommandBuffer::GetSetIfConditionKernel() {
  if (!set_if_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetIfConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_if_condition_kernel_,
        SetIfConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_if_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetIfElseConditionKernel*>
CudaCommandBuffer::GetSetIfElseConditionKernel() {
  if (!set_if_else_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec,
                        cuda::GetSetIfElseConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_if_else_condition_kernel_,
        SetIfElseConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_if_else_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetCaseConditionKernel*>
CudaCommandBuffer::GetSetCaseConditionKernel() {
  if (!set_case_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetCaseConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_case_condition_kernel_,
        SetCaseConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_case_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetForConditionKernel*>
CudaCommandBuffer::GetSetForConditionKernel() {
  if (!set_for_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetSetForConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_for_condition_kernel_,
        SetForConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_for_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::SetWhileConditionKernel*>
CudaCommandBuffer::GetSetWhileConditionKernel() {
  if (!set_while_condition_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec,
                        cuda::GetSetWhileConditionKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(
        set_while_condition_kernel_,
        SetWhileConditionKernel::FactoryType::Create(parent_, spec));
  }
  return &set_while_condition_kernel_;
}

absl::StatusOr<CudaCommandBuffer::NoOpKernel*>
CudaCommandBuffer::GetNoOpKernel() {
  if (!noop_kernel_) {
    TF_ASSIGN_OR_RETURN(auto spec, cuda::GetNoOpKernelLoaderSpec());
    TF_ASSIGN_OR_RETURN(noop_kernel_,
                        NoOpKernel::FactoryType::Create(parent_, spec));
  }
  return &noop_kernel_;
}

std::unique_ptr<GpuCommandBuffer> CudaCommandBuffer::CreateNestedCommandBuffer(
    CUgraph graph) {
  return std::unique_ptr<CudaCommandBuffer>(
      new CudaCommandBuffer(Mode::kNested, parent_, graph,
                            /*is_owned_graph=*/false));
}

absl::StatusOr<GraphNodeHandle> CudaCommandBuffer::CreateMemsetNode(
    const Dependencies& dependencies, DeviceMemoryBase destination,
    BitPattern bit_pattern, size_t num_elements) {
  CudaContext* cuda_context =
      tensorflow::down_cast<CudaContext*>(parent_->gpu_context());
  VLOG(2) << "Add memset node to a graph " << graph_
          << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements
          << "; context: " << cuda_context->context()
          << "; deps: " << dependencies.size();

  CUDA_MEMSET_NODE_PARAMS params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  std::vector<CUgraphNode> deps = ToCudaGraphHandles(dependencies);

  CUgraphNode node_handle = nullptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuGraphAddMemsetNode(&node_handle, graph_, deps.data(), deps.size(),
                           &params, cuda_context->context()),
      "Failed to add memset node to a CUDA graph"));

  return FromCudaGraphHandle(node_handle);
}

absl::Status CudaCommandBuffer::UpdateMemsetNode(GraphNodeHandle node_handle,
                                                 DeviceMemoryBase destination,
                                                 BitPattern bit_pattern,
                                                 size_t num_elements) {
  CudaContext* cuda_context =
      tensorflow::down_cast<CudaContext*>(parent_->gpu_context());
  VLOG(2) << "Set memset node params " << node_handle << " in graph executable "
          << exec_ << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements
          << "; context: " << cuda_context->context();

  CUDA_MEMSET_NODE_PARAMS params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  return cuda::ToStatus(
      cuGraphExecMemsetNodeSetParams(exec_, ToCudaGraphHandle(node_handle),
                                     &params, cuda_context->context()),
      "Failed to set memset node params");
}

}  // namespace stream_executor::gpu
