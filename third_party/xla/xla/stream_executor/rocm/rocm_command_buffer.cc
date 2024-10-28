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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
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

hipDeviceptr_t AsDevicePtr(const DeviceMemoryBase& mem) {
  return absl::bit_cast<hipDeviceptr_t>(mem.opaque());
}

using GraphNodeHandle = GpuCommandBuffer::GraphNodeHandle;

// Converts a platform independent GraphNodeHandle into a HIP specific
// hipGraphNode_t.
hipGraphNode_t ToHipGraphHandle(GpuCommandBuffer::GraphNodeHandle handle) {
  return absl::bit_cast<hipGraphNode_t>(handle);
}

// Converts a list of platform independent GraphNodeHandles into a list of
// HIP specific hipGraphNode_t.
std::vector<hipGraphNode_t> ToHipGraphHandles(
    absl::Span<const GraphNodeHandle> opaque_handles) {
  std::vector<hipGraphNode_t> handles;
  handles.reserve(opaque_handles.size());
  for (const GraphNodeHandle opaque_handle : opaque_handles) {
    handles.push_back(ToHipGraphHandle(opaque_handle));
  }
  return handles;
}

// Converts a HIP specific hipGraphNode_t into a platform independent
// GraphNodeHandle. This function will be removed once all Node factory
// functions have been migrated into the subclasses.
GraphNodeHandle FromHipGraphHandle(hipGraphNode_t handle) {
  return absl::bit_cast<GpuCommandBuffer::GraphNodeHandle>(handle);
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

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateMemsetNode(
    const Dependencies& dependencies, DeviceMemoryBase destination,
    BitPattern bit_pattern, size_t num_elements) {
  VLOG(2) << "Add memset node to a graph " << graph_
          << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements
          << "; context: " << parent_->gpu_context()
          << "; deps: " << dependencies.size();

  hipMemsetParams params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  std::vector<hipGraphNode_t> deps = ToHipGraphHandles(dependencies);

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphAddMemsetNode(&node_handle, graph_, deps.data(),
                                           deps.size(), &params),
               "Failed to add memset node to a HIP graph"));
  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::UpdateMemsetNode(GraphNodeHandle node_handle,
                                                 DeviceMemoryBase destination,
                                                 BitPattern bit_pattern,
                                                 size_t num_elements) {
  VLOG(2) << "Set memset node params " << node_handle << " in graph executable "
          << exec_ << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements;

  hipMemsetParams params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  return ToStatus(wrap::hipGraphExecMemsetNodeSetParams(
                      exec_, ToHipGraphHandle(node_handle), &params),
                  "Failed to set memset node params");
}

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateMemcpyD2DNode(
    const Dependencies& dependencies, DeviceMemoryBase destination,
    DeviceMemoryBase source, uint64_t size) {
  VLOG(2) << "Add memcpy d2d node to a graph " << graph_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size << "; context: " << parent_->gpu_context()
          << "; deps: " << dependencies.size();

  std::vector<hipGraphNode_t> deps = ToHipGraphHandles(dependencies);

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipGraphAddMemcpyNode1D(&node_handle, graph_, deps.data(),
                                    deps.size(), AsDevicePtr(destination),
                                    AsDevicePtr(source), size,
                                    hipMemcpyDeviceToDevice),
      "Failed to add memcpy d2d node to a HIP graph"));
  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::UpdateMemcpyD2DNode(
    GraphNodeHandle node_handle, DeviceMemoryBase destination,
    DeviceMemoryBase source, uint64_t size) {
  VLOG(2) << "Set memcpy d2d node params " << node_handle
          << " in graph executable " << exec_
          << "; dst: " << destination.opaque() << "; src: " << source.opaque()
          << "; size: " << size;

  return ToStatus(
      wrap::hipGraphExecMemcpyNodeSetParams1D(
          exec_, ToHipGraphHandle(node_handle), AsDevicePtr(destination),
          AsDevicePtr(source), size, hipMemcpyDeviceToDevice),
      "Failed to set memcpy d2d node params");
}

}  // namespace stream_executor::gpu
