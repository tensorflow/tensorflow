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
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_command_buffer.h"
#include "xla/stream_executor/gpu/gpu_driver.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_kernel.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/env.h"
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

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateChildNode(
    const Dependencies& dependencies, const CommandBuffer& nested) {
  hipGraph_t child_graph =
      tensorflow::down_cast<const RocmCommandBuffer&>(nested).graph_;
  VLOG(2) << "Create a new node by cloning the child graph " << child_graph
          << " and add it to " << graph_ << "; deps: " << dependencies.size();

  std::vector<hipGraphNode_t> deps = ToHipGraphHandles(dependencies);

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(ToStatus(
      wrap::hipGraphAddChildGraphNode(&node_handle, graph_, deps.data(),
                                      deps.size(), child_graph),
      "Failed to create a child graph node and add it to a HIP graph"));
  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::UpdateChildNode(GraphNodeHandle node_handle,
                                                const CommandBuffer& nested) {
  hipGraph_t child_graph =
      tensorflow::down_cast<const RocmCommandBuffer&>(nested).graph_;

  VLOG(2) << "Set child node params " << node_handle << " in graph executable "
          << exec_ << "to params contained in " << child_graph;

  return ToStatus(wrap::hipGraphExecChildGraphNodeSetParams(
                      exec_, ToHipGraphHandle(node_handle), child_graph),
                  "Failed to set HIP graph child node params");
}

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateKernelNode(
    const Dependencies& dependencies, const ThreadDim& threads,
    const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Add kernel node to a graph " << graph_
          << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes
          << "; deps: " << dependencies.size();

  hipKernelNodeParams params{};

  hipFunction_t function =
      static_cast<const RocmKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDim.x = blocks.x;
  params.gridDim.z = blocks.y;
  params.gridDim.z = blocks.z;
  params.blockDim.x = threads.x;
  params.blockDim.y = threads.y;
  params.blockDim.z = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipFuncSetAttribute(function,
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  std::vector<hipGraphNode_t> deps = ToHipGraphHandles(dependencies);

  hipGraphNode_t node_handle = nullptr;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphAddKernelNode(&node_handle, graph_, deps.data(),
                                           deps.size(), &params),
               "Failed to add kernel node to a HIP graph"));

  return FromHipGraphHandle(node_handle);
}

absl::Status RocmCommandBuffer::UpdateKernelNode(
    GraphNodeHandle node_handle, const ThreadDim& threads,
    const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Set kernel node params " << node_handle << " in graph executable "
          << exec_ << "; kernel: " << kernel.name() << "; gdx: " << blocks.x
          << " gdy: " << blocks.y << " gdz: " << blocks.z
          << " bdx: " << threads.x << " bdy: " << threads.y
          << " bdz: " << threads.z << "; shmem: " << shared_mem_bytes;

  hipKernelNodeParams params{};

  hipFunction_t function =
      static_cast<const RocmKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDim.x = blocks.x;
  params.gridDim.z = blocks.y;
  params.gridDim.z = blocks.z;
  params.blockDim.x = threads.x;
  params.blockDim.y = threads.y;
  params.blockDim.z = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(ToStatus(
        wrap::hipFuncSetAttribute(function,
                                  hipFuncAttributeMaxDynamicSharedMemorySize,
                                  shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  return ToStatus(wrap::hipGraphExecKernelNodeSetParams(
                      exec_, ToHipGraphHandle(node_handle), &params),
                  "Failed to set HIP graph kernel node params");
}

absl::StatusOr<GraphNodeHandle> RocmCommandBuffer::CreateBarrierNode(
    const Dependencies& dependencies) {
  VLOG(2) << "Add empty node to a graph " << graph_
          << "; deps: " << dependencies.size();

  hipGraphNode_t barrier_handle = nullptr;
  std::vector<hipGraphNode_t> deps = ToHipGraphHandles(dependencies);

  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphAddEmptyNode(&barrier_handle, graph_, deps.data(),
                                          deps.size()),
               "Failed to add empty node to a HIP graph"));

  return FromHipGraphHandle(barrier_handle);
}

absl::Status RocmCommandBuffer::Trace(
    Stream* stream, absl::AnyInvocable<absl::Status()> function) {
  TF_RETURN_IF_ERROR(CheckNotFinalized());
  TF_ASSIGN_OR_RETURN(size_t count, GpuDriver::GraphGetNodeCount(graph_));
  if (count != 0 || !is_owned_graph_)
    return absl::InternalError(
        "Stream can't be traced on non empty command buffer");

  VLOG(5) << "Trace into GPU command buffer graph " << graph_
          << " on a stream: " << stream;

  hipStream_t stream_handle = AsGpuStreamValue(stream);

  // Switch stream into the capture mode.
  uint64_t start_nanos = tsl::Env::Default()->NowNanos();
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipStreamBeginCapture(stream_handle,
                                           hipStreamCaptureModeThreadLocal),
               "Failed to begin stream capture"));
  auto traced = function();

  // Always stop capturing the stream before checking `traced` result.
  VLOG(5) << "End stream " << stream << " capture";
  hipGraph_t captured_graph;
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipStreamEndCapture(stream_handle, &captured_graph),
               "Failed to end stream capture"));
  TF_RETURN_IF_ERROR(
      ToStatus(wrap::hipGraphDestroy(std::exchange(graph_, captured_graph)),
               "Failed to destroy HIP graph"));
  uint64_t end_nanos = tsl::Env::Default()->NowNanos();

  if (!traced.ok())
    return absl::InternalError(
        absl::StrCat("Failed to capture gpu graph: ", traced.message()));

  VLOG(5) << "Traced into the GPU command buffer graph " << graph_ << " (took "
          << (end_nanos - start_nanos) / 1000 << " μs)";

  return absl::OkStatus();
}

}  // namespace stream_executor::gpu
