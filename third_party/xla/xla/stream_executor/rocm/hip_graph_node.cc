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

#include "xla/stream_executor/rocm/hip_graph_node.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "rocm/include/hip/driver_types.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/rocm/rocm_command_buffer.h"
#include "xla/stream_executor/rocm/rocm_driver_wrapper.h"
#include "xla/stream_executor/rocm/rocm_kernel.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"

namespace stream_executor::gpu {
namespace {
hipDeviceptr_t AsDevicePtr(const DeviceMemoryBase& mem) {
  return absl::bit_cast<hipDeviceptr_t>(mem.opaque());
}
}  // namespace

absl::Status HipGraphNode::UpdateMemsetNode(DeviceMemoryBase destination,
                                            BitPattern bit_pattern,
                                            size_t num_elements) {
  VLOG(2) << "Set memset node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "; dst: " << destination.opaque()
          << "; bit_pattern: " << bit_pattern.ToString()
          << "; num_elements: " << num_elements;

  hipMemsetParams params{};
  params.dst = AsDevicePtr(destination);
  params.elementSize = bit_pattern.GetElementSize();
  params.height = 1;
  params.pitch = 0;  // unused if height is 1
  params.value = bit_pattern.GetPatternBroadcastedToUint32();
  params.width = num_elements;

  return ToStatus(wrap::hipGraphExecMemsetNodeSetParams(command_buffer_->exec_,
                                                        handle, &params),
                  "Failed to set memset node params");
}

absl::Status HipGraphNode::UpdateMemcpyD2DNode(DeviceMemoryBase destination,
                                               DeviceMemoryBase source,
                                               uint64_t size) {
  VLOG(2) << "Set memcpy d2d node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "; dst: " << destination.opaque()
          << "; src: " << source.opaque() << "; size: " << size;

  return ToStatus(wrap::hipGraphExecMemcpyNodeSetParams1D(
                      command_buffer_->exec_, handle, AsDevicePtr(destination),
                      AsDevicePtr(source), size, hipMemcpyDeviceToDevice),
                  "Failed to set memcpy d2d node params");
}

absl::Status HipGraphNode::UpdateChildNode(const CommandBuffer& nested) {
  hipGraph_t child_graph =
      tensorflow::down_cast<const RocmCommandBuffer&>(nested).graph_;

  VLOG(2) << "Set child node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "to params contained in " << child_graph;

  return ToStatus(wrap::hipGraphExecChildGraphNodeSetParams(
                      command_buffer_->exec_, handle, child_graph),
                  "Failed to set HIP graph child node params");
}

absl::Status HipGraphNode::UpdateKernelNode(
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Set kernel node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "; kernel: " << kernel.name()
          << "; gdx: " << blocks.x << " gdy: " << blocks.y
          << " gdz: " << blocks.z << " bdx: " << threads.x
          << " bdy: " << threads.y << " bdz: " << threads.z
          << "; shmem: " << shared_mem_bytes;

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

  return ToStatus(wrap::hipGraphExecKernelNodeSetParams(command_buffer_->exec_,
                                                        handle, &params),
                  "Failed to set HIP graph kernel node params");
}

absl::Status HipGraphNode::SetExecutionEnabled(
    CommandBuffer& root_command_buffer, bool enabled) {
  // Node is enabled if value != 0, otherwise the node is disabled.
  unsigned value = enabled ? 1 : 0;
  hipGraphExec_t exec =
      static_cast<RocmCommandBuffer&>(root_command_buffer).exec_;
  VLOG(2) << "Set HIP executable graph " << exec << " node " << handle
          << " enabled flag to " << value;
  return ToStatus(wrap::hipGraphNodeSetEnabled(exec, handle, value),
                  "Failed to set HIP graph node enabled flag");
}
}  // namespace stream_executor::gpu
