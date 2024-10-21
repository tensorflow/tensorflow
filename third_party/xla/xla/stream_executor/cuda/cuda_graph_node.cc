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

#include "xla/stream_executor/cuda/cuda_graph_node.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_command_buffer.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/cuda/cuda_kernel.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"

namespace stream_executor::gpu {
namespace {
CUdeviceptr AsDevicePtr(const DeviceMemoryBase& mem) {
  return absl::bit_cast<CUdeviceptr>(mem.opaque());
}
}  // namespace

absl::Status CudaGraphNode::UpdateMemsetNode(DeviceMemoryBase destination,
                                             BitPattern bit_pattern,
                                             size_t num_elements) {
  CudaContext* cuda_context = tensorflow::down_cast<CudaContext*>(
      command_buffer_->parent_->gpu_context());
  VLOG(2) << "Set memset node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "; dst: " << destination.opaque()
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
      cuGraphExecMemsetNodeSetParams(command_buffer_->exec_, handle, &params,
                                     cuda_context->context()),
      "Failed to set memset node params");
}

absl::Status CudaGraphNode::UpdateMemcpyD2DNode(DeviceMemoryBase destination,
                                                DeviceMemoryBase source,
                                                uint64_t size) {
  CudaContext* cuda_context = tensorflow::down_cast<CudaContext*>(
      command_buffer_->parent_->gpu_context());
  VLOG(2) << "Set memcpy d2d node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "; dst: " << destination.opaque()
          << "; src: " << source.opaque() << "; size: " << size
          << "; context: " << cuda_context->context();

  CUDA_MEMCPY3D params{};
  params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  params.srcDevice = AsDevicePtr(source);
  params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  params.dstDevice = AsDevicePtr(destination);
  params.WidthInBytes = size;
  params.Height = 1;
  params.Depth = 1;

  return cuda::ToStatus(
      cuGraphExecMemcpyNodeSetParams(command_buffer_->exec_, handle, &params,
                                     cuda_context->context()),
      "Failed to set memcpy d2d node params");
}

absl::Status CudaGraphNode::UpdateChildNode(const CommandBuffer& nested) {
  CUgraph child_graph =
      tensorflow::down_cast<const CudaCommandBuffer&>(nested).graph_;
  VLOG(2) << "Set child node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "to params contained in " << child_graph;

  return cuda::ToStatus(cuGraphExecChildGraphNodeSetParams(
                            command_buffer_->exec_, handle, child_graph),
                        "Failed to set CUDA graph child node params");
}

absl::Status CudaGraphNode::UpdateKernelNode(
    const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
    const KernelArgsPackedArrayBase& args) {
  const uint64_t shared_mem_bytes = args.number_of_shared_bytes();

  VLOG(2) << "Set kernel node params " << handle << " in graph executable "
          << command_buffer_->exec_ << "; kernel: " << kernel.name()
          << "; gdx: " << blocks.x << " gdy: " << blocks.y
          << " gdz: " << blocks.z << " bdx: " << threads.x
          << " bdy: " << threads.y << " bdz: " << threads.z
          << "; shmem: " << shared_mem_bytes;

  CUDA_KERNEL_NODE_PARAMS params{};
  CUfunction function = static_cast<const CudaKernel&>(kernel).gpu_function();
  params.func = function;
  params.gridDimX = blocks.x;
  params.gridDimY = blocks.y;
  params.gridDimZ = blocks.z;
  params.blockDimX = threads.x;
  params.blockDimY = threads.y;
  params.blockDimZ = threads.z;
  params.sharedMemBytes = shared_mem_bytes;
  params.kernelParams = const_cast<void**>(args.argument_addresses().data());
  params.extra = nullptr;

  // TODO(ezhulenev): Why do we do it on every call to launch kernel? This
  // should be moved one level up to se::Kernel level, and done just once (or
  // updated once we get a new larger shared memory request).
  if (shared_mem_bytes != 0) {
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuFuncSetAttribute(function,
                           CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                           shared_mem_bytes),
        "Failed to set shared memory size"));
  }

  return cuda::ToStatus(
      cuGraphExecKernelNodeSetParams(command_buffer_->exec_, handle, &params),
      "Failed to set CUDA graph kernel node params");
}

absl::Status CudaGraphNode::SetExecutionEnabled(
    CommandBuffer& root_command_buffer, bool enabled) {
  // Node is enabled if value != 0, otherwise the node is disabled.
  unsigned value = enabled ? 1 : 0;
  CUgraphExec exec = static_cast<CudaCommandBuffer&>(root_command_buffer).exec_;
  VLOG(2) << "Set CUDA executable graph " << exec << " node " << handle
          << " enabled flag to " << value;
  return cuda::ToStatus(cuGraphNodeSetEnabled(exec, handle, value),
                        "Failed to set CUDA graph node enabled flag");
}
}  // namespace stream_executor::gpu
