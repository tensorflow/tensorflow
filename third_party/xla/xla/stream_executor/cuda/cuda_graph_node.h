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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_NODE_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_NODE_H_

#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_graph_node.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::gpu {

class CudaCommandBuffer;

// The CUDA specific implementation of the GpuGraphNode. It represents a node
// in a CUDA graph with methods to update its properties. It's tied to
// CudaCommandBuffer and can't be used on its own.
class CudaGraphNode : public GpuGraphNode {
 public:
  explicit CudaGraphNode(CUgraphNode node_handle,
                         CudaCommandBuffer* command_buffer)
      : GpuGraphNode(node_handle), command_buffer_(command_buffer) {}

  absl::Status UpdateMemsetNode(DeviceMemoryBase destination,
                                BitPattern bit_pattern,
                                size_t num_elements) override;

  absl::Status UpdateMemcpyD2DNode(DeviceMemoryBase destination,
                                   DeviceMemoryBase source,
                                   uint64_t size) override;

  absl::Status UpdateChildNode(const CommandBuffer& nested) override;

  absl::Status UpdateKernelNode(const ThreadDim& threads,
                                const BlockDim& blocks, const Kernel& kernel,
                                const KernelArgsPackedArrayBase& args) override;

  absl::Status SetExecutionEnabled(CommandBuffer& root_command_buffer,
                                   bool enabled) override;

 private:
  CudaCommandBuffer* command_buffer_;
};
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_GRAPH_NODE_H_
