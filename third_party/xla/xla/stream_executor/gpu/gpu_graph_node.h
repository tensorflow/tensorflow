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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_GRAPH_NODE_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_GRAPH_NODE_H_

#include <cstddef>
#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"

namespace stream_executor::gpu {

// A handle to a Gpu graph node and a metadata describing its properties. Each
// command (launch, memcpy, etc.) creates one or more graph nodes. It is an
// implementation detail of the GpuCommandBuffer and should not be used outside
// of GpuCommandBuffer and its subclasses.
class GpuGraphNode {
 public:
  GpuGraphNode() = default;
  explicit GpuGraphNode(GpuGraphNodeHandle handle) : handle(handle) {}

  // A handle to the gpu graph node corresponding to a command. The handle
  // will be migrated into the subclasses once all GpuDriver calls are moved.
  // NOLINTNEXTLINE(*-class-member-naming)
  GpuGraphNodeHandle handle = nullptr;

  // Updates the memset node with the given parameters. Will return an error
  // if the given node has not been created as a memset node. This function
  // will become pure virtual once all GpuDriver calls are moved into
  // subclasses.
  virtual absl::Status UpdateMemsetNode(DeviceMemoryBase destination,
                                        BitPattern bit_pattern,
                                        size_t num_elements) {
    return absl::UnimplementedError("Not a memset node");
  };

  // Updates the memcpy node with the given parameters. Will return an error
  // if the given node has not been created as a memcpy node. This function
  // will become pure virtual once all GpuDriver calls were moved into
  // subclasses.
  virtual absl::Status UpdateMemcpyD2DNode(DeviceMemoryBase destination,
                                           DeviceMemoryBase source,
                                           uint64_t size) {
    return absl::UnimplementedError("Not a memcpy node");
  }

  // Associate another command buffer with this child node. Will return an
  // error if the given node has not been created as a child node. This
  // function will become pure virtual once all GpuDriver calls were moved
  // into subclasses.
  virtual absl::Status UpdateChildNode(const CommandBuffer& nested) {
    return absl::UnimplementedError("Not a child node");
  }

  // Updates the kernel launch node with the given parameters. Will return an
  // error if the given node has not been created as a kernel launch node.
  // This function will become pure virtual once all GpuDriver calls were
  // moved into subclasses.
  virtual absl::Status UpdateKernelNode(const ThreadDim& threads,
                                        const BlockDim& blocks,
                                        const Kernel& kernel,
                                        const KernelArgsPackedArrayBase& args) {
    return absl::UnimplementedError("Not a kernel launch node");
  };

  // Enables or disablesthe execution of this node in the graph.
  // This function will become pure virtual once all GpuDriver calls were
  // moved into subclasses.
  virtual absl::Status SetExecutionEnabled(CommandBuffer& root_command_buffer,
                                           bool enabled) {
    return absl::UnimplementedError("Not supported");
  }

  virtual ~GpuGraphNode() = default;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_GRAPH_NODE_H_
