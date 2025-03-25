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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/scoped_update_mode.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// Implements CommandBuffer for AMD GPUs.
class RocmCommandBuffer : public CommandBuffer {
 public:
  // Creates a new ROCm command buffer and the underlying HIP graph.
  static absl::StatusOr<std::unique_ptr<RocmCommandBuffer>> Create(
      Mode mode, StreamExecutor* parent);

  //===--------------------------------------------------------------------===//
  // Command buffer API
  //===--------------------------------------------------------------------===//

  absl::StatusOr<NodeHandle> CreateEmptyNode(
      Dependencies dependencies) override;

  // Adds a kernel launch command that depends on the commands in
  // deps.
  absl::StatusOr<NodeHandle> CreateLaunchNode(Dependencies dependencies,
                                              const ThreadDim& threads,
                                              const BlockDim& blocks,
                                              const Kernel& kernel,
                                              const KernelArgs& args) override;

  absl::Status UpdateLaunchNode(NodeHandle node, const ThreadDim& threads,
                                const BlockDim& blocks, const Kernel& kernel,
                                const KernelArgs& args) override;

  absl::StatusOr<NodeHandle> CreateChildNode(
      Dependencies dependencies, const CommandBuffer& child) override;

  absl::Status UpdateChildNode(NodeHandle node,
                               const CommandBuffer& child) override;

  // Adds a device-to-device memory copy that depends on the commands in
  // deps.
  absl::StatusOr<NodeHandle> CreateMemcpyD2DNode(Dependencies dependencies,
                                                 DeviceMemoryBase dst,
                                                 DeviceMemoryBase src,
                                                 uint64_t size) override;

  absl::Status UpdateMemcpyD2DNode(NodeHandle node, DeviceMemoryBase dst,
                                   DeviceMemoryBase src,
                                   uint64_t size) override;

  // Adds a memset command that depends on the commands in deps.
  absl::StatusOr<NodeHandle> CreateMemsetNode(Dependencies dependencies,
                                              DeviceMemoryBase dst,
                                              BitPattern bit_pattern,
                                              size_t num_elements) override;

  absl::Status UpdateMemsetNode(NodeHandle node, DeviceMemoryBase dst,
                                BitPattern bit_pattern,
                                size_t num_elements) override;

  //--------------------------------------------------------------------------//
  // Command buffer condtitional commands API
  //--------------------------------------------------------------------------//

  absl::StatusOr<ConditionalHandle> CreateConditionalHandle() override;

  // Adds a new conditional node to the graph and creates a
  // corresponding nested command buffer.
  absl::StatusOr<ConditionalNodeResult> CreateConditionalNode(
      Dependencies dependencies, ConditionalHandle conditional,
      ConditionType type) override {
    return absl::UnimplementedError("CreateConditionalNode");
  }

  absl::StatusOr<NodeHandle> CreateIfElseConditionNode(
      Dependencies dependencies, ConditionalHandle then_condition,
      ConditionalHandle else_condition, DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("CreateIfElseConditionNode");
  }

  absl::Status UpdateIfElseConditionNode(
      NodeHandle node, ConditionalHandle then_condition,
      ConditionalHandle else_condition, DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("UpdateIfElseConditionNode");
  }

  absl::StatusOr<NodeHandle> CreateIfConditionNode(
      Dependencies dependencies, ConditionalHandle then_condition,
      DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("CreateIfConditionNode");
  }

  absl::Status UpdateIfConditionNode(NodeHandle node,
                                     ConditionalHandle then_condition,
                                     DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("UpdateIfConditionNode");
  }

  absl::StatusOr<NodeHandle> CreateForConditionNode(
      Dependencies dependencies, ConditionalHandle condition,
      DeviceMemory<int32_t> loop_counter, int32_t iterations) override {
    return absl::UnimplementedError("CreateForConditionNode");
  }

  absl::Status UpdateForConditionNode(NodeHandle node,
                                      ConditionalHandle condition,
                                      DeviceMemory<int32_t> loop_counter,
                                      int32_t iterations) override {
    return absl::UnimplementedError("UpdateForConditionNode");
  }

  absl::StatusOr<NodeHandle> CreateWhileConditionNode(
      Dependencies dependencies, ConditionalHandle condition,
      DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("CreateWhileConditionNode");
  }

  absl::Status UpdateWhileConditionNode(NodeHandle node,
                                        ConditionalHandle condition,
                                        DeviceMemory<bool> predicate) override {
    return absl::UnimplementedError("UpdateWhileConditionNode");
  }

  absl::StatusOr<NodeHandle> CreateCaseConditionNode(
      Dependencies dependencies, std::array<ConditionalHandle, 8> conditions,
      DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      int32_t num_branches, bool enable_conditional_default) override {
    return absl::UnimplementedError("CreateCaseConditionNode");
  }

  absl::Status UpdateCaseConditionNode(
      NodeHandle node, std::array<ConditionalHandle, 8> conditions,
      DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      int32_t num_branches, bool enable_conditional_default) override {
    return absl::UnimplementedError("UpdateCaseConditionNode");
  }

  absl::Status Submit(Stream* stream) override;

  absl::Status Finalize() override;

  ~RocmCommandBuffer() override;

 private:
  RocmCommandBuffer(Mode mode, StreamExecutor* parent, hipGraph_t graph,
                    bool is_owned_graph)
      : CommandBuffer(mode), graph_(graph), is_owned_graph_(is_owned_graph) {}

  absl::Status PrepareFinalization();

  absl::StatusOr<NodeHandle> CreateKernelNode(
      Dependencies dependencies, const ThreadDim& threads,
      const BlockDim& blocks, const Kernel& kernel,
      const KernelArgsPackedArrayBase& args);

  absl::Status UpdateKernelNode(NodeHandle node_handle,
                                const ThreadDim& threads,
                                const BlockDim& blocks, const Kernel& kernel,
                                const KernelArgsPackedArrayBase& args);

  absl::Status Trace(Stream* stream,
                     absl::AnyInvocable<absl::Status()> function) override;

  absl::Status SetNodeExecutionEnabled(NodeHandle node_handle, bool enabled);

  absl::Status LaunchGraph(Stream* stream);

  absl::StatusOr<size_t> GetNodeCount() const;

  absl::Status WriteGraphToDotFile(absl::string_view path);

  absl::Status InstantiateGraph();

  absl::Status CheckCanBeUpdated();

  StreamExecutor* stream_executor_;

  static_assert(std::is_pointer_v<hipGraph_t>, "hipGraph_t must be a pointer");
  static_assert(std::is_pointer_v<hipGraphExec_t>,
                "hipGraphExec_t must be a pointer");

  hipGraph_t graph_ = nullptr;  // owned if `is_owned_graph_`
  bool is_owned_graph_ = true;  // ownership of `graph_`

  hipGraphExec_t exec_ = nullptr;    // owned if `is_owned_graph_exec_`
  bool is_owned_graph_exec_ = true;  // ownership of `is_owned_graph_exec_`
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_COMMAND_BUFFER_H_
