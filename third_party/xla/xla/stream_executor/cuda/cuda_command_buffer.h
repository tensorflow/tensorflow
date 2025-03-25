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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/cuda/cuda_context.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/scoped_update_mode.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

#if CUDA_VERSION < 12030
typedef cuuint64_t CUgraphConditionalHandle;
#endif

namespace stream_executor::gpu {

//===----------------------------------------------------------------------===//
// CudaCommandBuffer resource usage tracking
//===----------------------------------------------------------------------===//

static std::atomic<int64_t> allocated_execs(0);
static std::atomic<int64_t> alive_execs(0);

// This class implements CommandBuffer for Nvidia GPUs.
class CudaCommandBuffer final : public CommandBuffer {
 public:
  // Creates a new CUDA command buffer and the underlying CUDA graph.
  static absl::StatusOr<std::unique_ptr<CudaCommandBuffer>> Create(
      Mode mode, StreamExecutor* stream_executor, CudaContext* cuda_context);

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
      ConditionType type) override;

  absl::StatusOr<NodeHandle> CreateIfElseConditionNode(
      Dependencies dependencies, ConditionalHandle then_condition,
      ConditionalHandle else_condition, DeviceMemory<bool> predicate) override;

  absl::Status UpdateIfElseConditionNode(NodeHandle node,
                                         ConditionalHandle then_condition,
                                         ConditionalHandle else_condition,
                                         DeviceMemory<bool> predicate) override;

  absl::StatusOr<NodeHandle> CreateIfConditionNode(
      Dependencies dependencies, ConditionalHandle then_condition,
      DeviceMemory<bool> predicate) override;

  absl::Status UpdateIfConditionNode(NodeHandle node,
                                     ConditionalHandle then_condition,
                                     DeviceMemory<bool> predicate) override;

  absl::StatusOr<NodeHandle> CreateForConditionNode(
      Dependencies dependencies, ConditionalHandle condition,
      DeviceMemory<int32_t> loop_counter, int32_t iterations) override;

  absl::Status UpdateForConditionNode(NodeHandle node,
                                      ConditionalHandle condition,
                                      DeviceMemory<int32_t> loop_counter,
                                      int32_t iterations) override;

  absl::StatusOr<NodeHandle> CreateWhileConditionNode(
      Dependencies dependencies, ConditionalHandle condition,
      DeviceMemory<bool> predicate) override;

  absl::Status UpdateWhileConditionNode(NodeHandle node,
                                        ConditionalHandle condition,
                                        DeviceMemory<bool> predicate) override;

  absl::StatusOr<NodeHandle> CreateCaseConditionNode(
      Dependencies dependencies, std::array<ConditionalHandle, 8> conditions,
      DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      int32_t num_branches, bool enable_conditional_default) override;

  absl::Status UpdateCaseConditionNode(
      NodeHandle node, std::array<ConditionalHandle, 8> conditions,
      DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      int32_t num_branches, bool enable_conditional_default) override;

  absl::Status Submit(Stream* stream) override;
  absl::Status Finalize() override;

  ~CudaCommandBuffer() override;

 private:
  CudaCommandBuffer(Mode mode, StreamExecutor* stream_executor,
                    CudaContext* cuda_context, CUgraph graph,
                    bool is_owned_graph)
      : CommandBuffer(mode),
        stream_executor_(stream_executor),
        cuda_context_(cuda_context),
        graph_(graph),
        is_owned_graph_(is_owned_graph) {
    VLOG(5) << "Created command buffer for graph " << graph_
            << "; mode=" << ModeToString(mode)
            << "; is_owned_graph=" << is_owned_graph_;
  }

  static int64_t NotifyExecCreated();
  static int64_t NotifyExecDestroyed();
  static int64_t AliveExecs();

  using NoOpKernel = TypedKernel<>;
  absl::StatusOr<NoOpKernel*> GetNoOpKernel();

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

  // absl::Status SetNodeExecutionEnabled(NodeHandle node_handle,
  //                                      bool enabled);

  absl::Status LaunchGraph(Stream* stream);

  absl::StatusOr<size_t> GetNodeCount() const;

  absl::Status PrepareFinalization();

  absl::Status WriteGraphToDotFile(absl::string_view path);

  absl::Status InstantiateGraph();

  absl::Status CheckCanBeUpdated();

  absl::StatusOr<std::vector<NodeHandle>> GetNodeDependencies(NodeHandle node);

  // A signature of a device kernels updating conditional handle(s).
  using SetIfConditionKernel =
      TypedKernel<CUgraphConditionalHandle, DeviceMemory<bool>>;

  using SetIfElseConditionKernel =
      TypedKernel<CUgraphConditionalHandle, CUgraphConditionalHandle,
                  DeviceMemory<bool>>;

  using SetCaseConditionKernel =
      TypedKernel<CUgraphConditionalHandle, CUgraphConditionalHandle,
                  CUgraphConditionalHandle, CUgraphConditionalHandle,
                  CUgraphConditionalHandle, CUgraphConditionalHandle,
                  CUgraphConditionalHandle, CUgraphConditionalHandle,
                  DeviceMemory<uint8_t>, bool, int32_t, int32_t, bool>;

  using SetForConditionKernel =
      TypedKernel<CUgraphConditionalHandle, DeviceMemory<int32_t>, int32_t>;

  using SetWhileConditionKernel =
      TypedKernel<CUgraphConditionalHandle, DeviceMemory<bool>>;

  // Lazy loaded auxiliary kernels required for building CUDA graphs (no-op
  // barriers, updating conditional handles, etc.).
  SetIfConditionKernel set_if_condition_kernel_;
  SetIfElseConditionKernel set_if_else_condition_kernel_;
  SetCaseConditionKernel set_case_condition_kernel_;
  SetForConditionKernel set_for_condition_kernel_;
  SetWhileConditionKernel set_while_condition_kernel_;

  NoOpKernel noop_kernel_;

  StreamExecutor* stream_executor_;

  CudaContext* cuda_context_;

  static_assert(std::is_pointer_v<CUgraph>, "CUgraph must be a pointer");
  static_assert(std::is_pointer_v<CUgraphExec>,
                "CUgraphExec must be a pointer");

  CUgraph graph_ = nullptr;     // owned if `is_owned_graph_`
  bool is_owned_graph_ = true;  // ownership of `graph_`
  bool finalized_ = false;

  CUgraphExec exec_ = nullptr;  // owned if `is_owned_graph_exec_`
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_COMMAND_BUFFER_H_
