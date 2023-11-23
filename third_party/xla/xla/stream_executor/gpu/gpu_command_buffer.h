/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_

#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"

namespace stream_executor::gpu {

// GpuCommandBuffer provides platform-specific CommandBufferInterface
// implementation (it's backed by CUDA or HIP graphs on NVIDIA and AMD devices).
class GpuCommandBuffer : public internal::CommandBufferInterface {
 public:
  GpuCommandBuffer(CommandBuffer::Mode mode, GpuExecutor* parent,
                   GpuGraphHandle graph, bool is_owned_graph = true);
  ~GpuCommandBuffer() override;

  tsl::Status Trace(Stream* stream,
                    absl::AnyInvocable<tsl::Status()> function) override;

  tsl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                     const Kernel& kernel, const KernelArgs& args) override;

  tsl::Status AddNestedCommandBuffer(const CommandBuffer& nested) override;

  tsl::Status MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                   const DeviceMemoryBase& src,
                                   uint64_t size) override;

  tsl::Status If(StreamExecutor* executor, DeviceMemory<bool> predicate,
                 CommandBuffer::Builder then_builder) override;

  tsl::Status Finalize() override;
  tsl::Status Update() override;

  GpuGraphExecHandle executable() const { return exec_; }
  GpuGraphHandle graph() const { return graph_; }

  CommandBuffer::Mode mode() const override { return mode_; }
  CommandBuffer::State state() const override { return state_; }

  // A helper template for launching typed kernels.
  template <typename... Params, typename... Args>
  tsl::Status Launch(const TypedKernel<Params...>& kernel,
                     const ThreadDim& threads, const BlockDim& blocks,
                     Args... args);

  // We track the total number of allocated and alive executable graphs in the
  // process to track the command buffers resource usage. Executable graph
  // allocates resources on a GPU devices (rule of thumb is ~8kb per node), so
  // we have to be careful not to keep too many of them alive for too long, or
  // we have a higher risk of OOM errors.
  //
  // TODO(ezhulenev): We need to have a policy for how to evict unused
  // executable graph instances from a device, currently lifetime of an
  // executable graph is tied to a parent command buffer, and we can have
  // thousands of command buffers alive at the same time.
  static int64_t AllocatedExecs();
  static int64_t AliveExecs();

  static GpuCommandBuffer* Cast(CommandBuffer* command_buffer) {
    return static_cast<GpuCommandBuffer*>(command_buffer->implementation());
  }

  static const GpuCommandBuffer* Cast(const CommandBuffer* command_buffer) {
    return static_cast<const GpuCommandBuffer*>(
        command_buffer->implementation());
  }

 private:
  using Dependencies = absl::InlinedVector<GpuGraphNodeHandle, 1>;

  // A signature of a device kernels updating conditional handle.
  using SetConditionKernel =
      TypedKernel<GpuGraphConditionalHandle, DeviceMemory<bool>>;

  // Overwrites the `exec_` handle in a Gpu command buffer by `exec`, and
  // restores to the original handle when destroyed. This allows us updating
  // primary graph executable using nested command buffers (command buffers that
  // do not have their own executable), which is required for updating
  // conditional commands.
  struct ScopedGpuGraphExec {
    ScopedGpuGraphExec(GpuCommandBuffer* cmd_buffer, GpuGraphExecHandle exec);
    ~ScopedGpuGraphExec();

    GpuCommandBuffer* cmd_buffer;
    GpuGraphExecHandle restore;
    bool restore_is_owned;
  };

  // For each conditional node in the Gpu graph we keep a record of conditional
  // command buffers attached to a node, so we can apply updates to them.
  struct ConditionalCommandBuffers {
    void Add(GpuGraphConditionalHandle handle, CommandBuffer command_buffer);

    std::vector<GpuGraphConditionalHandle> handles;
    std::vector<CommandBuffer> command_buffers;
  };

  // TODO(ezhulenev): Currently we serialize all Gpu nodes by adding a
  // dependency between all nodes added to a command buffer. We need a concept
  // of a barrier at a command buffer level.
  Dependencies GetDependencies();

  // Returns OK status if command buffer is not finalized and it is still
  // possible to add new commands to it, otherwise returns internal error.
  tsl::Status CheckNotFinalized();

  // Returns OK status if command buffer is primary, otherwise returns internal
  // error.
  tsl::Status CheckPrimary();

  static_assert(std::is_pointer_v<GpuGraphHandle>,
                "GpuGraphHandle must be a pointer");
  static_assert(std::is_pointer_v<GpuGraphExecHandle>,
                "GpuGraphExecHandle must be a pointer");
  static_assert(std::is_pointer_v<GpuGraphNodeHandle>,
                "GpuGraphNodeHandle must be a pointer");

  CommandBuffer::Mode mode_;
  CommandBuffer::State state_ = CommandBuffer::State::kCreate;

  GpuExecutor* parent_;  // not owned, must outlive *this

  GpuGraphHandle graph_ = nullptr;  // owned if `is_owned_graph_`
  bool is_owned_graph_ = true;      // ownership of `graph_`

  GpuGraphExecHandle exec_ = nullptr;  // owned if `is_owned_graph_exec_`
  bool is_owned_graph_exec_ = true;    // ownership of `is_owned_graph_exec_`

  // Handles to graph nodes corresponding to command buffer commands. Owned by
  // the `graph_` instance.
  std::vector<GpuGraphNodeHandle> nodes_;

  // Command buffers for conditional nodes in the Gpu graph. Underlying Gpu
  // graphs owned by the `graph_` instance.
  std::vector<ConditionalCommandBuffers> conditional_command_buffers_;

  // Track the number of command buffer updates for debugging.
  int64_t num_updates_ = 0;

  // Tracks indices into internal data structures during command buffer updates.
  struct UpdateState {
    // Index points to the graph node inside `nodes_` that will be updated next.
    int64_t node_idx = 0;

    // Index points to the conditional command buffers that will be updated next
    // when we'll be updating next conditional command (If, Case, While).
    int64_t conditional_idx = 0;
  };

  UpdateState update_state_;
};

template <typename... Params, typename... Args>
inline tsl::Status GpuCommandBuffer::Launch(
    const TypedKernel<Params...>& kernel, const ThreadDim& threads,
    const BlockDim& blocks, Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  TF_RETURN_IF_ERROR(Launch(threads, blocks, kernel, *kernel_args));
  return tsl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Implementation details device kernels required by GpuCommandBuffer.
//===----------------------------------------------------------------------===//

// See `cuda_conditional_kernels.cu.cc` for CUDA implementations. These are
// various kernels that update Gpu conditionals based on the device memory
// values, and allow implementing on-device control flow via conditional command
// buffers.

void* GetSetConditionKernel();

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_
