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
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/stream_executor_internal.h"
#include "tsl/platform/status.h"

namespace stream_executor::gpu {

// GpuCommandBuffer provides platform-specific CommandBufferInterface
// implementation (it's backed by CUDA or HIP graphs on NVIDIA and AMD devices).
class GpuCommandBuffer : public internal::CommandBufferInterface {
 public:
  GpuCommandBuffer(CommandBuffer::Mode mode, GpuExecutor* parent,
                   GpuGraphHandle graph);
  ~GpuCommandBuffer() override;

  tsl::Status Trace(Stream* stream,
                    absl::AnyInvocable<tsl::Status()> function) override;

  tsl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                     const Kernel& kernel, const KernelArgs& args) override;

  tsl::Status AddNestedCommandBuffer(const CommandBuffer& nested) override;

  tsl::Status MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                   const DeviceMemoryBase& src,
                                   uint64_t size) override;

  tsl::Status Finalize() override;
  tsl::Status Update() override;

  GpuGraphExecHandle executable() const { return exec_; }
  GpuGraphHandle graph() const { return graph_; }

  CommandBuffer::Mode mode() const override { return mode_; }
  CommandBuffer::State state() const override { return state_; }

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

  static const GpuCommandBuffer* Cast(const CommandBuffer* command_buffer) {
    return static_cast<const GpuCommandBuffer*>(
        command_buffer->implementation());
  }

 private:
  // TODO(ezhulenev): Currently we serialize all Gpu nodes by adding a
  // dependency between all nodes added to a command buffer. We need a concept
  // of a barrier at a command buffer level.
  using Dependencies = absl::InlinedVector<GpuGraphNodeHandle, 1>;
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

  GpuExecutor* parent_;                // not owned, must outlive *this
  GpuGraphHandle graph_ = nullptr;     // owned handle
  GpuGraphExecHandle exec_ = nullptr;  // owned handle

  // Handles to graph nodes corresponding to command buffer commands. Owned by
  // the `graph_` instance.
  std::vector<GpuGraphNodeHandle> nodes_;

  // When command buffer is in update state this index will point to the graph
  // node inside `nodes_` that will be updated next.
  int64_t node_update_idx_ = 0;

  // Track the number of command buffer updates for debugging.
  int64_t num_updates_ = 0;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_COMMAND_BUFFER_H_
