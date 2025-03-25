/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_
#define XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/errors.h"

namespace stream_executor {

class Stream;

//===----------------------------------------------------------------------===//
// CommandBuffer
//===----------------------------------------------------------------------===//

// Command buffer represent a "bundle of work items" for StreamExecutor device
// that can be submitted with one API call, e.g. command buffer might have
// multiple device kernels and synchronization barriers between them. Command
// buffers allow to amortize the cost of launching "work" on device by building
// it on the host ahead of time without expensive interaction with underlying
// device.

// CommandBuffer lowers CommandBufferCmdSequence to a GPU graph, commands in the
// sequence have dependencies that can specified by command index in the
// sequence. CommandBuffer implementation will convert dependencies through
// command index to node dependencies in the implementation.
class CommandBuffer {
 public:
  // std::vector<NodeHandle> are created by node factory functions
  // and can be referenced in node update functions. The handle has the same
  // properties as a pointer (can be constructed from a nullptr, trivial
  // copyable, POD, etc.), that's why we use a pointer to define it.

  // NodeHandleOpaque is an opaque type that won't be ODR used,
  // hence doesn't need to fully defined. It's an implementation detail of the
  // NodeHandle defined below.
  struct NodeHandleOpaque;
  struct ConditionalHandleOpaque;

  using NodeHandle = NodeHandleOpaque*;
  using Dependencies = absl::InlinedVector<NodeHandle, 1>;
  static std::string GraphNodeHandlesToString(
      absl::Span<const NodeHandle> handles) {
    std::vector<std::string> elements;
    elements.reserve(handles.size());
    for (const auto& handle : handles) {
      elements.push_back(
          absl::StrCat("0x", absl::Hex(reinterpret_cast<uintptr_t>(handle))));
    }
    return absl::StrCat("std::vector<NodeHandle>: [",
                        absl::StrJoin(elements, ", "), "]");
  }

  // A graph conditional handle is an opaque handle that is tied to a nested
  // command buffer. Its value determines whether the nested command buffer
  // is executed or not. Set condition functions will update the conditional
  // handles values. The handle has the same properties as a pointer (can be
  // constructed from a nullptr, trivially copyable, POD, etc.), that's why
  // we use a pointer to define it.
  using ConditionalHandle = ConditionalHandleOpaque*;

  struct ConditionalNodeResult {
    NodeHandle node_handle;
    std::unique_ptr<CommandBuffer> command_buffer;
  };

  enum class ConditionType { kIf, kWhile };

  // Command buffers have two modes of execution:
  //
  //   (1) kPrimary: command buffer can be submitted for execution via
  //                 StreamExecutor APIs
  //   (2) kNested:  command buffer can be executed only within a primary
  //                 command buffer
  //
  enum class Mode { kPrimary, kNested };

  CommandBuffer(Mode mode) : mode_(mode) {}
  virtual ~CommandBuffer() = default;

  CommandBuffer(const CommandBuffer&) = delete;
  void operator=(const CommandBuffer&) = delete;

  friend absl::string_view ModeToString(Mode mode) {
    switch (mode) {
      case CommandBuffer::Mode::kPrimary:
        return "primary";
      case CommandBuffer::Mode::kNested:
        return "nested";
    }
  }

  //===--------------------------------------------------------------------===//
  // Command buffer API
  //===--------------------------------------------------------------------===//

  // Adds an execution barrier that depends on the commands in deps.
  virtual absl::StatusOr<NodeHandle> CreateEmptyNode(
      Dependencies dependencies) = 0;

  // Adds a kernel launch command that depends on the commands in
  // deps.
  virtual absl::StatusOr<NodeHandle> CreateLaunchNode(
      Dependencies dependencies, const ThreadDim& threads,
      const BlockDim& blocks, const Kernel& kernel, const KernelArgs& args) = 0;

  virtual absl::Status UpdateLaunchNode(NodeHandle node,
                                        const ThreadDim& threads,
                                        const BlockDim& blocks,
                                        const Kernel& kernel,
                                        const KernelArgs& args) = 0;

  // Type-safe wrapper for launching typed kernels. Notice that the order of
  // arguments is different do disambiguate from the regular launch API.
  template <typename... Params, typename... Args>
  absl::StatusOr<NodeHandle> CreateTypedLaunchNode(
      Dependencies dependencies, const ThreadDim& threads,
      const BlockDim& blocks, const TypedKernel<Params...>& kernel,
      Args... args);

  template <typename... Params, typename... Args>
  absl::Status UpdateTypedLaunchNode(NodeHandle node, const ThreadDim& threads,
                                     const BlockDim& blocks,
                                     const TypedKernel<Params...>& kernel,
                                     Args... args);

  virtual absl::StatusOr<NodeHandle> CreateChildNode(
      Dependencies dependencies, const CommandBuffer& child) = 0;

  virtual absl::Status UpdateChildNode(NodeHandle node,
                                       const CommandBuffer& child) = 0;

  // Adds a device-to-device memory copy that depends on the commands in
  // deps.
  virtual absl::StatusOr<NodeHandle> CreateMemcpyD2DNode(
      Dependencies dependencies, DeviceMemoryBase dst, DeviceMemoryBase src,
      uint64_t size) = 0;

  virtual absl::Status UpdateMemcpyD2DNode(NodeHandle node,
                                           DeviceMemoryBase dst,
                                           DeviceMemoryBase src,
                                           uint64_t size) = 0;

  // Adds a memset command that depends on the commands in deps.
  virtual absl::StatusOr<NodeHandle> CreateMemsetNode(Dependencies dependencies,
                                                      DeviceMemoryBase dst,
                                                      BitPattern bit_pattern,
                                                      size_t num_elements) = 0;

  virtual absl::Status UpdateMemsetNode(NodeHandle node, DeviceMemoryBase dst,
                                        BitPattern bit_pattern,
                                        size_t num_elements) = 0;

  //--------------------------------------------------------------------------//
  // Command buffer condtitional commands API
  //--------------------------------------------------------------------------//

  // Create a new conditional handle in the underlying
  // graph.
  virtual absl::StatusOr<ConditionalHandle> CreateConditionalHandle() = 0;

  // Adds a new conditional node to the graph and creates a
  // corresponding nested command buffer.
  virtual absl::StatusOr<ConditionalNodeResult> CreateConditionalNode(
      Dependencies dependencies, ConditionalHandle conditional,
      ConditionType type) = 0;

  virtual absl::StatusOr<NodeHandle> CreateIfElseConditionNode(
      Dependencies dependencies, ConditionalHandle then_condition,
      ConditionalHandle else_condition, DeviceMemory<bool> predicate) = 0;

  virtual absl::Status UpdateIfElseConditionNode(
      NodeHandle node, ConditionalHandle then_condition,
      ConditionalHandle else_condition, DeviceMemory<bool> predicate) = 0;

  virtual absl::StatusOr<NodeHandle> CreateIfConditionNode(
      Dependencies dependencies, ConditionalHandle then_condition,
      DeviceMemory<bool> predicate) = 0;

  virtual absl::Status UpdateIfConditionNode(NodeHandle node,
                                             ConditionalHandle then_condition,
                                             DeviceMemory<bool> predicate) = 0;

  virtual absl::StatusOr<NodeHandle> CreateForConditionNode(
      Dependencies dependencies, ConditionalHandle condition,
      DeviceMemory<int32_t> loop_counter, int32_t iterations) = 0;

  virtual absl::Status UpdateForConditionNode(
      NodeHandle node, ConditionalHandle condition,
      DeviceMemory<int32_t> loop_counter, int32_t iterations) = 0;

  virtual absl::StatusOr<NodeHandle> CreateWhileConditionNode(
      Dependencies dependencies, ConditionalHandle condition,
      DeviceMemory<bool> predicate) = 0;

  virtual absl::Status UpdateWhileConditionNode(
      NodeHandle node, ConditionalHandle condition,
      DeviceMemory<bool> predicate) = 0;

  virtual absl::StatusOr<NodeHandle> CreateCaseConditionNode(
      Dependencies dependencies, std::array<ConditionalHandle, 8> conditions,
      DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      int32_t num_branches, bool enable_conditional_default) = 0;

  virtual absl::Status UpdateCaseConditionNode(
      NodeHandle node, std::array<ConditionalHandle, 8> conditions,
      DeviceMemory<uint8_t> index, bool index_is_bool, int32_t batch_offset,
      int32_t num_branches, bool enable_conditional_default) = 0;

  // Submits the command buffer for execution.
  virtual absl::Status Submit(Stream* stream) {
    return absl::UnimplementedError("Not implemented for this command buffer.");
  }

  //--------------------------------------------------------------------------//
  // Command buffer state management API
  //--------------------------------------------------------------------------//

  // Finalizes command buffer and makes it executable. Once command buffer is
  // finalized no commands can be added to it.
  virtual absl::Status Finalize() = 0;

  // Returns command buffer execution mode.
  Mode mode() const { return mode_; }

  //--------------------------------------------------------------------------//
  // Command buffer tracing API
  //--------------------------------------------------------------------------//
 private:
  friend class TraceCommandBufferFactory;

  // Tracing APIs are private because they do not compose with command buffer
  // updates. Instead of tracing directly into the command buffer users should
  // create traced command buffers using factory methods and add them to primary
  // command buffers as nested operations.
  Mode mode_;

  // Traces `function` invocation by recording all operations on the `stream`
  // into the command buffer. Command buffer must be empty.
  virtual absl::Status Trace(Stream* stream,
                             absl::AnyInvocable<absl::Status()> function) = 0;
};

//===----------------------------------------------------------------------===//
// CommandBuffer templates implementation below
//===----------------------------------------------------------------------===//

template <typename... Params, typename... Args>
inline absl::StatusOr<CommandBuffer::NodeHandle>
CommandBuffer::CreateTypedLaunchNode(Dependencies dependencies,
                                     const ThreadDim& threads,
                                     const BlockDim& blocks,
                                     const TypedKernel<Params...>& kernel,
                                     Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  return CreateLaunchNode(dependencies, threads, blocks, *kernel, *kernel_args);
}

template <typename... Params, typename... Args>
inline absl::Status CommandBuffer::UpdateTypedLaunchNode(
    CommandBuffer::NodeHandle node, const ThreadDim& threads,
    const BlockDim& blocks, const TypedKernel<Params...>& kernel,
    Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  return UpdateLaunchNode(node, threads, blocks, *kernel, *kernel_args);
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_
