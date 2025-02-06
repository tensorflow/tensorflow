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
#include <variant>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/lib/gtl/int_type.h"
#include "tsl/platform/errors.h"

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
class CommandBuffer {
 public:
  // Execution scope enables fine-grained synchronization scopes inside
  // commands buffers. Implementation is very backend-specific and for CUDA/ROCM
  // backends it's implemented as DAG edges. By default all commands launched in
  // the `kDefaultExecutionScope` execution scope.
  //
  // Example #1: independent execution scopes and independent barriers
  //
  // ExecutionScope #0       ExecutionScope #1
  //
  //          A                        D
  //          B                        E
  // ----- barrier -----      ----- barrier -----
  //          C                        F
  //
  //   (1) Commands A and B can run concurrently and must complete before C.
  //   (2) Commands D and E can run concurrently and must complete before F.
  //   (3) There is no syncrhonization between execution scopes, and commands
  //       from different execution scopes can execute concurrently with each
  //       other as long as they satisfy constraints of their respective
  //       execution scopes.
  //
  //
  //
  // Example #2: dependencies between scopes and inter-scope barriers
  //
  // ExecutionScope #0       ExecutionScope #1
  //
  //          A                        D
  //          B                        E
  // ----------------- barrier ------------------
  //          C                        F
  //
  //   (1) Commands A and B can run concurrently and must complete before
  //       C and F.
  //   (2) Commands D and E can run concurrently and must complete before
  //       C and F.
  //   (3) Commands C and F can run concurrently.
  //   (4) All commands before a shared barrier (in both excecution scopes)
  //       should complete before any command after a berrier starts execution.
  //
  //
  //
  // Example #3: one-directional barriers between execution scopes
  //
  // ExecutionScope #0       ExecutionScope #1
  //
  //          A
  //          B
  // ----- barrier -----               D
  //          C            \           E
  //                           ----- barrier -----
  //                                   F
  //
  //   (1) Commands A and B can run concurrently and must complete before
  //       C and F.
  //   (2) Commands D and E can run concurrently and must complete before
  //       F (does not synchronize with C).
  //   (3) Commands C and F can run concurrently.
  //
  //  This is a more fine-grained barrier than in example #2: it enforces
  //  synchronization from execution scope #0 to execution scope #1 but no
  //  synchronization in other direction. For CUDA/ROCM backend it has the same
  //  semantics as stream wait operation.
  //
  TSL_LIB_GTL_DEFINE_INT_TYPE(ExecutionScopeId, uint64_t);
  static constexpr auto kDefaultExecutionScope = ExecutionScopeId(0);

  // Builder constructs nested command buffers owned by a parent command buffer.
  //
  // Builder can use arbitrary number of nested execution scopes, the only
  // requirement is that after builder constructed all commands, they all must
  // be synchronized with a default execution scope.
  using Builder = std::function<absl::Status(CommandBuffer*)>;

  // An extension of a `Builder` defined above that builds a nested command
  // buffer in a given execution scope. Builder can use arbitrary number of
  // nested execution scopes, the only requirement is that after builder
  // constructed all commands, they all must be synchronized with an execution
  // scope passed as an argument.
  using ExecutionScopeBuilder =
      std::function<absl::Status(ExecutionScopeId, CommandBuffer*)>;

  CommandBuffer() = default;
  virtual ~CommandBuffer() = default;

  CommandBuffer(const CommandBuffer&) = delete;
  void operator=(const CommandBuffer&) = delete;

  // Command buffer state:
  //
  //   (1) kCreate:    a new command buffer under construction
  //   (2) kUpdate:    updating a previously finalized command buffer
  //   (3) kFinalized: command buffer ready for execution
  //
  // Supported state transitions:
  //
  //   (1) Finalize: (kCreate|kUpdate) -> kFinalized
  //   (2) Update:   kFinalized -> kUpdate
  //
  enum class State { kCreate, kUpdate, kFinalized };

  // Command buffers have two modes of execution:
  //
  //   (1) kPrimary: command buffer can be submitted for execution via
  //                 StreamExecutor APIs
  //   (2) kNested:  command buffer can be executed only within a primary
  //                 command buffer
  //
  enum class Mode { kPrimary, kNested };

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

  // Adds an execution barrier to a given execution scope: all commands added
  // before a barrier in a the execution scope will complete before any of the
  // commands added after a barrier in the same execution scope.
  virtual absl::Status Barrier(ExecutionScopeId execution_scope_id) = 0;

  // Adds an execution barrier that synchronizes commands across multiple
  // execution scopes. See example #2 in execution scope id documentation.
  virtual absl::Status Barrier(
      absl::Span<const ExecutionScopeId> execution_scope_ids) = 0;

  // Adds an execution barrier from execution scope `from_execution_scope_id` to
  // execution scope `to_execution_scope_id`. See example #3 for details.
  virtual absl::Status Barrier(ExecutionScopeId from_execution_scope_id,
                               ExecutionScopeId to_execution_scope_id) = 0;

  // Adds an execution barrier to the default execution scope.
  absl::Status Barrier() { return Barrier(kDefaultExecutionScope); }

  // Adds a kernel launch command.
  virtual absl::Status Launch(ExecutionScopeId execution_scope_id,
                              const ThreadDim& threads, const BlockDim& blocks,
                              const Kernel& kernel, const KernelArgs& args) = 0;

  // Adds a kernel launch command to the default execution scope.
  absl::Status Launch(const ThreadDim& threads, const BlockDim& blocks,
                      const Kernel& kernel, const KernelArgs& args) {
    return Launch(kDefaultExecutionScope, threads, blocks, kernel, args);
  }

  // Type-safe wrapper for launching typed kernels. Notice that the order of
  // arguments is different do disambiguate from the regular launch API.
  template <typename... Params, typename... Args>
  absl::Status Launch(const TypedKernel<Params...>& kernel,
                      ExecutionScopeId execution_scope_id,
                      const ThreadDim& threads, const BlockDim& blocks,
                      Args... args);

  // Type-safe wrapper for launching typed kernels in default execution scope.
  template <typename... Params, typename... Args>
  absl::Status Launch(const TypedKernel<Params...>& kernel,
                      const ThreadDim& threads, const BlockDim& blocks,
                      Args... args) {
    return Launch(kernel, kDefaultExecutionScope, threads, blocks, args...);
  }

  // Adds a nested command buffer.
  virtual absl::Status AddNestedCommandBuffer(
      ExecutionScopeId execution_scope_id, const CommandBuffer& nested) = 0;

  // Adds a nested command buffer to the default execution scope.
  absl::Status AddNestedCommandBuffer(const CommandBuffer& nested) {
    return AddNestedCommandBuffer(kDefaultExecutionScope, nested);
  }

  // Adds a device-to-device memory copy.
  virtual absl::Status MemcpyDeviceToDevice(ExecutionScopeId execution_scope_id,
                                            DeviceMemoryBase* dst,
                                            const DeviceMemoryBase& src,
                                            uint64_t size) = 0;

  // Adds a device-to-device memory copy to the default execution scope.
  absl::Status MemcpyDeviceToDevice(DeviceMemoryBase* dst,
                                    const DeviceMemoryBase& src,
                                    uint64_t size) {
    return MemcpyDeviceToDevice(kDefaultExecutionScope, dst, src, size);
  }

  // Adds a memset command.
  virtual absl::Status Memset(ExecutionScopeId execution_scope_id,
                              DeviceMemoryBase* dst, BitPattern bit_pattern,
                              size_t num_elements) = 0;

  // Adds a memset command to the default execution scope.
  absl::Status Memset(DeviceMemoryBase* dst, BitPattern bit_pattern,
                      size_t num_elements) {
    return Memset(kDefaultExecutionScope, dst, bit_pattern, num_elements);
  }

  //--------------------------------------------------------------------------//
  // Command buffer condtitional commands API
  //--------------------------------------------------------------------------//

  // Adds a conditional operation that will execute a command buffer constructed
  // by `then_builder` if `pred` value is `true`.
  virtual absl::Status If(ExecutionScopeId execution_scope_id,
                          DeviceMemory<bool> pred, Builder then_builder) = 0;

  // Adds a conditional If operation to default execution scope.
  absl::Status If(DeviceMemory<bool> pred, Builder then_builder) {
    return If(kDefaultExecutionScope, pred, then_builder);
  }

  // Adds a conditional operation that will execute a command buffer constructed
  // by `then_builder` if `pred` value is `true`, or a command buffer
  // constructed by `else_builder` if `pred` is `false`.
  virtual absl::Status IfElse(ExecutionScopeId execution_scope_id,
                              DeviceMemory<bool> pred, Builder then_builder,
                              Builder else_builder) = 0;

  // Adds a conditional IfElse operation to default execution scope.
  absl::Status IfElse(DeviceMemory<bool> pred, Builder then_builder,
                      Builder else_builder) {
    return IfElse(kDefaultExecutionScope, pred, then_builder, else_builder);
  }

  // Adds a conditional operation that will execute a command buffer constructed
  // by the `branches` builder at `index`. If `index` is out of range, then it
  // will run a conditional command buffer constructed by the last builder.
  //
  // See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case
  virtual absl::Status Case(ExecutionScopeId execution_scope_id,
                            DeviceMemory<int32_t> index,
                            std::vector<Builder> branches) = 0;

  // Adds a conditional Case operation to default execution scope.
  absl::Status Case(DeviceMemory<int32_t> index,
                    std::vector<Builder> branches) {
    return Case(kDefaultExecutionScope, index, branches);
  }

  // Adds a conditional operation that will execute a command buffer constructed
  // by the `body_builder` exactly `num_iteration` times. This means the
  // condition is known at compile time (`num_iteration` < `loop_counter`), and
  // does not require a `cond_builder`.
  virtual absl::Status For(ExecutionScopeId execution_scope_id,
                           int32_t num_iteration,
                           DeviceMemory<int32_t> loop_counter,
                           Builder body_builder) = 0;

  // Adds a conditional For operation to default execution scope.
  absl::Status For(int32_t num_iteration, DeviceMemory<int32_t> loop_counter,
                   Builder body_builder) {
    return For(kDefaultExecutionScope, num_iteration, loop_counter,
               body_builder);
  }

  // Adds a conditional operation that will execute a command buffer constructed
  // by the `cond_builder` that must update `pred` value, and then depending on
  // the value might execute command buffer constructed by `body_builder` and
  // `cond_builder`. Will continue while `pred` value (which is continuously
  // updated by `cond_builder`) is `true`.
  //
  // In pseudocode:
  //
  //   cond_builder()
  //   while(pred):
  //     body_builder()
  //     cond_builder()
  //
  // We use execution scope builder for the condition because we have to build
  // condition twice: (1) before the conditional node in the scope defined by
  // `execution_scope_id` (2) inside the loop body with default execution scope.
  virtual absl::Status While(ExecutionScopeId execution_scope_id,
                             DeviceMemory<bool> pred,
                             ExecutionScopeBuilder cond_builder,
                             Builder body_builder) = 0;

  // Adds a conditional While operation to default execution scope.
  absl::Status While(DeviceMemory<bool> pred,
                     ExecutionScopeBuilder cond_builder, Builder body_builder) {
    return While(kDefaultExecutionScope, pred, cond_builder, body_builder);
  }

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

  // Begins command buffer update. Command buffer update should be finalized
  // before it can be executed.
  virtual absl::Status Update() = 0;

  // Returns command buffer execution mode.
  virtual Mode mode() const = 0;

  // Returns command buffer state.
  virtual State state() const = 0;

  //--------------------------------------------------------------------------//
  // Command buffer tracing API
  //--------------------------------------------------------------------------//
 private:
  friend class TraceCommandBufferFactory;
  // Tracing APIs are private because they do not compose with command buffer
  // updates. Instead of tracing directly into the command buffer users should
  // create traced command buffers using factory methods and add them to primary
  // command buffers as nested operations.

  // Traces `function` invocation by recording all operations on the `stream`
  // into the command buffer. Command buffer must be empty.
  virtual absl::Status Trace(Stream* stream,
                             absl::AnyInvocable<absl::Status()> function) = 0;
};

//===----------------------------------------------------------------------===//
// CommandBuffer templates implementation below
//===----------------------------------------------------------------------===//

template <typename... Params, typename... Args>
inline absl::Status CommandBuffer::Launch(const TypedKernel<Params...>& kernel,
                                          ExecutionScopeId execution_scope_id,
                                          const ThreadDim& threads,
                                          const BlockDim& blocks,
                                          Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  TF_RETURN_IF_ERROR(
      Launch(execution_scope_id, threads, blocks, *kernel, *kernel_args));
  return absl::OkStatus();
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_
