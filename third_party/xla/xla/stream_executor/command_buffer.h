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
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/stream_executor/bit_pattern.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/lib/gtl/int_type.h"

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
  // Command represents an operation recorded into a command buffer. It's owned
  // by the command buffer and returned to the caller to enable efficient
  // command buffer updates.
  class Command {
   public:
    virtual ~Command() = default;

   protected:
    Command() = default;

    Command(const Command&) = default;
    Command& operator=(const Command&) = default;

    Command(Command&&) = default;
    Command& operator=(Command&&) = default;
  };

  // A callback to construct a nested `command_buffer` by creating commands in
  // it. Created commands must execute after `dependencies`, and the callback
  // must return a vector of commands that will be used as external dependencies
  // for the next callback recording into the same command buffer.
  using CreateCommands =
      absl::AnyInvocable<absl::StatusOr<std::vector<const Command*>>(
          CommandBuffer* command_buffer,
          absl::Span<const Command* const> dependencies)>;

  // A callback to update a nested `command_buffer` owned by a conditional
  // command. At command buffer update time we can't change the dependency
  // structure of the previously created commands, and can only update the
  // parameters of the commands (i.e. device memory pointers).
  using UpdateCommands =
      absl::AnyInvocable<absl::Status(CommandBuffer* command_buffer)>;

  CommandBuffer() = default;
  virtual ~CommandBuffer() = default;

  CommandBuffer(const CommandBuffer&) = delete;
  void operator=(const CommandBuffer&) = delete;

  // A base class for external resources that can be attached to a CommandBuffer
  // instance. When a CommandBuffer instance is destroyed, all attached
  // resources are destroyed as well.
  class Resource {
   public:
    virtual ~Resource() = default;
  };

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

  template <typename Sink>
  friend void AbslStringify(Sink& sink, State state) {
    switch (state) {
      case CommandBuffer::State::kCreate:
        sink.Append("create");
        break;
      case CommandBuffer::State::kUpdate:
        sink.Append("update");
        break;
      case CommandBuffer::State::kFinalized:
        sink.Append("finalized");
        break;
    }
  }

  // Command buffers have two modes of execution:
  //
  //   (1) kPrimary: command buffer can be submitted for execution via
  //                 StreamExecutor APIs
  //   (2) kNested:  command buffer can be executed only within a primary
  //                 command buffer
  //
  enum class Mode { kPrimary, kNested };

  template <typename Sink>
  friend void AbslStringify(Sink& sink, Mode mode) {
    switch (mode) {
      case CommandBuffer::Mode::kPrimary:
        sink.Append("primary");
        break;
      case CommandBuffer::Mode::kNested:
        sink.Append("nested");
        break;
    }
  }

  //===--------------------------------------------------------------------===//
  // Command buffer API
  //===--------------------------------------------------------------------===//

  // Creates an empty command.
  virtual absl::StatusOr<const Command*> CreateEmptyCmd(
      absl::Span<const Command* const> dependencies,
      StreamPriority priority = StreamPriority::Default) = 0;

  // Creates a kernel launch command.
  virtual absl::StatusOr<const Command*> CreateLaunch(
      const ThreadDim& threads, const BlockDim& blocks, const Kernel& kernel,
      const KernelArgs& args, absl::Span<const Command* const> dependencies,
      StreamPriority priority = StreamPriority::Default) = 0;

  // Updates a kernel launch command.
  virtual absl::Status UpdateLaunch(const Command* command,
                                    const ThreadDim& threads,
                                    const BlockDim& blocks,
                                    const Kernel& kernel,
                                    const KernelArgs& args) = 0;

  // Type-safe wrapper for launching typed kernels. Notice that the order of
  // arguments is different do disambiguate from the regular launch API.
  template <typename... Params, typename... Args>
  absl::StatusOr<const Command*> CreateLaunch(
      const TypedKernel<Params...>& kernel, const ThreadDim& threads,
      const BlockDim& blocks, absl::Span<const Command* const> dependencies,
      Args... args);

  // Type-safe wrapper for updating typed kernels. Notice that the order of
  // arguments is different do disambiguate from the regular launch API.
  template <typename... Params, typename... Args>
  absl::Status UpdateLaunch(const Command* command,
                            const TypedKernel<Params...>& kernel,
                            const ThreadDim& threads, const BlockDim& blocks,
                            Args... args);

  // Creates a child command from a pre-recorded command buffer.
  virtual absl::StatusOr<const Command*> CreateChildCommand(
      const CommandBuffer& nested,
      absl::Span<const Command* const> dependencies) = 0;

  // Updates a child command with a new command buffer. New command buffer must
  // be compatible with the one that was used to create command.
  virtual absl::Status UpdateChildCommand(const Command* command,
                                          const CommandBuffer& nested) = 0;

  // Creates a child command from a user-provided callback by recording commands
  // into the child command buffer owned by this command buffer. In contrast to
  // the child command created from pre-recorded command buffer (see API
  // above), such child commands can be efficiently updated by updating the
  // command buffer itself.
  virtual absl::StatusOr<const Command*> CreateChildCommand(
      absl::AnyInvocable<absl::Status(CommandBuffer*)> record_fn,
      absl::Span<const Command* const> dependencies) = 0;

  // Updates a child command using a user-provided callback to record command
  // updates into the child command buffer owned by this command buffer. In
  // contrast to the child command update from pre-recorded command buffer (see
  // API above) it allows fine grained command update. Updating from a
  // pre-recorded command buffer requires swapping the whole child command
  // buffer to a new one, which might be expensive for large command buffers.
  virtual absl::Status UpdateChildCommand(
      const Command* command,
      absl::AnyInvocable<absl::Status(CommandBuffer*)> update_fn) = 0;

  // Creates a device-to-device memory copy.
  virtual absl::StatusOr<const Command*> CreateMemcpyD2D(
      DeviceAddressBase* dst, const DeviceAddressBase& src, uint64_t size,
      absl::Span<const Command* const> dependencies) = 0;

  // Updates a device-to-device memory copy.
  virtual absl::Status UpdateMemcpyD2D(const Command* command,
                                       DeviceAddressBase* dst,
                                       const DeviceAddressBase& src,
                                       uint64_t size) = 0;

  // Creates a memset command.
  virtual absl::StatusOr<const Command*> CreateMemset(
      DeviceAddressBase* dst, BitPattern bit_pattern, size_t num_elements,
      absl::Span<const Command* const> dependencies) = 0;

  // Updates a memset command.
  virtual absl::Status UpdateMemset(const Command* command,
                                    DeviceAddressBase* dst,
                                    const BitPattern& bit_pattern,
                                    size_t num_elements) = 0;

  //--------------------------------------------------------------------------//
  // Command buffer DNN graph API
  //--------------------------------------------------------------------------//

  // Creates a DNN graph launch command.
  virtual absl::StatusOr<const Command*> CreateDnnGraphCommand(
      dnn::DnnGraph&, Stream&, absl::Span<DeviceAddressBase> operands,
      absl::Span<const Command* const> dependencies) = 0;

  // Updates a DNN graph command.
  virtual absl::Status UpdateDnnGraphCommand(
      const Command*, dnn::DnnGraph&, Stream&,
      absl::Span<DeviceAddressBase> operands) = 0;

  //--------------------------------------------------------------------------//
  // Command buffer condtitional commands API
  //--------------------------------------------------------------------------//

  // Creates a conditional operation that will execute a command buffer
  // constructed by the `branches` builder at `index`. If `index` is out of
  // range, then it will run a conditional command buffer constructed by the
  // last builder.
  //
  // See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case
  virtual absl::StatusOr<const Command*> CreateCase(
      DeviceAddress<int32_t> index, std::vector<CreateCommands> create_branches,
      absl::Span<const Command* const> dependencies) = 0;

  virtual absl::StatusOr<const Command*> CreateCase(
      DeviceAddress<bool> index, std::vector<CreateCommands> create_branches,
      absl::Span<const Command* const> dependencies) = 0;

  // Updates a Case command.
  virtual absl::Status UpdateCase(
      const Command* command, DeviceAddress<int32_t> index,
      std::vector<UpdateCommands> update_branches) = 0;

  virtual absl::Status UpdateCase(
      const Command* command, DeviceAddress<bool> index,
      std::vector<UpdateCommands> update_branches) = 0;

  // Creates a conditional operation that will execute a command buffer
  // constructed by the `cond_builder` that must update `pred` value, and then
  // depending on the value might execute command buffer constructed by
  // `body_builder` and `cond_builder`. Will continue while `pred` value (which
  // is continuously updated by `cond_builder`) is `true`.
  //
  // In pseudocode:
  //
  //   cond_builder()
  //   while(pred):
  //     body_builder()
  //     cond_builder()
  //
  virtual absl::StatusOr<const Command*> CreateWhile(
      DeviceAddress<bool> pred, CreateCommands create_cond,
      CreateCommands create_body,
      absl::Span<const Command* const> dependencies) = 0;

  // Updates a While command.
  virtual absl::Status UpdateWhile(const Command* command,
                                   DeviceAddress<bool> pred,
                                   UpdateCommands update_cond,
                                   UpdateCommands update_body) = 0;

  // Set the priority of all nodes in the command buffer.
  virtual absl::Status SetPriority(StreamPriority priority) = 0;

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

  virtual std::string ToString() const = 0;

  //--------------------------------------------------------------------------//
  // Command buffer user-attached resources API
  //--------------------------------------------------------------------------//

  // Returns a pointer to the resource of the given type, or nullptr if resource
  // of the given type is not attached to this stream executor.
  template <typename ConcreteResource>
  ConcreteResource* GetOrNullResource() {
    static_assert(std::is_base_of_v<Resource, ConcreteResource>);
    return static_cast<ConcreteResource*>(
        GetOrNullResource(GetResourceTypeId<ConcreteResource>()));
  }

  // Returns a pointer to the resource of the given type, or creates a new
  // resource of the given type and attaches it to this stream executor.
  template <typename ConcreteResource>
  ConcreteResource* GetOrCreateResource(
      absl::FunctionRef<std::unique_ptr<ConcreteResource>()> create) {
    static_assert(std::is_base_of_v<Resource, ConcreteResource>);
    return static_cast<ConcreteResource*>(GetOrCreateResource(
        GetResourceTypeId<ConcreteResource>(), [&] { return create(); }));
  }

  // Returns a pointer to the resource of the given type, or creates a new
  // resource of the given type and attaches it to this stream executor.
  template <typename ConcreteResource>
  ConcreteResource* GetOrCreateResource() {
    return GetOrCreateResource<ConcreteResource>(
        [] { return std::make_unique<ConcreteResource>(); });
  }

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

  // We use ResourceTypeId to distinguish between different resource types.
  TSL_LIB_GTL_DEFINE_INT_TYPE(ResourceTypeId, int64_t);

  Resource* GetOrNullResource(ResourceTypeId type_id);
  Resource* GetOrCreateResource(
      ResourceTypeId type_id,
      absl::FunctionRef<std::unique_ptr<Resource>()> create);

  template <typename F>
  static ResourceTypeId GetResourceTypeId() {
    static const ResourceTypeId id = GetNextResourceTypeId();
    return id;
  }

  static ResourceTypeId GetNextResourceTypeId();

  absl::Mutex resource_mutex_;
  absl::flat_hash_map<ResourceTypeId, std::unique_ptr<Resource>> resources_
      ABSL_GUARDED_BY(resource_mutex_);
};

//===----------------------------------------------------------------------===//
// CommandBuffer templates implementation below
//===----------------------------------------------------------------------===//

template <typename... Params, typename... Args>
absl::StatusOr<const CommandBuffer::Command*> CommandBuffer::CreateLaunch(
    const TypedKernel<Params...>& kernel, const ThreadDim& threads,
    const BlockDim& blocks, absl::Span<const Command* const> dependencies,
    Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  return CreateLaunch(threads, blocks, *kernel, *kernel_args, dependencies);
}

template <typename... Params, typename... Args>
absl::Status CommandBuffer::UpdateLaunch(const Command* command,
                                         const TypedKernel<Params...>& kernel,
                                         const ThreadDim& threads,
                                         const BlockDim& blocks, Args... args) {
  auto kernel_args = PackKernelArgs(kernel, args...);
  return UpdateLaunch(command, threads, blocks, *kernel, *kernel_args);
}

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_COMMAND_BUFFER_H_
