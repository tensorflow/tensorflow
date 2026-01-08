/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_STATE_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_STATE_H_

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace stream_executor {
class CommandBuffer;
}

namespace xla::gpu {

// Forward declare and prepare to migration to `Command` type name.
class CommandBufferCmd;
using Command = CommandBufferCmd;

// A base class for externally managed command state.
//
// Commands can be executed concurrently on many stream executors (underlying
// devices) and command buffers. Managing per-executor state can become
// expensive as it requires synchronization. Furthermore the number of command
// buffers command is recorded into is unbounded as they come and go (command
// buffers evicted and reconstructed) which makes it hard to manage the
// lifetime of resources attached to command buffers.
//
// Externally managed state (owned and synchronized by CommandBufferThunk)
// allows commands to attach a piece of information to command buffer in a
// safe and performant way.
//
// To make a command stateful, it needs a `CommandStateManager` indirection:
//
//   class MyCommand : public Command {
//     public:
//
//     // Container for mutable state required for command execution.
//     struct MyState : CommandState {
//       ...
//     };
//
//     absl::StatusOr<Command*> Record(...) override {
//       // Attach a new instance of `MyState` to the active command buffer.
//       // When a command buffer will be destroyed, the state will be destroyed
//       // as well automatically by XLA runtime. If this command will be
//       // recorded into another command buffer, the state will be re-created
//       // automatically using the provided callback.
//       //
//       // CommandBufferThunk guarantees that the state manger passed to a
//       // command recording function is tied to exactly the same command/
//       // buffer that command is recording into.
//       MyState* my_state = record_params.state.GetOrCreate<MyState>(this,
//         [&] { // create a new instnace of `MyState` });
//       ...
//     }
//
//   };
//
class CommandState {
 public:
  virtual ~CommandState() = default;
};

// Command state manager owns command state recorded into the `command_buffer`
// by commands in a command sequence. State is created lazily the first time
// command is recorded using a given state manager (into a given command
// buffer). State manager is owned by a command buffer thunk together with
// the command buffer itself and they are destroyed together, which ties state
// lifetime to the command buffer.
//
// Note that the same command can be recorded as a part of multiple iterations
// of unrolled loop, and for this reason the state can be attached to a
// concreate iteration index. Also for unrolled loops the same command can be
// recorded into multiple command buffers (for cond and body computations), and
// for this reason state is attached to a triple: (command, command_buffer,
// unroll_iteration).
class CommandStateManager {
 public:
  template <typename State>
  State* absl_nullable GetOrNull(
      const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
      int64_t unroll_iteration = 0);

  template <typename State>
  State* absl_nonnull GetOrCreate(
      const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
      absl::FunctionRef<std::unique_ptr<State>()> create,
      int64_t unroll_iteration = 0);

  template <typename State>
  State* absl_nonnull GetOrCreate(
      const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
      int64_t unroll_iteration = 0);

 private:
  // We use strongly typed TypeId to distinguish between different state types.
  TSL_LIB_GTL_DEFINE_INT_TYPE(TypeId, int64_t);

  static TypeId GetNextTypeId();

  template <typename T>
  static TypeId GetTypeId() {
    static const TypeId id = GetNextTypeId();
    return id;
  }

  CommandState* absl_nullable GetOrNull(
      const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
      TypeId type_id, int64_t unroll_iteration);

  CommandState* absl_nonnull GetOrCreate(
      const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
      TypeId type_id, int64_t unroll_iteration,
      absl::FunctionRef<std::unique_ptr<CommandState>()> create);

  using Key = std::tuple<const Command*, const stream_executor::CommandBuffer*,
                         TypeId, int64_t>;
  absl::flat_hash_map<Key, std::unique_ptr<CommandState>> state_;
};

//===----------------------------------------------------------------------===//
// CommandStateManager templates implementation
//===----------------------------------------------------------------------===//

template <typename State>
State* CommandStateManager::GetOrNull(
    const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
    int64_t unroll_iteration) {
  static_assert(std::is_base_of_v<CommandState, State>);
  return static_cast<State*>(
      GetOrNull(cmd, command_buffer, GetTypeId<State>(), unroll_iteration));
}

template <typename State>
State* CommandStateManager::GetOrCreate(
    const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
    absl::FunctionRef<std::unique_ptr<State>()> create,
    int64_t unroll_iteration) {
  static_assert(std::is_base_of_v<CommandState, State>);
  return static_cast<State*>(GetOrCreate(cmd, command_buffer,
                                         GetTypeId<State>(), unroll_iteration,
                                         [&] { return create(); }));
}

template <typename State>
State* CommandStateManager::GetOrCreate(
    const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
    int64_t unroll_iteration) {
  return GetOrCreate<State>(
      cmd, command_buffer, [] { return std::make_unique<State>(); },
      unroll_iteration);
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_STATE_H_
