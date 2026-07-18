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

#include <cstdint>
#include <memory>
#include <tuple>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace stream_executor {
class CommandBuffer;
}

namespace xla::gpu {

// Forward declaration.
class Command;

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
// Note that the same command executor can be recorded into multiple nested
// command buffers that belong to the same top-level executable command buffer,
// i.e. this can happed with nested control flow. For this reason the key
// for the state is a pair or `Command` and `se::CommandBuffer` which fully
// identify where exactly command is being recorded.
//
// IMPORTANT: Command can be recorded into the command buffer multiple times
// for different record ids (see `CommandExecutor::RecordId`). The typical
// command lifecycle is:
//
// (1) Command::Prepare() - prepare state for execution
// (2) Command::Initialize() - initialize resources for execution (or state)
// (3) Command::Record(create) - record commands into the command buffer
// (4) Command::Record(update) - update previousy recorded command
//
// Steps (1) and (2) called exactly one time for each top level XLA program
// execution. Steps (3) and (4) can be called multiple times. In the most
// common scenario step (3) called once when command recorded first time, and
// then every time XLA program executes it calls step (4) to update command
// with new buffer addresses.
class CommandStateManager {
 public:
  template <typename State>
  State* absl_nullable GetOrNull(
      const Command* cmd, const stream_executor::CommandBuffer* command_buffer);

  template <typename State>
  State* absl_nonnull GetOrCreate(
      const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
      absl::FunctionRef<std::unique_ptr<State>()> create);

  template <typename State>
  State* absl_nonnull GetOrCreate(
      const Command* cmd, const stream_executor::CommandBuffer* command_buffer);

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
      TypeId type_id);

  CommandState* absl_nonnull GetOrCreate(
      const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
      TypeId type_id,
      absl::FunctionRef<std::unique_ptr<CommandState>()> create);

  using Key =
      std::tuple<const Command*, const stream_executor::CommandBuffer*, TypeId>;
  absl::flat_hash_map<Key, std::unique_ptr<CommandState>> state_;
};

//===----------------------------------------------------------------------===//
// CommandStateManager templates implementation
//===----------------------------------------------------------------------===//

template <typename State>
State* CommandStateManager::GetOrNull(
    const Command* cmd, const stream_executor::CommandBuffer* command_buffer) {
  static_assert(std::is_base_of_v<CommandState, State>);
  return static_cast<State*>(
      GetOrNull(cmd, command_buffer, GetTypeId<State>()));
}

template <typename State>
State* CommandStateManager::GetOrCreate(
    const Command* cmd, const stream_executor::CommandBuffer* command_buffer,
    absl::FunctionRef<std::unique_ptr<State>()> create) {
  static_assert(std::is_base_of_v<CommandState, State>);
  return static_cast<State*>(GetOrCreate(
      cmd, command_buffer, GetTypeId<State>(), [&] { return create(); }));
}

template <typename State>
State* CommandStateManager::GetOrCreate(
    const Command* cmd, const stream_executor::CommandBuffer* command_buffer) {
  return GetOrCreate<State>(cmd, command_buffer,
                            [] { return std::make_unique<State>(); });
}

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_STATE_H_
