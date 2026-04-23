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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_H_

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/platform.h"
#include "xla/xla.pb.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CommandType
//===----------------------------------------------------------------------===//

// clang-format off
#define XLA_GPU_COMMAND_LIST(V)                              \
  V(kTracedCommand, "TracedCommand")       \
  V(kComputationIdCmd, "ComputationIdCmd")                   \
  V(kLaunchCmd, "LaunchCmd")                                 \
  V(kCustomKernelLaunchCmd, "CustomKernelLaunchCmd")         \
  V(kCublasLtCmd, "CublasLtCmd")                             \
  V(kCuDnnCmd, "CuDnnCmd")                                   \
  V(kGemmCmd, "GemmCmd")                                     \
  V(kMemcpyDeviceToDeviceCmd, "MemcpyDeviceToDeviceCmd")     \
  V(kMemzeroCmd, "MemzeroCmd")                               \
  V(kMemset32Cmd, "Memset32Cmd")                             \
  V(kCaseCmd, "CaseCmd")                                     \
  V(kWhileCmd, "WhileCmd")                                   \
  V(kCustomCallCmd, "CustomCallCmd")                         \
  V(kBarrierCmd, "BarrierCmd")                               \
  V(kCollectiveCmd, "CollectiveCmd")                         \
  V(kAllReduceCmd, "AllReduceCmd")                           \
  V(kReduceScatterCmd, "ReduceScatterCmd")                   \
  V(kAllToAllCmd, "AllToAllCmd")                             \
  V(kAllGatherCmd, "AllGatherCmd")                           \
  V(kCollectiveBroadcastCmd, "CollectiveBroadcastCmd")       \
  V(kCollectivePermuteCmd, "CollectivePermuteCmd")           \
  V(kRaggedAllToAllCmd, "RaggedAllToAllCmd")                 \
  V(kRecvCmd, "RecvCmd")                                     \
  V(kSendCmd, "SendCmd")                                     \
  V(kAsyncDone, "AsyncDone")                                 \
  V(kUnknownCmd, "UnknownCmd") \
  // clang-format on

enum class CommandType : int32_t {
#define DECLARE_ENUM(enum_name, cmd_name, ...) enum_name,
  XLA_GPU_COMMAND_LIST(DECLARE_ENUM)
#undef DECLARE_ENUM
};

std::string CommandTypeString(CommandType type);

template <typename Sink>
void AbslStringify(Sink& sink, CommandType type) {
  sink.Append(CommandTypeString(type));
}

// Returns true if command type corresponds to a collective operation.
bool IsCollectiveCommand(CommandType type);

//===----------------------------------------------------------------------===//
// Command
//===----------------------------------------------------------------------===//

// Command is a Thunk counterpart that instead of launching operations directly
// on the underlying device records them into command buffers.
//
// Commands have the same execution stages as thunks as they are executed by a
// command buffer thunk: Prepare, Initialize and Record (Execute). See Thunk
// documentation for details.
//
// Commands must be thread safe as they can be recorded into multiple command
// buffers concurrently on different stream executors.
//
// IMPORTANT: In contrast to GPU thunks, commands MUST be stateless. Thunk state
// typically belongs to the Thunk instance itself, and tends to be kept in
// synchronized hash maps keyed by `se::StreamExecutor*` pointer.
// Commands on the other hand should attach state to the underlying command
// buffer, and because the number of command buffers that can be instantiated
// from a command sequence is unbounded (as we have an eviction policy for
// command buffers), keeping a state in a map inside the command will lead to
// memory leaks.
//
// Commands have an external state manager, which is responsible for managing
// the lifetime of command state. See `CommandState` and
// `CommandCommandStateManager` documentation for details and example. If
// command want's to attach some mutable state to the command buffer, it must be
// done with a state manager.
class Command : public Thunk {
 public:
  explicit Command(CommandType cmd_type,
                   se::StreamPriority priority = se::StreamPriority::Default)
      : Thunk(Thunk::Kind::kCommand, ThunkInfo{}),
        cmd_type_(cmd_type),
        priority_(priority) {
    token_ = Resource::Create(Resource::kToken);
  }

  virtual ~Command() = default;

 protected:
  // Constructor for Thunk subclasses that are also Commands. Preserves the
  // caller's Thunk::Kind and ThunkInfo instead of using Kind::kCommand and an
  // empty ThunkInfo.
  Command(CommandType cmd_type, Thunk::Kind thunk_kind, ThunkInfo thunk_info,
          se::StreamPriority priority = se::StreamPriority::Default)
      : Thunk(thunk_kind, std::move(thunk_info)),
        cmd_type_(cmd_type),
        priority_(priority) {
    token_ = Resource::Create(Resource::kToken);
  }

 public:
  // Parameters for recording commands into the command buffer.
  struct RecordParams {
    // An external state manager that gives efficient access to per-device state
    // to commands without a need to add expensive synchronization.
    CommandStateManager& state;

    // Buffer allocations that changed since the last call to `Record`. Buffer
    // allocation indices are sorted. CommandExecutor and individual commands
    // rely on this information to skip unnecessary updates.
    std::optional<std::vector<BufferAllocation::Index>> updated_allocs;

    // A flag indicating whether we record commands at command buffer thunk
    // initialization time.
    bool is_initialization = false;

    // The CommandBufferUpdateMode for the enclosing command buffer thunk.
    DebugOptions::CommandBufferUpdateMode command_buffer_update_mode =
        DebugOptions::ALWAYS_UPDATE;
  };

  // Create new commands in the command buffer using the given dependencies.
  struct RecordCreate {
    absl::Span<const se::CommandBuffer::Command* const> dependencies;
  };

  // Update previously recorded commands in the command buffer.
  struct RecordUpdate {
    const se::CommandBuffer::Command* command;
  };

  // When recording a command into the command buffer we can either update
  // previously recorded commands or create new ones. The command DAG structure
  // can be defined only when we record commands the first time, after that we
  // can only update previously recorded commands parameters (i.e. with pointers
  // to new buffer allocations).
  using RecordAction = std::variant<RecordCreate, RecordUpdate>;

  // Commands are not executed directly as Thunks; they are recorded into
  // command buffers via Record(). ExecuteOnStream is not supported.
  absl::Status ExecuteOnStream(const ExecuteParams& params) override {
    return absl::UnimplementedError(
        "Command cannot be executed directly as a Thunk; use Record() instead");
  }

  // Records commands into the command buffer. Returned commands will be passed
  // back on the next call to `Record` into the same command buffer, so that it
  // can do efficient command buffer updates.
  virtual absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) {
    return absl::UnimplementedError("Record is not implemented");
  }

  // Returns true if command requires initialization (has to be recorded at
  // command buffer thunk initialization).
  //
  // Today this is only true for collective commands that might use NCCL for
  // communication. With NCCL, all participating ranks must record collective
  // commands at the same time, if some ranks will skip command updates (because
  // they got lucky and got the same buffer allocations), it will lead to
  // deadlocks. By forcing the command update at thunk initialization time, we
  // ensure that all ranks execute NCCL command update.
  virtual bool requires_initialization() const { return false; }

  // Returns true if this command is implemented via CUDA stream activity
  // tracing (i.e. a subclass of TracedCommand).
  virtual bool IsTracedCommand() const { return false; }

  // Returns true if command supports loop unroll, the while loop can be
  // unrolled only if it has pre-known trip count and also all commands from the
  // body commands are unrollable.
  virtual bool support_loop_unroll() const { return true; }

  std::shared_ptr<Resource> token() const { return token_; }

  CommandType command_type() const { return cmd_type_; }
  se::StreamPriority priority() const { return priority_; }
  void set_priority(se::StreamPriority priority) { priority_ = priority; }

  std::string ToString(int indent) const override {
    return CommandTypeString(cmd_type_);
  }

  // Recursively walks all the commands nested inside *this one and calls
  // the user-provided callback on every command. Always starts traversal with
  // *this. These overloads accept Command*-typed callbacks and complement the
  // Thunk*-typed Walk overloads inherited from Thunk.
  template <typename F, WalkCallback<F, Command*>* = nullptr>
  std::invoke_result_t<F, Command*> Walk(F&& callback);
  template <typename F, WalkCallback<F, const Command*>* = nullptr>
  std::invoke_result_t<F, const Command*> Walk(F&& callback) const;

 protected:
  // WalkNested uses Thunk::Walker = absl::FunctionRef<absl::Status(Thunk*)>.
  // Subclasses that have nested commands must override this.
  absl::Status WalkNested(Walker callback) override { return absl::OkStatus(); }

 private:
  CommandType cmd_type_;

  // The token resource is used to specify additional dependency across
  // commands, like control dependency across HLO operators, and LHS scheduling
  // dependency.
  std::shared_ptr<Resource> token_;

  // Command priority, currently only support default, lowest and highest
  // priority.
  se::StreamPriority priority_ = se::StreamPriority::Default;
};

// Returns true if command is a collective one.
inline bool IsCollectiveCommand(const Command& cmd) {
  return IsCollectiveCommand(cmd.command_type());
}

//===----------------------------------------------------------------------===//
// Command templates implementation.
//===----------------------------------------------------------------------===//

template <typename F, Command::WalkCallback<F, Command*>*>
std::invoke_result_t<F, Command*> Command::Walk(F&& callback) {
  if constexpr (std::is_void_v<std::invoke_result_t<F, Command*>>) {
    Walk([f = std::forward<F>(callback)](Command* command) {
      return (f(command), absl::OkStatus());
    }).IgnoreError();  // Error can never happen here.
  } else {
    RETURN_IF_ERROR(callback(this));
    // Adapt Command*-typed callback to Thunk::Walker (Thunk*-typed) for
    // WalkNested. The down_cast is safe because WalkNested only visits
    // Commands in a Command context.
    return WalkNested([&callback](Thunk* thunk) -> absl::Status {
      return callback(tsl::down_cast<Command*>(thunk));
    });
  }
}

template <typename F, Command::WalkCallback<F, const Command*>*>
std::invoke_result_t<F, const Command*> Command::Walk(F&& callback) const {
  return const_cast<Command*>(this)->Walk(  // NOLINT
      std::forward<F>(callback));
}

//===----------------------------------------------------------------------===//
// Asynchronous commands
//===----------------------------------------------------------------------===//

// A base class for a command that starts an asynchronous execution.
class AsyncStartCommand : public Command {
 public:
  using Command::Command;

  // At run time async command might behave like a synchronous one, i.e.
  // some collective operations if they can't be overlapped with compute
  // operations executed like they have synchronous execution semantics.
  virtual bool IsAsync() const = 0;
};

// A command that completes an `async_start` command.
class AsyncDoneCommand : public Command {
 public:
  explicit AsyncDoneCommand(const AsyncStartCommand* async_start)
      : Command(CommandType::kAsyncDone), async_start_(async_start) {
    DCHECK(async_start_) << "AsyncStart command must be not null";
  }

  const AsyncStartCommand* async_start() const { return async_start_; }
  bool IsAsync() const { return async_start_->IsAsync(); }

 private:
  const AsyncStartCommand* async_start_;
};

//===----------------------------------------------------------------------===//
// CommandSequence
//===----------------------------------------------------------------------===//

// A sequence of commands (corresponds to a ThunkSequence from the Thunk API).
//
// Commands are stored as raw pointers in append order. Ownership is split:
// - Commands created during conversion (i.e. Command subclasses in
//   command_buffer_cmd.h that are not yet migrated to their corresponding
//   Thunk) are owned by this sequence's internal `owned_` vector.
// - Commands that live in a Thunk which itself implements Command (e.g.
//   ReplicaOrPartitionIdThunk) are borrowed: only a raw pointer is stored here,
//   and the ThunkSequence in CommandBufferThunk retains ownership.
class CommandSequence : public std::vector<Command*> {
 public:
  // Appends an owned command. Ownership is transferred to this sequence.
  void Append(std::unique_ptr<Command> command) {
    this->push_back(command.get());
    owned_.push_back(std::move(command));
  }

  // Appends a borrowed command. The caller must ensure the command outlives
  // this sequence (e.g. it is a Thunk that also implements Command, owned by
  // the ThunkSequence in CommandBufferThunk).
  void Append(Command* command) { this->push_back(command); }

  // Constructs a new command in place and transfers ownership to this sequence.
  template <typename Cmd, typename... Args>
  void Emplace(Args&&... args) {
    Append(std::make_unique<Cmd>(std::forward<Args>(args)...));
  }

  std::string ToString() const {
    std::string result;
    for (Command* cmd : *this) {
      result += cmd->ToString(0) + "\n";
    }
    return result;
  }

  absl::Status Walk(
      absl::FunctionRef<absl::Status(const Command*)> callback) const {
    for (Command* cmd : *this) {
      RETURN_IF_ERROR(cmd->Walk(callback));
    }
    return absl::OkStatus();
  }

  absl::Status Walk(absl::FunctionRef<absl::Status(Command*)> callback) {
    for (Command* cmd : *this) {
      RETURN_IF_ERROR(cmd->Walk(callback));
    }
    return absl::OkStatus();
  }

  void Walk(absl::FunctionRef<void(const Command*)> callback) const {
    for (Command* cmd : *this) {
      cmd->Walk(callback);
    }
  }

  void Walk(absl::FunctionRef<void(Command*)> callback) {
    for (Command* cmd : *this) {
      cmd->Walk(callback);
    }
  }

 private:
  // Owns commands that were constructed during conversion (i.e. not backed by
  // a Thunk). Borrowed commands (Thunks implementing Command) are not here.
  std::vector<std::unique_ptr<Command>> owned_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_H_
