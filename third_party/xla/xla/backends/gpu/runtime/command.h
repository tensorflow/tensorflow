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

#include <string>

#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/stream_executor/command_buffer.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// CommandType
//===----------------------------------------------------------------------===//

// clang-format off
#define XLA_GPU_COMMAND_LIST(V)                              \
  V(kEmptyCmd, "EmptyCmd")                                   \
  V(kChildCmd, "ChildCmd")                                   \
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
  V(kRecvCmd, "RecvCmd")                                     \
  V(kSendCmd, "SendCmd")                                     \
  V(kAsyncDone, "AsyncDone")                                 \
  V(kDynamicSliceFusionCmd, "DynamicSliceFusionCmd")         \
  V(kDynamicSliceCopyFusionCmd, "DynamicSliceCopyFusionCmd") \
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
class Command {
 public:
  using BufferUseVector = absl::InlinedVector<BufferUse, 4>;
  using ResourceUseVector = absl::InlinedVector<ResourceUse, 1>;

 public:
  explicit Command(CommandType cmd_type,
                   se::StreamPriority priority = se::StreamPriority::Default)
      : cmd_type_(cmd_type), priority_(priority) {
    token_ = Resource::Create(Resource::kToken);
    resources_.push_back(ResourceUse::Write(token_));
  }

  virtual ~Command() = default;

  // Parameters for recording commands into the command buffer.
  struct RecordParams {
    // An external state manager that gives efficient access to per-device state
    // to commands without a need to add expensive synchronization.
    CommandStateManager& state;

    // Buffer allocations that changed since the last call to `Record`. Buffer
    // allocation indices are sorted. CommandExecutor and individual commands
    // rely on this information to skip unnecessary updates.
    std::optional<std::vector<BufferAllocation::Index>> updated_allocs;

    // A flag indicating whether we record comands at command buffer thunk
    // initialization time.
    bool is_initialization = false;

    // The command sequence might be recorded in the loop unrolling pattern, so
    // the command sequence might be instantiated multiple times, we uses
    // unroll_iteration to locate the commands for current unroll iteration.
    int64_t unroll_iteration = 0;
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

  // See Thunk documentation for XLA execution stages (prepare, initialize,
  // execute). Commands mirror thunks as they are executed as CommandBufferThunk
  // that is plugged into the Thunk execution cycle.

  // Prepare command for execution by allowing command to request shared state
  // required for recording (i.e. collective commands request cliques).
  virtual absl::Status Prepare(const Thunk::PrepareParams& params) {
    return absl::OkStatus();
  }

  // Initialize a command for recording on a given executor. We split it into a
  // separate function to allow expensive initialization (e.g. device kernel
  // loading) to happen before a command buffer thunk execution.
  virtual absl::Status Initialize(const Thunk::InitializeParams& params,
                                  CommandStateManager& state) {
    return absl::OkStatus();
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
  virtual bool requires_initialization() { return false; }

  // Returns true if command supports loop unroll, the while loop can be
  // unrolled only if it has pre-known trip count and also all commands from the
  // body commands are unrollable..
  virtual bool support_loop_unroll() { return true; }

  // This is only true for DynamicSliceCopyFusionCmd when offset is dependents
  // on loop iteration. As the command of slice operation is access the sliced
  // memory region that varies across loop iterations, so even the original
  // buffer allocation is the same, it still requires to do update.
  virtual bool force_update() { return false; }

  // Returns all buffers used by the cmd. These will be used to track cmd
  // updates, thus they need to be consistent across calls to the function.
  virtual BufferUseVector buffers() const { return {}; }

  std::shared_ptr<Resource> token() const { return token_; }

  void add_resource_use(ResourceUse resource_use) {
    resources_.push_back(resource_use);
  }
  ResourceUseVector resources() const { return resources_; }

  // Returns true if command implemented as a nested command buffer.
  virtual bool IsNestedCommandBuffer() const { return false; }

  absl::string_view profile_annotation() const { return profile_annotation_; }
  void set_profile_annotation(absl::string_view profile_annotation) {
    profile_annotation_ = profile_annotation;
  }

  CommandType command_type() const { return cmd_type_; }
  se::StreamPriority priority() const { return priority_; }
  void set_priority(se::StreamPriority priority) { priority_ = priority; }

  virtual std::string ToString() const { return CommandTypeString(cmd_type_); }

 private:
  std::string profile_annotation_;
  CommandType cmd_type_;

  ResourceUseVector resources_;

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
// Asynchronous commands
//===----------------------------------------------------------------------===//

// A base class for a command that starts an asyncrhonous execution.
class AsyncStartCommand : public Command {
 public:
  using Command::Command;

  // At run time async command might behave like a syncrhonous one, i.e.
  // some collective operations if they can't be overlapped with compute
  // operations executed like they have syncrhonous execution semantics.
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
class CommandSequence : public std::vector<std::unique_ptr<Command>> {
 public:
  template <typename Command, typename... Args>
  void Emplace(Args&&... args) {
    this->emplace_back(std::make_unique<Command>(std::forward<Args>(args)...));
  }

  std::string ToString() const {
    std::string result;
    for (const auto& cmd : *this) {
      result += cmd->ToString() + "\n";
    }
    return result;
  }
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_H_
