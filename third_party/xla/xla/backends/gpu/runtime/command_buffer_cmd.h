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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/object_pool.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

class CommandBufferCmdExecutor;  // Forward declaration.

// clang-format off
#define COMMAND_BUFFER_CMD_LIST(V)                               \
  V(kEmptyCmd, "EmptyCmd")                                       \
  V(kChildCmd, "ChildCmd")                                       \
  V(kTracedCommandBufferCmd, "TracedCommandBufferCmd")           \
  V(kComputationIdCmd, "ComputationIdCmd")                       \
  V(kLaunchCmd, "LaunchCmd")                                     \
  V(kCustomKernelLaunchCmd, "CustomKernelLaunchCmd")             \
  V(kCublasLtCmd, "CublasLtCmd")                                 \
  V(kCuDnnCmd, "CuDnnCmd")                                       \
  V(kGemmCmd, "GemmCmd")                                         \
  V(kMemcpyDeviceToDeviceCmd, "MemcpyDeviceToDeviceCmd")         \
  V(kMemzeroCmd, "MemzeroCmd")                                   \
  V(kMemset32Cmd, "Memset32Cmd")                                 \
  V(kCaseCmd, "CaseCmd")                                         \
  V(kWhileCmd, "WhileCmd")                                       \
  V(kCustomCallCmd, "CustomCallCmd")                             \
  V(kBarrierCmd, "BarrierCmd")                                   \
  V(kCollectiveCmd, "CollectiveCmd")                             \
  V(kAllReduceCmd, "AllReduceCmd")                               \
  V(kReduceScatterCmd, "ReduceScatterCmd")                       \
  V(kAllToAllCmd, "AllToAllCmd")                                 \
  V(kAllGatherCmd, "AllGatherCmd")                               \
  V(kCollectiveBroadcastCmd, "CollectiveBroadcastCmd")           \
  V(kCollectivePermuteCmd, "CollectivePermuteCmd")               \
  V(kAsyncDone, "AsyncDone")                                     \
  V(kDynamicSliceFusionCmd, "DynamicSliceFusionCmd")             \
  V(kDynamicSliceCopyFusionCmd, "DynamicSliceCopyFusionCmd")     \
  V(kUnknownCmd, "UnknownCmd") \
  // clang-format on

enum class CommandBufferCmdType : int32_t {
#define DECLARE_ENUM(enum_name, cmd_name, ...) enum_name,
  COMMAND_BUFFER_CMD_LIST(DECLARE_ENUM)
#undef DECLARE_ENUM
};

std::string CommandBufferCmdString(CommandBufferCmdType type);

//===----------------------------------------------------------------------===//
// CommandBufferCmd
//===----------------------------------------------------------------------===//

using ResourceUseVector = absl::InlinedVector<ResourceUse, 1>;

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
// synchronized hash maps keyed by `se::StreamExecutor*` pointer. Commands on
// the other hand should attach state to the underlying command buffer, and
// because the number of command buffers that can be instantiated from a command
// sequence is unbounded (as we have an eviction policy for command buffers),
// keeping a state in a map inside the command will lead to memory leaks.
//
// Commands have an external state manager, which is responsible for managing
// the lifetime of command state. See `State` and `StateManager` classes below.
//
// To make command stateful, it needs a `params.state` indirection:
//
//   class MyCommand : public CommandBufferCmd {
//     public:
//
//     // Container for mutable state required for command execution.
//     struct MyState : CommandBufferCmd::State {
//       ...
//     };
//
//     absl::StatusOr<Command*> Record(...) override {
//       // Attach a new instance of `MyState` to the `command_buffer`. When
//       // command buffer will be destroyed, the state will be destroyed as
//       // well automatically by XLA runtime. If this command will be recorded
//       // into another command buffer, the state will be re-created
//       // automatically using the provided callback.
//       MyState* my_state = record_params.state.GetOrCreate<MyState>(this,
//         command_buffer, [&] { // create MyState for a `command_buffer` });
//       ...
//     }
//
//   };
//
class CommandBufferCmd {
 public:
  explicit CommandBufferCmd(
      CommandBufferCmdType cmd_type,
      se::StreamPriority priority = se::StreamPriority::Default)
      : cmd_type_(cmd_type), priority_(priority) {
    token_ = Resource::Create(Resource::kToken);
    resources_.push_back(ResourceUse::Write(token_));
  }

  virtual ~CommandBufferCmd() = default;

  using BufferUseVector = absl::InlinedVector<BufferUse, 4>;

  // A base class for externally managed command state.
  //
  // Commands can be executed concurrently for many stream executors (underlying
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
  // See example above next to `CommandBufferCmd` definition.
  class State {
   public:
    virtual ~State() = default;
  };

  // An external manager for a state attached to commands recorded into command
  // buffers (same command can be recorded into multiple command buffers).
  class StateManager {
   public:
    virtual ~StateManager() = default;

    template <typename ConcreteState>
    ConcreteState* absl_nullable GetOrNull(
        const CommandBufferCmd* absl_nonnull cmd,
        const se::CommandBuffer* absl_nonnull command_buffer,
        int64_t unroll_iteration = 0) {
      static_assert(std::is_base_of_v<State, ConcreteState>);
      return static_cast<ConcreteState*>(GetOrNull(
          cmd, command_buffer, GetTypeId<ConcreteState>(), unroll_iteration));
    }

    template <typename ConcreteState>
    ConcreteState* absl_nonnull GetOrCreate(
        const CommandBufferCmd* absl_nonnull cmd,
        const se::CommandBuffer* absl_nonnull command_buffer,
        absl::FunctionRef<std::unique_ptr<ConcreteState>()> create,
        int64_t unroll_iteration = 0) {
      static_assert(std::is_base_of_v<State, ConcreteState>);
      return static_cast<ConcreteState*>(
          GetOrCreate(cmd, command_buffer, GetTypeId<ConcreteState>(),
                      unroll_iteration, [&] { return create(); }));
    }

    template <typename ConcreteState>
    ConcreteState* absl_nonnull GetOrCreate(
        const CommandBufferCmd* absl_nonnull cmd,
        const se::CommandBuffer* absl_nonnull command_buffer,
        int64_t unroll_iteration = 0) {
      return GetOrCreate<ConcreteState>(
          cmd, command_buffer, [] { return std::make_unique<ConcreteState>(); },
          unroll_iteration);
    }

   private:
    // We use TypeId to distinguish between different state types.
    TSL_LIB_GTL_DEFINE_INT_TYPE(TypeId, int64_t);

    template <typename F>
    static TypeId GetTypeId() {
      static const TypeId id = GetNextTypeId();
      return id;
    }

    static TypeId GetNextTypeId();

    State* absl_nullable GetOrNull(
        const CommandBufferCmd* absl_nonnull cmd,
        const se::CommandBuffer* absl_nonnull command_buffer, TypeId type_id,
        int64_t unroll_iteration);

    State* absl_nonnull GetOrCreate(
        const CommandBufferCmd* absl_nonnull cmd,
        const se::CommandBuffer* absl_nonnull command_buffer, TypeId type_id,
        int64_t unroll_iteration,
        absl::FunctionRef<std::unique_ptr<State>()> create);

    using Key = std::tuple<const CommandBufferCmd*, const se::CommandBuffer*,
                           TypeId, int64_t>;
    absl::flat_hash_map<Key, std::unique_ptr<State>> state_;
  };

  // Parameters for recording commands into the command buffer.
  struct RecordParams {
    // An external state manager that gives efficient access to per-device state
    // to commands without a need to add expensive synchronization.
    StateManager& state;

    // Buffer allocations that changed since the last call to `Record`. Buffer
    // allocation indices are sorted. CommandBufferCmdExecutor and individual
    // commands rely on this information to skip unnecessary updates.
    std::optional<std::vector<BufferAllocation::Index>> updated_allocs;

    // A flag indicating whether we record comands at command buffer thunk
    // initialization time.
    bool is_initialization = false;

    // The command sequence might be recorded in the loop unrolling pattern, so
    // the command sequence might be instantiated multiple times, we uses
    // unroll_iteration to locate the commands for current unroll iteration.
    int64_t unroll_iteration = 0;

    // The command buffer that is recording the commands. Must be set before
    // calling Record().
    se::CommandBuffer* absl_nonnull command_buffer = nullptr;

    // A flag indicating whether we finalize the command buffer after recording.
    bool is_finalize = true;

    // The executor that is recording the commands. Used for dependency
    // resolution within the executor. May be null when recording commands
    // outside of an executor context.
    const CommandBufferCmdExecutor* absl_nullable executor = nullptr;

    // External dependencies that source commands in this executor must wait on.
    // When multiple CommandBufferCmdExecutors are recorded into the same
    // command buffer (e.g., A -> B -> C), the source commands of executor B
    // must depend on the sink commands of executor A. These external
    // dependencies are passed in via this field to establish cross-executor
    // ordering.
    //
    // Example: WhileCmd loop unrolling. When a WhileCmd has a known trip count,
    // we unroll the loop by recording cond_commands and body_commands executors
    // multiple times into a single command buffer:
    //   [cond_0] -> [body_0] -> [cond_1] -> [body_1] -> ... -> [cond_N] ->
    //   [body_N]
    // Each iteration's body_commands must wait on the preceding cond_commands,
    // and each cond_commands (except the first) must wait on the preceding
    // body_commands. The external_dependencies field carries these
    // cross-executor dependencies between unrolled iterations.
    std::vector<const se::CommandBuffer::Command*> external_dependencies;
  };

  using CreateCommand =
      absl::FunctionRef<absl::StatusOr<const se::CommandBuffer::Command*>()>;

  using UpdateCommand = absl::FunctionRef<absl::Status(
      const se::CommandBuffer::Command* command)>;

  absl::Status HandleCmdCreateOrUpdate(RecordParams& record_params,
                                       CreateCommand create_command,
                                       UpdateCommand update_command);

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
                                  StateManager& state) {
    return absl::OkStatus();
  }

  // Records commands into the command buffer. Returned commands will be passed
  // back on the next call to `Record` into the same command buffer, so that it
  // can do efficient command buffer updates.
  virtual absl::Status Record(const Thunk::ExecuteParams& execute_params,
                              RecordParams& record_params) {
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

  void add_resouce_use(ResourceUse resource_use) {
    resources_.push_back(resource_use);
  }
  ResourceUseVector resources() const { return resources_; }

  // Returns true if command implemented as a nested command buffer.
  virtual bool IsNestedCommandBuffer() const { return false; }

  absl::string_view profile_annotation() const { return profile_annotation_; }
  void set_profile_annotation(absl::string_view profile_annotation) {
    profile_annotation_ = profile_annotation;
  }

  CommandBufferCmdType command_type() const { return cmd_type_; }
  se::StreamPriority priority() const { return priority_; }
  void set_priority(se::StreamPriority priority) { priority_ = priority; }

  virtual std::string ToString() const {
    return CommandBufferCmdString(cmd_type_);
  }

  // Return the dependencies of the command from within the executor, if the
  // command is a source command, it will return the executor dependencies
  // specified in record_params.
  std::vector<const se::CommandBuffer::Command*> Dependencies(
      const RecordParams& record_params) const;

 private:
  std::string profile_annotation_;
  CommandBufferCmdType cmd_type_;

  ResourceUseVector resources_;

  // The token resource is used to specify additional dependency across
  // commands, like control dependency across HLO operators, and LHS scheduling
  // dependency.
  std::shared_ptr<Resource> token_;

  // Command priority, currently only support default, lowest and highest
  // priority.
  se::StreamPriority priority_ = se::StreamPriority::Default;
};

// A sequence of commands (corresponds to a ThunkSequence from the Thunk API).
class CommandBufferCmdSequence
    : public std::vector<std::unique_ptr<CommandBufferCmd>> {
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

//===----------------------------------------------------------------------===//
// CommandBufferCmdExecutor
//===----------------------------------------------------------------------===//

// Command executor is responsible for recording commands sequence into the
// underlying command buffer and setting up dependencies between commands.
class CommandBufferCmdExecutor {
 public:
  CommandBufferCmdExecutor() = default;
  CommandBufferCmdExecutor(CommandBufferCmdExecutor&&) = default;
  CommandBufferCmdExecutor& operator=(CommandBufferCmdExecutor&&) = default;

  using RecordParams = CommandBufferCmd::RecordParams;

  // Synchronization mode defines how much concurrency is allowed between
  // commands in the sequence.
  enum class SynchronizationMode {
    // Serializes execution of all commands recorded into the command buffer
    // by adding a dependency between them.
    kSerialize,

    // Relies on execution graph to insert dependencies between commands
    // that have buffer of resource conflicts, and building a DAG of commands.
    kConcurrent,

    // Uses the same latency hidden scheduling results used in the thunk
    // scheduling.
    kLHS,
  };

  template <typename Sink>
  friend void AbslStringify(Sink& sink, SynchronizationMode mode) {
    switch (mode) {
      case SynchronizationMode::kSerialize:
        sink.Append("serialize");
        break;
      case SynchronizationMode::kConcurrent:
        sink.Append("concurrent");
        break;
      case SynchronizationMode::kLHS:
        sink.Append("lhs");
        break;
    }
  }

  // Creates a command executor from a sequence of commands using given
  // synchronization mode.
  static absl::StatusOr<CommandBufferCmdExecutor> Create(
      CommandBufferCmdSequence commands,
      SynchronizationMode synchronization_mode);

  // Prepares all commands added to a sequence.
  absl::Status Prepare(const Thunk::PrepareParams& params);

  // Initializes all commands added to a sequence.
  absl::Status Initialize(const Thunk::InitializeParams& params,
                          CommandBufferCmd::StateManager& state);

  // Records commands into the command buffer. This method automatically
  // switches between `RecordCreate` or `RecordUpdate` depending on the command
  // buffer state.

  // This Record function allows multiple CommandbufferCmdEXecutor to be
  // recorded into a single command buffer. e.g. we can have Executor A, B, C to
  // be recorded into the same command buffer in the order of A -> B -> C. In
  // this pattern, B's source commands will depend on A's sink commands, and C's
  // source commands will also depend on B's sink commands.

  // If record_action is `RecordCreate`, it will set up initial
  // dependencies for recorded commands by the `dependencies` parameter.
  // If record_action is `RecordUpdate`, it will only update previously
  // recorded commands' dependencies, no other actions.

  // Records commands into the command buffer. Uses record_params.is_finalize
  // to determine whether to finalize the command buffer after recording.
  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) const;

  // Returns buffers referenced by commands in this sequence.
  const absl::flat_hash_set<BufferUse>& buffers() const;

  // Returns buffer allocations indices referenced by commands in this sequence.
  absl::Span<const BufferAllocation::Index> allocs_indices() const;

  bool empty() const { return commands_.empty(); }
  size_t size() const { return commands_.size(); }

  bool requires_initialization() const {
    return absl::c_any_of(commands_, [](const auto& cmd) {
      return cmd->requires_initialization();
    });
  }

  bool force_update() const {
    return absl::c_any_of(commands_,
                          [](const auto& cmd) { return cmd->force_update(); });
  }

  bool support_loop_unroll() const {
    return absl::c_all_of(
        commands_, [](const auto& cmd) { return cmd->support_loop_unroll(); });
  }

  // Returns all commands associated with the given ids on a certain unroll
  // iteration.
  std::vector<const se::CommandBuffer::Command*> SourceCommands(
      const RecordParams& record_params) const;

  std::vector<const se::CommandBuffer::Command*> SinkCommands(
      const RecordParams& record_params) const;

  // Renders the execution graph using default renderer. Returns url of the
  // rendered graph, or an error if rendering failed.
  absl::StatusOr<std::string> RenderExecutionGraph();

  // Returns true if the given command is a source command (has no dependencies
  // within this executor).
  bool IsSource(const CommandBufferCmd* absl_nonnull cmd) const;

  // Returns dependencies of the given command.
  std::vector<const se::CommandBuffer::Command*> Dependencies(
      const RecordParams& record_params,
      const CommandBufferCmd* absl_nonnull cmd) const;

  using CreateCommand =
      absl::FunctionRef<absl::StatusOr<const se::CommandBuffer::Command*>()>;

  using UpdateCommand = absl::FunctionRef<absl::Status(
      const se::CommandBuffer::Command* absl_nonnull command)>;

  absl::Status HandleCmdCreateOrUpdate(RecordParams& record_params,
                                       CommandBufferCmd* absl_nonnull cmd,
                                       CreateCommand create_command,
                                       UpdateCommand update_command) const;

 private:
  // We use index into the `commands_` vector as a command id.
  using CommandId = int64_t;

  // A state associated with commands in the sequence. We rely on this state to
  // efficiently update command recorded into the command buffer.
  struct RecordState : public CommandBufferCmd::State {
    const se::CommandBuffer::Command* command;
  };

  CommandBufferCmdExecutor(SynchronizationMode synchronization_mode,
                           CommandBufferCmdSequence commands,
                           std::optional<ExecutionGraph> execution_graph);

  absl::Status CheckCommandBufferState(
      se::CommandBuffer* absl_nonnull command_buffer,
      se::CommandBuffer::State expected_state) const;

  // Returns true if command has no dependencies.
  bool IsSource(CommandId id) const;

  // Returns true if command is not a dependency of any other commands.
  bool IsSink(CommandId id) const;

  // Returns dependencies of the command with the given id.
  std::vector<const se::CommandBuffer::Command*> Dependencies(
      const RecordParams& record_params, CommandId id) const;

  SynchronizationMode synchronization_mode_;
  CommandBufferCmdSequence commands_;

  // In automatic synchronization mode we build an execution graph for the
  // sequence of commands and use it to set up dependencies between commands.
  std::optional<ExecutionGraph> execution_graph_;

  // Buffers referenced by commands in this sequence.
  absl::flat_hash_set<BufferUse> buffers_;

  // Unique buffer allocations indices referenced by all commands in this
  // sequence (sorted by the buffer allocation index).
  std::vector<BufferAllocation::Index> allocs_indices_;

  // A mapping from command id to unique buffer allocations indices referenced
  // by the command (sorted by the buffer allocation index).
  std::vector<std::vector<BufferAllocation::Index>> cmd_allocs_indices_;
};

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

// A cache for traced command buffers that will re-trace on change in buffer
// allocations that are relevant for `buffers` passed to constructor. We use a
// very simple most-recently-used cache of traced command buffers as in practice
// subsequent calls to XLA executable tend to reuse the same allocations.
class TracedCommandBuffer : public CommandBufferCmd::State {
 public:
  explicit TracedCommandBuffer(const CommandBufferCmd* absl_nonnull trace_cmd,
                               CommandBufferCmd::BufferUseVector buffers,
                               int64_t capacity = 16);

  // Returns cached command buffer traced using the same buffer addresses or
  // traces and caches a new command buffer using user provided callback.
  absl::StatusOr<se::CommandBuffer* absl_nonnull> GetOrTraceCommandBuffer(
      const BufferAllocations* absl_nonnull buffer_allocation,
      se::StreamExecutor* absl_nonnull executor,
      se::Stream* absl_nonnull stream,
      absl::FunctionRef<absl::Status(se::Stream*)> trace,
      se::StreamPriority priority = se::StreamPriority::Default);

 private:
  std::vector<BufferAllocation::Index> allocs_indices_;

  struct Entry {
    std::vector<se::DeviceAddressBase> recorded_allocs;
    std::unique_ptr<se::CommandBuffer> command_buffer;
  };
  const CommandBufferCmd* absl_nonnull trace_cmd_;
  int64_t capacity_;
  std::vector<Entry> entries_;
};

//===----------------------------------------------------------------------===//
// TracedCommandBufferCmd
//===----------------------------------------------------------------------===//

// A base class for commands implemented as tracing of stream activities.
class TracedCommandBufferCmd : public CommandBufferCmd {
 protected:
  explicit TracedCommandBufferCmd(CommandBufferCmdType cmd_type);

  // Creates a command buffer by calling a user-provided `trace` function and
  // adds it as a nested command to `command_buffer`. Traced command buffers
  // cached and reused in an instance of `TracedCommandBuffer` kept in `state`.
  absl::Status RecordTracedCommand(
      const Thunk::ExecuteParams& execute_params, RecordParams& record_params,
      absl::FunctionRef<absl::Status(se::Stream*)> trace);
};

//===----------------------------------------------------------------------===//
// EmptyCmd
//===----------------------------------------------------------------------===//

class EmptyCmd : public CommandBufferCmd {
 public:
  explicit EmptyCmd();

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override { return {}; }
};

//===----------------------------------------------------------------------===//
// AsyncDoneCmd
//===----------------------------------------------------------------------===//

class AsyncDoneCmd : public CommandBufferCmd {
 public:
  explicit AsyncDoneCmd(
      std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override { return {}; }

  bool IsAsync() const { return async_events_ != nullptr; }
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events() const {
    return async_events_;
  }

 private:
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events_;
};

//===----------------------------------------------------------------------===//
// ComputationIdCmd (ReplicaId and PartitionId)
//===----------------------------------------------------------------------===//

class ComputationIdCmd : public CommandBufferCmd {
 public:
  enum class Kind { kReplica, kPartition };

  ComputationIdCmd(BufferAllocation::Slice dest, Kind kind);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  BufferAllocation::Slice dest_;
  Kind kind_;
};

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

class LaunchCmd : public CommandBufferCmd {
 public:
  LaunchCmd(std::string kernel_name,
            absl::Span<const BufferAllocation::Slice> args,
            absl::Span<const BufferUse::MemoryAccess> args_access,
            LaunchDimensions dims, int64_t shmem_bytes,
            std::optional<stream_executor::gpu::TmaMetadata> tma_metadata =
                std::nullopt);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  std::string kernel_name_;
  std::vector<BufferAllocation::Slice> args_;
  std::vector<BufferUse::MemoryAccess> args_access_;
  LaunchDimensions dims_;
  int64_t shmem_bytes_;
  std::optional<stream_executor::gpu::TmaMetadata> tma_metadata_;

  // Command sequence can be recorded concurrently for multiple command buffers
  // on different stream executors and we need to synchronize mutable state.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<se::Kernel>> kernels_
      ABSL_GUARDED_BY(mutex_);
};

//===----------------------------------------------------------------------===//
// CustomKernelLaunchCmd
//===----------------------------------------------------------------------===//

class CustomKernelLaunchCmd : public CommandBufferCmd {
 public:
  CustomKernelLaunchCmd(absl::Span<const BufferAllocation::Slice> args,
                        absl::Span<const BufferUse::MemoryAccess> args_access,
                        CustomKernel custom_kernel);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  std::vector<BufferAllocation::Slice> args_;
  std::vector<BufferUse::MemoryAccess> args_access_;
  CustomKernel custom_kernel_;

  // Command sequence can be recorded concurrently for multiple command buffers
  // on different stream executors and we need to synchronize mutable state.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<se::Kernel>> kernels_
      ABSL_GUARDED_BY(mutex_);
};

//===----------------------------------------------------------------------===//
// MemcpyDeviceToDeviceCmd
//===----------------------------------------------------------------------===//

class MemcpyDeviceToDeviceCmd : public CommandBufferCmd {
 public:
  MemcpyDeviceToDeviceCmd(ShapedSlice dst, ShapedSlice src, int64_t num_bytes);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  ShapedSlice dst_;
  ShapedSlice src_;
  uint64_t num_bytes_;
};

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

class MemzeroCmd : public CommandBufferCmd {
 public:
  explicit MemzeroCmd(ShapedSlice dst);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  ShapedSlice dst_;
};

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

class Memset32Cmd : public CommandBufferCmd {
 public:
  Memset32Cmd(BufferAllocation::Slice dst, uint32_t bit_pattern);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  BufferAllocation::Slice dst_;
  uint32_t bit_pattern_;
};

//===----------------------------------------------------------------------===//
// ChildCmd
//===----------------------------------------------------------------------===//

class ChildCmd : public CommandBufferCmd {
 public:
  explicit ChildCmd(CommandBufferCmdExecutor child_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  bool requires_initialization() override;

  bool force_update() override;

  bool support_loop_unroll() override { return false; }

  BufferUseVector buffers() const override;

 private:
  CommandBufferCmdExecutor child_commands_;
};

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

class CaseCmd : public CommandBufferCmd {
 public:
  CaseCmd(ShapedSlice index, std::vector<CommandBufferCmdExecutor> branches);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  bool requires_initialization() override;

  bool force_update() override;

  bool support_loop_unroll() override { return false; }

  BufferUseVector buffers() const override;

 private:
  ShapedSlice index_;
  bool index_is_bool_;
  std::vector<CommandBufferCmdExecutor> branches_;
};

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

class WhileCmd : public CommandBufferCmd {
 public:
  WhileCmd(BufferAllocation::Slice pred, CommandBufferCmdExecutor cond_commands,
           CommandBufferCmdExecutor body_commands,
           std::optional<int64_t> trip_count = std::nullopt,
           bool enable_loop_unroll = false);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Prepare(const Thunk::PrepareParams& params) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  bool requires_initialization() override;

  bool force_update() override;

  // We have not tried unrolling the loop inside another loop, so marking it
  // unsupported for now.
  bool support_loop_unroll() override { return false; }

  BufferUseVector buffers() const override;

 private:
  BufferAllocation::Slice pred_;
  CommandBufferCmdExecutor cond_commands_;
  CommandBufferCmdExecutor body_commands_;
  std::optional<int64_t> trip_count_;
  bool enable_loop_unroll_ = false;
  bool is_unrolled_loop_ = false;
};

//===----------------------------------------------------------------------===//
// GemmCmd
//===----------------------------------------------------------------------===//

class GemmCmd : public TracedCommandBufferCmd {
 public:
  GemmCmd(GemmConfig config, const BufferAllocation::Slice& lhs_buffer,
          const BufferAllocation::Slice& rhs_buffer,
          const BufferAllocation::Slice& output_buffer,
          std::optional<BufferAllocation::Slice> workspace, bool deterministic);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

  bool IsNestedCommandBuffer() const final { return true; }

 private:
  const GemmConfig config_;
  const BufferAllocation::Slice lhs_buffer_;
  const BufferAllocation::Slice rhs_buffer_;
  const BufferAllocation::Slice output_buffer_;
  std::optional<BufferAllocation::Slice> workspace_;
  // Whether to run deterministically.
  const bool deterministic_;
};

//===----------------------------------------------------------------------===//
// CublasLtCmd
//===----------------------------------------------------------------------===//

class CublasLtCmd : public TracedCommandBufferCmd, public CublasLtMatmulThunk {
 public:
  explicit CublasLtCmd(const CublasLtMatmulThunk& matmul_thunk);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  // This is needed to avoid compile errors about "shadowed" virtual function
  absl::Status Initialize(const InitializeParams& params) override {
    return CublasLtMatmulThunk::Initialize(params);
  }

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

  bool IsNestedCommandBuffer() const final { return true; }
};

//===----------------------------------------------------------------------===//
// CuDnnCmd
//===----------------------------------------------------------------------===//

class CuDnnCmd : public TracedCommandBufferCmd {
 public:
  CuDnnCmd(absl::Span<const BufferAllocation::Slice> args,
           std::shared_ptr<se::dnn::LazyDnnGraph> graph);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

  bool IsNestedCommandBuffer() const final { return true; }

 private:
  std::vector<BufferAllocation::Slice> args_;
  const std::shared_ptr<se::dnn::LazyDnnGraph> graph_;
};

//===----------------------------------------------------------------------===//
// CustomCallCmd
//===----------------------------------------------------------------------===//

class CustomCallCmd : public CommandBufferCmd {
 public:
  using CustomCallTarget = CustomCallThunk::CustomCallTarget;
  using AttributesMap = ffi::AttributesMap;

  // This is a legacy custom call API that is discouraged, and will be
  // deprecated once XLA:FFI mechanism is ready.
  CustomCallCmd(std::string target_name, CustomCallTarget call_target,
                std::vector<NullableShapedSlice> operands,
                std::vector<NullableShapedSlice> results,
                absl::string_view opaque)
      : CommandBufferCmd(CommandBufferCmdType::kCustomCallCmd),
        target_name_(std::move(target_name)),
        call_target_(std::move(call_target)),
        opaque_(opaque),
        operands_(std::move(operands)),
        results_(std::move(results)) {}

  CustomCallCmd(std::string target_name, XLA_FFI_Handler* handler,
                std::vector<NullableShapedSlice> operands,
                std::vector<NullableShapedSlice> results,
                ffi::CallFrame call_frame,
                std::shared_ptr<ffi::ExecutionState> execution_state,
                const HloComputation* called_computation)
      : CommandBufferCmd(CommandBufferCmdType::kCustomCallCmd),
        target_name_(std::move(target_name)),
        handler_(handler),
        call_frame_(std::move(call_frame)),
        execution_state_(std::move(execution_state)),
        call_frames_([this] { return call_frame_->Copy(); }),
        called_computation_(called_computation),
        operands_(std::move(operands)),
        results_(std::move(results)) {}

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;
  bool IsNestedCommandBuffer() const final { return true; }

 private:
  absl::Status RecordLegacyCustomCall(const Thunk::ExecuteParams& execute_param,
                                      RecordParams& record_params);

  absl::Status RecordXlaFfiCall(const Thunk::ExecuteParams& execute_param,
                                RecordParams& record_params);

  std::string target_name_;

  // This is a legacy custom call API that is discouraged, and will be
  // deprecated once XLA:FFI mechanism is ready.
  CustomCallTarget call_target_;
  std::string opaque_;

  // XLA FFI provides a right type safe mechanism for registering external
  // functions with XLA runtime. It's under construction, and still misses
  // a lot of features. Long term it will replace legacy custom calls.
  XLA_FFI_Handler* handler_ = nullptr;

  // Reference call frame pre-initialized at construction time.
  std::optional<ffi::CallFrame> call_frame_;

  // Execution state bound to the FFI handler. It is initialized by the
  // corresponding Thunk at construction time.
  std::shared_ptr<ffi::ExecutionState> execution_state_;

  // A pool of call frames used at run time. Newly created call frames are
  // copied from the reference call frame and updated with buffer addresses.
  std::optional<ObjectPool<ffi::CallFrame>> call_frames_;

  const HloComputation* called_computation_;

  std::vector<NullableShapedSlice> operands_;
  std::vector<NullableShapedSlice> results_;
};

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

class CollectiveCmd : public CommandBufferCmd {
 public:
  CollectiveCmd(CommandBufferCmdType cmd_type, CollectiveConfig config,
                std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::Status Prepare(const Thunk::PrepareParams& params) final;

  bool requires_initialization() override { return true; }

  bool IsNestedCommandBuffer() const final { return true; }

  absl::Status RecordTracedCommand(
      const Thunk::ExecuteParams& execute_params, RecordParams& record_params,
      absl::FunctionRef<absl::Status(se::Stream*)> trace);

  bool IsAsync() const { return async_events_ != nullptr; }
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events() const {
    return async_events_;
  }

 protected:
  const CollectiveConfig& config() const { return config_; }

 private:
  CollectiveConfig config_;
  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events_;
};

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

class AllReduceCmd : public CollectiveCmd {
 public:
  AllReduceCmd(CollectiveConfig config, ReductionKind reduction_kind,
               absl::Span<const CollectiveThunk::Buffer> buffers,
               std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  ReductionKind reduction_kind_;
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// ReduceScatterCmd
//===----------------------------------------------------------------------===//

class ReduceScatterCmd : public CollectiveCmd {
 public:
  ReduceScatterCmd(CollectiveConfig config, ReductionKind reduction_kind,
                   absl::Span<const CollectiveThunk::Buffer> buffers,
                   std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  ReductionKind reduction_kind_;
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// AllToAllCmd
//===----------------------------------------------------------------------===//

class AllToAllCmd : public CollectiveCmd {
 public:
  AllToAllCmd(CollectiveConfig config, bool has_split_dimension,
              absl::Span<const CollectiveThunk::Buffer> buffers,
              std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  bool has_split_dimension_;
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

class AllGatherCmd : public CollectiveCmd {
 public:
  AllGatherCmd(CollectiveConfig config,
               absl::Span<const CollectiveThunk::Buffer> buffers,
               std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// CollectiveBroadcastCmd
//===----------------------------------------------------------------------===//

class CollectiveBroadcastCmd : public CollectiveCmd {
 public:
  CollectiveBroadcastCmd(
      CollectiveConfig config,
      absl::Span<const CollectiveThunk::Buffer> buffers,
      std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// CollectivePermuteCmd
//===----------------------------------------------------------------------===//

class CollectivePermuteCmd : public CollectiveCmd {
 public:
  CollectivePermuteCmd(
      CollectiveConfig config, P2PConfig p2p_config,
      absl::Span<const CollectiveThunk::Buffer> buffers,
      std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

 private:
  P2PConfig p2p_config_;
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// DynamicSliceFusionCmd
//===----------------------------------------------------------------------===//

class DynamicSliceFusionCmd : public CommandBufferCmd {
 public:
  DynamicSliceFusionCmd(
      CommandBufferCmdExecutor embedded_commands,
      std::vector<std::optional<BufferAllocation::Slice>> arguments,
      std::vector<BufferAllocation> fake_allocations,
      std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>
          offsets,
      std::vector<std::optional<Shape>> orig_shapes,
      std::vector<std::optional<Shape>> sliced_shapes,
      std::vector<std::optional<PrimitiveType>> offset_primitive_types,
      std::optional<
          const DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata*>
          offset_as_function_of_indvar_metadata = std::nullopt);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Prepare(const Thunk::PrepareParams& params) final;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  BufferUseVector buffers() const override;

  bool force_update() override { return true; }

  bool requires_initialization() override;

  bool support_loop_unroll() override { return true; }

  bool IsNestedCommandBuffer() const final { return true; }

 private:
  CommandBufferCmdExecutor embedded_commands_;
  std::vector<DynamicSliceThunk::SliceDef> slices_;
  std::vector<BufferAllocation> fake_allocations_;

  // Pinned host memory for transferring offset values from device to host.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::MemoryAllocation>>
      offsets_allocs_ ABSL_GUARDED_BY(mutex_);

  // Pre-computed size requirement for `offsets_allocs_`.
  int64_t offsets_allocs_size_ = 0;

  // A mapping from argument index to the base offset in the `offsets_allocs_`.
  std::vector<int64_t> offsets_allocs_base_;

  // mapping from original allocation index to allocation index of embedded
  // command sequences.
  absl::flat_hash_map<int64_t, std::optional<BufferAllocation::Slice>>
      embeded_to_origin_slice_map_;

  // This structure holds the metadata for offset computations on host. It
  // stores a single induction variable initialization module, its update module
  // and the offsets that are a function of the induction variable.
  std::optional<
      const DynamicSliceThunk::OffsetAsFunctionOfIndvarModulesMetadata*>
      offset_as_function_of_indvar_metadata_;
};

//===----------------------------------------------------------------------===//
// DynamicSliceCopyFusionCmd
//===----------------------------------------------------------------------===//

// DynamicSliceCopyFusionCmd is a command that copies a slice from one
// buffer to another, it is only supported for static slice.
class DynamicSliceCopyFusionCmd : public CommandBufferCmd {
 public:
  DynamicSliceCopyFusionCmd(const BufferAllocation::Slice& source_buffer,
                            const BufferAllocation::Slice& destination_buffer,
                            uint64_t mem_size,
                            DynamicMemcpyThunk::Offsets offsets);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      RecordParams& record_params) override;

  bool force_update() override { return offsets_.depends_on_loop; }

  bool support_loop_unroll() override { return true; }

  BufferUseVector buffers() const override;

 private:
  const BufferAllocation::Slice source_buffer_;
  const BufferAllocation::Slice destination_buffer_;
  uint64_t mem_size_;
  DynamicMemcpyThunk::Offsets offsets_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_
