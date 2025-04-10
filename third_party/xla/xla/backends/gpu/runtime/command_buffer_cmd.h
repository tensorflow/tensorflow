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
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/ffi/api/c_api.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/shape.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla::gpu {

// clang-format off
#define COMMAND_BUFFER_CMD_LIST(V)                       \
  V(kTracedCommandBufferCmd, "TracedCommandBufferCmd")   \
  V(kComputationIdCmd, "ComputationIdCmd")               \
  V(kLaunchCmd, "LaunchCmd")                             \
  V(kCustomKernelLaunchCmd, "CustomKernelLaunchCmd")     \
  V(kCublasLtCmd, "CublasLtCmd")                         \
  V(kCuDnnCmd, "CuDnnCmd")                               \
  V(kGemmCmd, "GemmCmd")                                 \
  V(kMemcpyDeviceToDeviceCmd, "MemcpyDeviceToDeviceCmd") \
  V(kMemzeroCmd, "MemzeroCmd")                           \
  V(kMemset32Cmd, "Memset32Cmd")                         \
  V(kCaseCmd, "CaseCmd")                                 \
  V(kWhileCmd, "WhileCmd")                               \
  V(kCustomCallCmd, "CustomCallCmd")                     \
  V(kBarrierCmd, "BarrierCmd")                           \
  V(kCollectiveCmd, "CollectiveCmd")                     \
  V(kAllReduceCmd, "AllReduceCmd")                       \
  V(kReduceScatter, "ReduceScatterCmd")                  \
  V(kAllToAll, "AllToAllCmd")                            \
  V(kAllGatherCmd, "AllGatherCmd")                       \
  V(kCollectiveBroadcastCmd, "CollectiveBroadcastCmd")   \
  V(kDynamicSliceFusionCmd, "DynamicSliceFusionCmd")     \
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

// Command is a Thunk counterpart that instead of launching operations directly
// on the underlying device records them into command buffers.
//
// Commands have the same execution stages as thunks as they are executed by a
// command buffer thunk: Prepare, Initialize and Record (Execute). See Thunk
// documentation for details.
//
// Commands must be thread safe as they can be recorded into multiple command
// buffers concurrently on different stream executors.
class CommandBufferCmd {
 public:
  CommandBufferCmd(CommandBufferCmdType cmd_type,
                   ExecutionStreamId execution_stream_id)
      : cmd_type_(cmd_type), execution_stream_id_(execution_stream_id) {}
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
    ConcreteState* GetOrNull(const CommandBufferCmd* cmd,
                             const se::CommandBuffer* command_buffer) {
      static_assert(std::is_base_of_v<State, ConcreteState>);
      return static_cast<ConcreteState*>(
          GetOrNull(cmd, command_buffer, GetTypeId<ConcreteState>()));
    }

    template <typename ConcreteState>
    ConcreteState* GetOrCreate(
        const CommandBufferCmd* cmd, const se::CommandBuffer* command_buffer,
        absl::FunctionRef<std::unique_ptr<ConcreteState>()> create) {
      static_assert(std::is_base_of_v<State, ConcreteState>);
      return static_cast<ConcreteState*>(GetOrCreate(cmd, command_buffer,
                                                     GetTypeId<ConcreteState>(),
                                                     [&] { return create(); }));
    }

    template <typename ConcreteState>
    ConcreteState* GetOrCreate(const CommandBufferCmd* cmd,
                               const se::CommandBuffer* command_buffer) {
      return GetOrCreate<ConcreteState>(cmd, command_buffer, [] {
        return std::make_unique<ConcreteState>();
      });
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

    State* GetOrNull(const CommandBufferCmd* cmd,
                     const se::CommandBuffer* command_buffer, TypeId type_id);

    State* GetOrCreate(const CommandBufferCmd* cmd,
                       const se::CommandBuffer* command_buffer, TypeId type_id,
                       absl::FunctionRef<std::unique_ptr<State>()> create);

    using Key =
        std::tuple<const CommandBufferCmd*, const se::CommandBuffer*, TypeId>;
    absl::flat_hash_map<Key, std::unique_ptr<State>> state_;
  };

  // Parameters for recording commands into the command buffer.
  struct RecordParams {
    // An external state manager that gives efficient access to per-device state
    // to commands without a need to add expensive synchronization.
    StateManager& state;
  };

  // A list of commands recorded into the command buffer (or updated).
  struct RecordedCommands {
    // Creates a recorded commands from a single se::CommandBuffer command.
    static absl::StatusOr<RecordedCommands> Create(
        absl::StatusOr<const se::CommandBuffer::Command*> command);

    absl::InlinedVector<const se::CommandBuffer::Command*, 2> commands;
  };

  // Create new commands in the command buffer using the given dependencies.
  struct RecordCreate {
    absl::Span<const se::CommandBuffer::Command*> dependencies;
  };

  // Update previously recorded commands in the command buffer.
  struct RecordUpdate {
    RecordedCommands recorded_commands;
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
  virtual absl::Status Prepare(
      const Thunk::PrepareParams& params,
      Thunk::ResourceRequestsInterface& resource_requests) {
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
  virtual absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) = 0;

  // For some commands need to force update on Record even the input device
  // pointers do not change, e.g. command that has state that can be changed by
  // CPU code.
  virtual bool force_update() { return false; }

  // Returns all buffers used by the cmd. These will be used to track cmd
  // updates, thus they need to be consistent across calls to the function.
  virtual BufferUseVector buffers() = 0;

  // Returns true if command implemented as a nested command buffer.
  virtual bool IsNestedCommandBuffer() const { return false; }

  absl::string_view profile_annotation() const { return profile_annotation_; }
  void set_profile_annotation(absl::string_view profile_annotation) {
    profile_annotation_ = profile_annotation;
  }

  CommandBufferCmdType command_type() const { return cmd_type_; }

  virtual std::string ToString() const {
    return CommandBufferCmdString(cmd_type_);
  }

  ExecutionStreamId execution_stream_id() const { return execution_stream_id_; }

 private:
  std::string profile_annotation_;
  CommandBufferCmdType cmd_type_;
  ExecutionStreamId execution_stream_id_;
};

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

// A sequence of command buffer commands that create or update a command buffer.
// You can think of CommandBufferCmdSequence as a mini interpreter whose sole
// purpose is to manipulate command buffers at run time.
class CommandBufferCmdSequence {
 public:
  CommandBufferCmdSequence() = default;
  CommandBufferCmdSequence(CommandBufferCmdSequence&&) = default;
  CommandBufferCmdSequence& operator=(CommandBufferCmdSequence&&) = default;

  // Synchronization mode defines how much concurrency is allowed between
  // commands in the sequence.
  enum class SynchronizationMode {
    // Serializes execution of all commands recorded into the command buffer
    // by adding a dependency between them.
    kSerialize,

    // Relies on execution graph to insert dependencies between commands
    // that have buffer of resource conflicts, and building a DAG of commands.
    kAutomatic
  };

  // A command buffer cmd sequence builder for lazy command sequence
  // construction.
  class Builder {
   public:
    void Append(std::unique_ptr<CommandBufferCmd> cmd);

    template <typename T, typename... Args>
    void Emplace(Args... args) {
      Append(std::make_unique<T>(std::forward<Args>(args)...));
    }

    CommandBufferCmdSequence Build(SynchronizationMode synchronization_mode) &&;

   private:
    std::vector<std::unique_ptr<CommandBufferCmd>> commands_;
  };

  enum class RecordMode {
    // In exclusive mode no one else is recording commands into the command
    // buffer argument, and cmd sequence is responsible for updating command
    // buffer state: finalizing after all commands recorded, and
    // switching to update state before recording updates.
    kExclusive,

    // In conditional mode multiple cmd sequences can be recorded into the
    // command buffer argument, and with command buffer state managed externally
    // cmd sequence should not finalize or update it. This mode is used when
    // command buffer cmd sequence is recorded into conditional command buffers
    // owned by the parent command buffer.
    kConditional
  };

  // Prepares all commands added to a sequence.
  absl::Status Prepare(const Thunk::PrepareParams& params,
                       Thunk::ResourceRequestsInterface& resource_requests);

  // Initializes all commands added to a sequence.
  absl::Status Initialize(const Thunk::InitializeParams& params,
                          CommandBufferCmd::StateManager& state);

  // Records all commands added to a sequence into the given command buffer.
  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const CommandBufferCmd::RecordParams& record_params,
                      se::CommandBuffer* command_buffer,
                      RecordMode mode = RecordMode::kExclusive);

  // Returns buffers referenced by commands in this sequence.
  const absl::flat_hash_set<BufferUse>& buffers() const;

  // Returns buffer allocations indices referenced by commands in this sequence.
  const absl::flat_hash_set<BufferAllocation::Index>& allocs_indices() const;

  bool empty() const { return commands_.empty(); }
  size_t size() const { return commands_.size(); }

  bool force_update() const {
    return absl::c_any_of(commands_,
                          [](const auto& cmd) { return cmd->force_update(); });
  }

 private:
  // A state associated with commands in the sequence. We rely on this state to
  // efficiently update command recorded into the command buffer.
  struct RecordState : public CommandBufferCmd::State {
    CommandBufferCmd::RecordedCommands recorded_commands;
  };

  CommandBufferCmdSequence(
      SynchronizationMode synchronization_mode,
      std::vector<std::unique_ptr<CommandBufferCmd>> commands);

  SynchronizationMode synchronization_mode_;
  std::vector<std::unique_ptr<CommandBufferCmd>> commands_;

  // Buffers referenced by commands in this sequence.
  absl::flat_hash_set<BufferUse> buffers_;

  // Buffer allocations indices referenced by commands in this sequence.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices_;
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
  explicit TracedCommandBuffer(const CommandBufferCmd* trace_cmd,
                               CommandBufferCmd::BufferUseVector buffers,
                               int64_t capacity = 16);

  // Returns cached command buffer traced using the same buffer addresses or
  // traces and caches a new command buffer using user provided callback.
  absl::StatusOr<se::CommandBuffer*> GetOrTraceCommandBuffer(
      const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
      se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace);

 private:
  std::vector<BufferAllocation::Index> allocs_indices_;

  struct Entry {
    std::vector<se::DeviceMemoryBase> recorded_allocs;
    std::unique_ptr<se::CommandBuffer> command_buffer;
  };
  const CommandBufferCmd* trace_cmd_;
  int64_t capacity_;
  std::vector<Entry> entries_;
};

//===----------------------------------------------------------------------===//
// TracedCommandBufferCmd
//===----------------------------------------------------------------------===//

// A base class for commands implemented as tracing of stream activities.
class TracedCommandBufferCmd : public CommandBufferCmd {
 protected:
  explicit TracedCommandBufferCmd(CommandBufferCmdType cmd_type,
                                  ExecutionStreamId execution_stream_id);

  // Creates a command buffer by calling a user-provided `trace` function and
  // adds it as a nested command to `command_buffer`. Traced command buffers
  // cached and reused in an instance of `TracedCommandBuffer` kept in `state`.
  absl::StatusOr<RecordedCommands> RecordTracedCommand(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer,
      absl::FunctionRef<absl::Status(se::Stream*)> trace);
};

//===----------------------------------------------------------------------===//
// ComputationIdCmd (ReplicaId and PartitionId)
//===----------------------------------------------------------------------===//

class ComputationIdCmd : public CommandBufferCmd {
 public:
  enum class Kind { kReplica, kPartition };

  ComputationIdCmd(ExecutionStreamId execution_stream_id,
                   BufferAllocation::Slice dest, Kind kind);

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice dest_;
  Kind kind_;
};

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

class LaunchCmd : public CommandBufferCmd {
 public:
  LaunchCmd(ExecutionStreamId execution_stream_id, std::string kernel_name,
            absl::Span<const BufferAllocation::Slice> args,
            absl::Span<const BufferUse::MemoryAccess> args_access,
            LaunchDimensions dims, int64_t shmem_bytes);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

 private:
  std::string kernel_name_;
  std::vector<BufferAllocation::Slice> args_;
  std::vector<BufferUse::MemoryAccess> args_access_;
  LaunchDimensions dims_;
  int64_t shmem_bytes_;

  // Command sequence can be recorded concurrently for multiple command buffers
  // on different stream executors and we need to synchronize mutable state.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<se::Kernel>> kernels_
      ABSL_GUARDED_BY(mutex_);
};

//===----------------------------------------------------------------------===//
// CustomKenelLaunchCmd
//===----------------------------------------------------------------------===//

class CustomKernelLaunchCmd : public CommandBufferCmd {
 public:
  CustomKernelLaunchCmd(ExecutionStreamId execution_stream_id,
                        absl::Span<const BufferAllocation::Slice> args,
                        absl::Span<const BufferUse::MemoryAccess> args_access,
                        CustomKernel custom_kernel);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

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
  MemcpyDeviceToDeviceCmd(ExecutionStreamId execution_stream_id,
                          BufferAllocation::Slice dst,
                          BufferAllocation::Slice src, int64_t num_bytes);

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice dst_;
  BufferAllocation::Slice src_;
  int64_t num_bytes_;
};

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

class MemzeroCmd : public CommandBufferCmd {
 public:
  MemzeroCmd(ExecutionStreamId execution_stream_id,
             BufferAllocation::Slice dst);

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice dst_;
};

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

class Memset32Cmd : public CommandBufferCmd {
 public:
  Memset32Cmd(ExecutionStreamId execution_stream_id,
              BufferAllocation::Slice dst, uint32_t bit_pattern);

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice dst_;
  uint32_t bit_pattern_;
};

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

class CaseCmd : public CommandBufferCmd {
 public:
  CaseCmd(ExecutionStreamId execution_stream_id, BufferAllocation::Slice index,
          bool index_is_bool,
          std::vector<CommandBufferCmdSequence> branches_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  bool force_update() override;

  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice index_;
  bool index_is_bool_;
  std::vector<CommandBufferCmdSequence> branches_commands_;
};

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

class WhileCmd : public CommandBufferCmd {
 public:
  WhileCmd(ExecutionStreamId execution_stream_id, BufferAllocation::Slice pred,
           CommandBufferCmdSequence cond_commands,
           CommandBufferCmdSequence body_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  bool force_update() override;

  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice pred_;
  CommandBufferCmdSequence cond_commands_;
  CommandBufferCmdSequence body_commands_;
};

//===----------------------------------------------------------------------===//
// GemmCmd
//===----------------------------------------------------------------------===//

class GemmCmd : public TracedCommandBufferCmd {
 public:
  GemmCmd(ExecutionStreamId execution_stream_id, GemmConfig config,
          const BufferAllocation::Slice& lhs_buffer,
          const BufferAllocation::Slice& rhs_buffer,
          const BufferAllocation::Slice& output_buffer,
          const BufferAllocation::Slice& workspace, bool deterministic);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

  bool IsNestedCommandBuffer() const final { return true; }

 private:
  const GemmConfig config_;
  const BufferAllocation::Slice lhs_buffer_;
  const BufferAllocation::Slice rhs_buffer_;
  const BufferAllocation::Slice output_buffer_;
  const BufferAllocation::Slice workspace_;
  // Whether to run deterministically.
  const bool deterministic_;
};

//===----------------------------------------------------------------------===//
// CublasLtCmd
//===----------------------------------------------------------------------===//

class CublasLtCmd : public TracedCommandBufferCmd {
 public:
  CublasLtCmd(ExecutionStreamId execution_stream_id, GemmConfig gemm_config,
              se::gpu::BlasLt::Epilogue epilogue, int64_t algorithm_idx,
              BufferAllocation::Slice a_buffer,
              BufferAllocation::Slice b_buffer,
              BufferAllocation::Slice c_buffer,
              BufferAllocation::Slice d_buffer,
              BufferAllocation::Slice bias_buffer /* may be null */,
              BufferAllocation::Slice aux_buffer /* may be null */,
              BufferAllocation::Slice a_scale_buffer /* may be null */,
              BufferAllocation::Slice b_scale_buffer /* may be null */,
              BufferAllocation::Slice c_scale_buffer /* may be null */,
              BufferAllocation::Slice d_scale_buffer /* may be null */,
              BufferAllocation::Slice d_amax_buffer /* may be null */,
              BufferAllocation::Slice workspace_buffer);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

  bool IsNestedCommandBuffer() const final { return true; }

 private:
  absl::StatusOr<se::gpu::BlasLt::MatmulPlan*> GetMatmulPlan(
      const se::Stream* stream);

  absl::StatusOr<se::gpu::BlasLt::MatmulAlgorithm> GetMatmulAlgorithm(
      const se::Stream* stream, const se::gpu::BlasLt::MatmulPlan* plan,
      int64_t max_workspace);

  absl::Mutex matmul_plans_cache_mutex_;
  absl::flat_hash_map<const se::Stream*, se::gpu::BlasLt::MatmulPlanPtr>
      matmul_plans_cache_ ABSL_GUARDED_BY(matmul_plans_cache_mutex_);

  absl::Mutex matmul_algorithm_cache_mutex_;
  absl::flat_hash_map<const se::gpu::BlasLt::MatmulPlan*,
                      se::gpu::BlasLt::MatmulAlgorithm>
      matmul_algorithm_cache_ ABSL_GUARDED_BY(matmul_algorithm_cache_mutex_);

  const GemmConfig gemm_config_;
  const se::gpu::BlasLt::Epilogue epilogue_;
  const int64_t algorithm_idx_;
  const BufferAllocation::Slice a_buffer_;
  const BufferAllocation::Slice b_buffer_;
  const BufferAllocation::Slice c_buffer_;
  const BufferAllocation::Slice d_buffer_;
  const BufferAllocation::Slice bias_buffer_;
  const BufferAllocation::Slice aux_buffer_;
  const BufferAllocation::Slice a_scale_buffer_;
  const BufferAllocation::Slice b_scale_buffer_;
  const BufferAllocation::Slice c_scale_buffer_;
  const BufferAllocation::Slice d_scale_buffer_;
  const BufferAllocation::Slice d_amax_buffer_;
  const BufferAllocation::Slice workspace_buffer_;
};

//===----------------------------------------------------------------------===//
// CuDnnCmd
//===----------------------------------------------------------------------===//

class CuDnnCmd : public TracedCommandBufferCmd {
 public:
  CuDnnCmd(ExecutionStreamId execution_stream_id,
           absl::Span<const BufferAllocation::Slice> args,
           std::shared_ptr<se::dnn::LazyDnnGraph> graph);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

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
  using Slice = CustomCallThunk::Slice;
  using CustomCallTarget = CustomCallThunk::CustomCallTarget;
  using AttributesMap = CustomCallThunk::AttributesMap;

  // This is a legacy custom call API that is discouraged, and will be
  // deprecated once XLA:FFI mechanism is ready.
  CustomCallCmd(ExecutionStreamId execution_stream_id, std::string target_name,
                CustomCallTarget call_target,
                std::vector<std::optional<Slice>> operands,
                std::vector<std::optional<Slice>> results,
                absl::string_view opaque)
      : CommandBufferCmd(CommandBufferCmdType::kCustomCallCmd,
                         execution_stream_id),
        target_name_(std::move(target_name)),
        call_target_(std::move(call_target)),
        opaque_(opaque),
        operands_(std::move(operands)),
        results_(std::move(results)) {}

  CustomCallCmd(ExecutionStreamId execution_stream_id, std::string target_name,
                XLA_FFI_Handler* handler,
                std::vector<std::optional<Slice>> operands,
                std::vector<std::optional<Slice>> results,
                AttributesMap attributes,
                const HloComputation* called_computation)
      : CommandBufferCmd(CommandBufferCmdType::kCustomCallCmd,
                         execution_stream_id),
        target_name_(std::move(target_name)),
        handler_(handler),
        attributes_(std::move(attributes)),
        called_computation_(called_computation),
        operands_(std::move(operands)),
        results_(std::move(results)) {}

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;
  bool IsNestedCommandBuffer() const final { return true; }

 private:
  absl::StatusOr<RecordedCommands> RecordLegacyCustomCall(
      const Thunk::ExecuteParams& execute_param,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer);

  absl::StatusOr<RecordedCommands> RecordXlaFfiCall(
      const Thunk::ExecuteParams& execute_param,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer);

  std::string target_name_;

  // This is a legacy custom call API that is discouraged, and will be
  // deprecated once XLA:FFI mechanism is ready.
  CustomCallTarget call_target_;
  std::string opaque_;

  // XLA FFI provides a right type safe mechanism for registering external
  // functions with XLA runtime. It's under construction, and still misses
  // a lot of features. Long term it will replace legacy custom calls.
  XLA_FFI_Handler* handler_ = nullptr;
  AttributesMap attributes_;
  const HloComputation* called_computation_;

  std::vector<std::optional<Slice>> operands_;
  std::vector<std::optional<Slice>> results_;
};

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

class CollectiveCmd : public CommandBufferCmd {
 public:
  CollectiveCmd(CommandBufferCmdType cmd_type,
                ExecutionStreamId execution_stream_id,
                ExecutionStreamId async_from_stream_id,
                CollectiveConfig config);

  absl::Status Prepare(
      const Thunk::PrepareParams& params,
      Thunk::ResourceRequestsInterface& resource_requests) final;

  bool force_update() override { return true; }

  bool IsNestedCommandBuffer() const final { return true; }

  absl::StatusOr<RecordedCommands> RecordTracedCommand(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer,
      absl::FunctionRef<absl::Status(se::Stream*)> trace);

  virtual AsyncStreamKind GetAsyncStreamKind() = 0;

  bool IsAsync() const {
    return async_from_stream_id_ != execution_stream_id();
  }

  CollectiveStreamId nccl_stream_id() {
    return xla::gpu::GetCollectiveStreamId(IsAsync(), GetAsyncStreamKind());
  }

  ExecutionStreamId async_from_stream_id() const {
    return async_from_stream_id_;
  }

 protected:
  const CollectiveConfig& config() const { return config_; }

 private:
  ExecutionStreamId async_from_stream_id_;
  CollectiveConfig config_;
};

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

class AllReduceCmd : public CollectiveCmd {
 public:
  AllReduceCmd(ExecutionStreamId execution_stream_id,
               ExecutionStreamId async_from_stream_id, CollectiveConfig config,
               ReductionKind reduction_kind,
               absl::Span<const CollectiveThunk::Buffer> buffers);

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

 private:
  ReductionKind reduction_kind_;
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// ReduceScatterCmd
//===----------------------------------------------------------------------===//

class ReduceScatterCmd : public CollectiveCmd {
 public:
  ReduceScatterCmd(ExecutionStreamId execution_stream_id,
                   ExecutionStreamId async_from_stream_id,
                   CollectiveConfig config, ReductionKind reduction_kind,
                   absl::Span<const CollectiveThunk::Buffer> buffers);

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

 private:
  ReductionKind reduction_kind_;
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// AllToAllCmd
//===----------------------------------------------------------------------===//

class AllToAllCmd : public CollectiveCmd {
 public:
  AllToAllCmd(ExecutionStreamId execution_stream_id,
              ExecutionStreamId async_from_stream_id, CollectiveConfig config,
              bool has_split_dimension,
              absl::Span<const CollectiveThunk::Buffer> buffers);

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

 private:
  bool has_split_dimension_;
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

class AllGatherCmd : public CollectiveCmd {
 public:
  AllGatherCmd(ExecutionStreamId execution_stream_id,
               ExecutionStreamId async_from_stream_id, CollectiveConfig config,
               absl::Span<const CollectiveThunk::Buffer> buffers);

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

 private:
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// CollectiveBroadcastCmd
//===----------------------------------------------------------------------===//

class CollectiveBroadcastCmd : public CollectiveCmd {
 public:
  CollectiveBroadcastCmd(ExecutionStreamId execution_stream_id,
                         ExecutionStreamId async_from_stream_id,
                         CollectiveConfig config,
                         absl::Span<const CollectiveThunk::Buffer> buffers);

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

 private:
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// DynamicSliceFusionCmd
//===----------------------------------------------------------------------===//

class DynamicSliceFusionCmd : public CommandBufferCmd {
 public:
  DynamicSliceFusionCmd(
      ExecutionStreamId execution_stream_id,
      CommandBufferCmdSequence embedded_commands,
      std::vector<std::optional<BufferAllocation::Slice>> arguments,
      std::vector<std::unique_ptr<BufferAllocation>> fake_allocations_,
      std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>>
          offsets,
      std::vector<std::optional<Shape>> orig_shapes,
      std::vector<std::optional<Shape>> sliced_shapes,
      std::vector<std::optional<uint64_t>> offset_byte_sizes);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Prepare(
      const Thunk::PrepareParams& params,
      Thunk::ResourceRequestsInterface& resource_requests) final;

  absl::StatusOr<RecordedCommands> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUseVector buffers() override;

  bool force_update() override;

  bool IsNestedCommandBuffer() const final { return true; }

 private:
  CommandBufferCmdSequence embedded_commands_;
  std::vector<DynamicSliceThunk::SliceDef> slices_;
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations_;

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
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_
