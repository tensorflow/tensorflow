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

#ifndef XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_
#define XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/ffi/api/c_api.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/runtime/custom_call_thunk.h"
#include "xla/service/gpu/runtime/nccl_api.h"
#include "xla/service/gpu/runtime/nccl_clique_key.h"
#include "xla/service/gpu/runtime/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/status.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla::gpu {

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
  explicit CommandBufferCmd(ExecutionStreamId execution_stream_id)
      : execution_stream_id_(execution_stream_id) {}
  virtual ~CommandBufferCmd() = default;

  enum class MemoryAccess { kRead, kWrite };

  // BufferUsage tracks memory access type for a buffer slice, so that we can
  // correctly insert command buffer barriers to avoid read/write conflicts.
  struct BufferUsage {
    BufferUsage(BufferAllocation::Slice slice, MemoryAccess access)
        : slice(slice), access(access) {}

    template <typename H>
    friend H AbslHashValue(H h, const BufferUsage& buffer) {
      return H::combine(std::move(h), buffer.slice, buffer.access);
    }

    bool operator==(const BufferUsage& other) const {
      return slice == other.slice && access == other.access;
    }

    BufferAllocation::Slice slice;
    MemoryAccess access;
  };

  using BufferUsageVector = absl::InlinedVector<BufferUsage, 4>;

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

  // An external manager for a state attached to commands.
  class StateManager {
   public:
    virtual ~StateManager() = default;

    template <typename ConcreteState>
    ConcreteState* GetOrNull(const CommandBufferCmd* cmd) {
      static_assert(std::is_base_of_v<State, ConcreteState>);
      return static_cast<ConcreteState*>(GetOrNull(cmd));
    }

    template <typename ConcreteState>
    ConcreteState* GetOrCreate(
        const CommandBufferCmd* cmd,
        absl::FunctionRef<std::unique_ptr<ConcreteState>()> create) {
      static_assert(std::is_base_of_v<State, ConcreteState>);
      return static_cast<ConcreteState*>(GetOrCreate(
          cmd, [&]() -> std::unique_ptr<State> { return create(); }));
    }

    template <typename ConcreteState>
    ConcreteState* GetOrCreate(const CommandBufferCmd* cmd) {
      static_assert(std::is_base_of_v<State, ConcreteState>);
      return static_cast<ConcreteState*>(
          GetOrCreate(cmd, [] { return std::make_unique<ConcreteState>(); }));
    }

   private:
    State* GetOrNull(const CommandBufferCmd* cmd);

    State* GetOrCreate(const CommandBufferCmd* cmd,
                       absl::FunctionRef<std::unique_ptr<State>()> create);

    absl::flat_hash_map<const CommandBufferCmd*, std::unique_ptr<State>> state_;
  };

  // Parameters for recording commands into the command buffer.
  struct RecordParams {
    // An external state manager that gives efficient access to per-device state
    // to commands without a need to add expensive synchronization.
    StateManager& state;

    // Execution scope id defines the default execution scope that should be
    // used for recording commands. Each individual command uses this scope plus
    // its own execution stream id to compute the execution scope that will be
    // used for adding commands to command buffer. It is a command sequence
    // responsibility to guarantee that all commands eventually will be
    // correctly synchronized with an execution scope id passed as argument.
    //
    // This argument allows conditional commands to record a command sequence
    // into non-default execution scope.
    se::CommandBuffer::ExecutionScopeId execution_scope_id =
        se::CommandBuffer::kDefaulExecutionScope;
  };

  // See Thunk documentation for XLA execution stages (prepare, initialize,
  // execute). Commands mirror thunks as they are executed as CommandBufferThunk
  // that is plugged into the Thunk execution cycle.

  // Prepare command for execution by allowing command to request shared state
  // required for recording (i.e. collective commands request cliques).
  virtual absl::Status Prepare(const Thunk::PrepareParams& params,
                               Thunk::ResourceRequests& resource_requests) {
    return absl::OkStatus();
  }

  // Initialize a command for recording on a given executor. We split it into a
  // separate function to allow expensive initialization (e.g. device kernel
  // loading) to happen before a command buffer thunk execution.
  virtual absl::Status Initialize(const Thunk::InitializeParams& params,
                                  StateManager& state) {
    return absl::OkStatus();
  }

  // Records command into the command buffer using given execution scope.
  virtual absl::Status Record(const Thunk::ExecuteParams& execute_params,
                              const RecordParams& record_params,
                              se::CommandBuffer* command_buffer) = 0;

  // For some commands need to force update on Record even the input device
  // pointers do not change, e.g. command that has state that can be changed by
  // CPU code.
  virtual bool force_update() { return false; }

  // Returns all buffers used by the cmd. These will be used to track cmd
  // updates, thus they need to be consistent across calls to the function.
  virtual BufferUsageVector buffers() = 0;

  // Returns true if command implemented as a nested command buffer.
  virtual bool IsNestedCommandBuffer() const { return false; }

  // Returns a command execution scope created from the specified
  // 'execution_stream_id'.
  se::CommandBuffer::ExecutionScopeId GetExecutionScope(
      const RecordParams& record_params,
      ExecutionStreamId execution_stream_id) const;

  // Return the execution scope created from the execution stream id of the
  // thunk which is lowered to current command.
  virtual se::CommandBuffer::ExecutionScopeId GetExecutionScope(
      const CommandBufferCmd::RecordParams& record_params) const;

  std::string_view profile_annotation() const { return profile_annotation_; }
  void set_profile_annotation(std::string_view profile_annotation) {
    profile_annotation_ = profile_annotation;
  }

  ExecutionStreamId execution_stream_id() const { return execution_stream_id_; }

 private:
  std::string profile_annotation_;
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
  // Synchronization mode defines how execution streams gets converted to
  // command buffer execution scopes and barriers.
  //
  // Each individual Thunk assigned an execution stream id, and we have explicit
  // inter-stream synchronization (`Thunk::Kind::kWaitForStreams`) between
  // streams. Thunks assigned to the same stream are implicitly synchronized.
  //
  // Command buffers on the other hand by default can execute commands
  // concurrently and require barriers to enforce execution order.
  //
  // WARNING: We do not have implicit synchronization between execution scopes
  // corresponding to different execution streams and rely on explicit barriers
  // emitted from thunks. Synchronization mode controls only barriers within
  // a single exection scope (corresponds to execution stream).
  enum class SynchronizationMode {
    // Adds barriers between all commands recorded into the same execution scope
    // (thunks sharing execution stream) and enforces completely serialized
    // execution order that matches what would happen in a ThunkSequence.
    kSerialize,

    // Relies on buffer use analysis to insert barriers only between commands
    // that have read-write conflicts into the same buffers. Conflicts are
    // detected only between commands using the same stream id, and inter-stream
    // synchronization is a user responsibility.
    kAutomatic
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

  explicit CommandBufferCmdSequence(SynchronizationMode synchronization_mode =
                                        SynchronizationMode::kAutomatic);

  void Append(std::unique_ptr<CommandBufferCmd> cmd);

  template <typename T, typename... Args>
  void Emplace(Args... args) {
    Append(std::make_unique<T>(std::forward<Args>(args)...));
  }

  // Prepares all commands added to a sequence.
  absl::Status Prepare(const Thunk::PrepareParams& params,
                       Thunk::ResourceRequests& resource_requests);

  // Initializes all commands added to a sequence.
  absl::Status Initialize(const Thunk::InitializeParams& params,
                          CommandBufferCmd::StateManager& state);

  // Records all commands added to a sequence into the given command buffer.
  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const CommandBufferCmd::RecordParams& record_params,
                      se::CommandBuffer* command_buffer,
                      RecordMode mode = RecordMode::kExclusive);

  // Returns buffers referenced by commands in this sequence.
  const absl::flat_hash_set<CommandBufferCmd::BufferUsage>& buffers() const;

  // Returns buffer allocations indices referenced by commands in this sequence.
  const absl::flat_hash_set<BufferAllocation::Index>& allocs_indices() const;

  // Returns a vector that tells if command at the given index requires a
  // barrier.
  std::vector<bool> barriers() const;

  bool empty() const { return commands_.empty(); }
  size_t size() const { return commands_.size(); }

  bool force_update() const {
    return absl::c_any_of(commands_, [](const CommandInfo& cmd_info) {
      return cmd_info.cmd->force_update();
    });
  }

 private:
  struct CommandInfo {
    std::unique_ptr<CommandBufferCmd> cmd;
    bool requires_barrier;
  };

  // Functions for tracking buffer usage of recorded commands and figuring out
  // when the next command requires a barrier for correctness.
  bool HasConflicts(ExecutionStreamId execution_stream_id,
                    const CommandBufferCmd::BufferUsageVector& buffers);
  void TrackBuffers(ExecutionStreamId execution_stream_id,
                    const CommandBufferCmd::BufferUsageVector& buffers);
  void ClearTrackedBuffers(ExecutionStreamId execution_stream_id);

  SynchronizationMode synchronization_mode_;
  std::vector<CommandInfo> commands_;

  // Buffers referenced by commands in this sequence.
  absl::flat_hash_set<CommandBufferCmd::BufferUsage> buffers_;

  // Buffer allocations indices referenced by commands in this sequence.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices_;

  // We track read and write sets of commands recorded into the command
  // sequence to detect conflicts and insert explicit barriers. These are the
  // buffer allocation slices used by commands appended since the last barrier.
  struct ReadWriteSet {
    absl::flat_hash_set<BufferAllocation::Slice> read;
    absl::flat_hash_set<BufferAllocation::Slice> write;
  };

  absl::flat_hash_map<ExecutionStreamId, ReadWriteSet> read_write_sets_;
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
  explicit TracedCommandBuffer(CommandBufferCmd::BufferUsageVector buffers,
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

  int64_t capacity_;
  std::vector<Entry> entries_;
};

//===----------------------------------------------------------------------===//
// TracedCommandBufferCmd
//===----------------------------------------------------------------------===//

// A base class for commands implemented as tracing of stream activities.
class TracedCommandBufferCmd : public CommandBufferCmd {
 protected:
  explicit TracedCommandBufferCmd(ExecutionStreamId execution_stream_id);

  // Creates a command buffer by calling a user-provided `trace` function and
  // adds it as a nested command to `command_buffer`. Traced command buffers
  // cached and reused in an instance of `TracedCommandBuffer` kept in `state`.
  absl::Status AddTracedCommandBuffer(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, se::CommandBuffer* command_buffer,
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

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  BufferAllocation::Slice dest_;
  Kind kind_;

  // Command sequence can be recorded concurrently for multiple command buffers
  // on different stream executors and we need to synchronize mutable state.
  absl::Mutex mutex_;

  // TODO(ezhulenev): This is a workaround for CUDA graphs + conditional nodes
  // bug that will be fixed in CUDA 12.4.1 release: currently it's impossible to
  // update a memset node inside a conditional graph. Instead of using memset
  // node we replace it with a kernel launch node of CUDA kernels doing 1D
  // memset. This should be removed when bug is fixed in CUDA.
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<se::Kernel>>
      memset_kernels_ ABSL_GUARDED_BY(mutex_);
};

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

class LaunchCmd : public CommandBufferCmd {
 public:
  LaunchCmd(ExecutionStreamId execution_stream_id, std::string kernel_name,
            absl::Span<const BufferAllocation::Slice> args,
            absl::Span<const MemoryAccess> args_access, LaunchDimensions dims,
            int64_t shmem_bytes);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  std::string kernel_name_;
  std::vector<BufferAllocation::Slice> args_;
  std::vector<MemoryAccess> args_access_;
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
                        absl::Span<const MemoryAccess> args_access,
                        CustomKernel custom_kernel);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  std::vector<BufferAllocation::Slice> args_;
  std::vector<MemoryAccess> args_access_;
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

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

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

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

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

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  BufferAllocation::Slice dst_;
  uint32_t bit_pattern_;
};

//===----------------------------------------------------------------------===//
// IfCmd
//===----------------------------------------------------------------------===//

class IfCmd : public CommandBufferCmd {
 public:
  IfCmd(ExecutionStreamId execution_stream_id, BufferAllocation::Slice pred,
        CommandBufferCmdSequence then_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  BufferAllocation::Slice pred_;
  CommandBufferCmdSequence then_commands_;
};

//===----------------------------------------------------------------------===//
// IfElseCmd
//===----------------------------------------------------------------------===//

class IfElseCmd : public CommandBufferCmd {
 public:
  IfElseCmd(ExecutionStreamId execution_stream_id, BufferAllocation::Slice pred,
            CommandBufferCmdSequence then_commands,
            CommandBufferCmdSequence else_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  BufferAllocation::Slice pred_;
  CommandBufferCmdSequence then_commands_;
  CommandBufferCmdSequence else_commands_;
};

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

class CaseCmd : public CommandBufferCmd {
 public:
  CaseCmd(ExecutionStreamId execution_stream_id, BufferAllocation::Slice index,
          std::vector<CommandBufferCmdSequence> branches_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  BufferAllocation::Slice index_;
  std::vector<CommandBufferCmdSequence> branches_commands_;
};

//===----------------------------------------------------------------------===//
// ForCmd
//===----------------------------------------------------------------------===//

class ForCmd : public CommandBufferCmd {
 public:
  ForCmd(ExecutionStreamId execution_stream_id, int32_t num_iterations,
         BufferAllocation::Slice loop_counter,
         CommandBufferCmdSequence body_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  int32_t num_iterations_;
  BufferAllocation::Slice loop_counter_;
  CommandBufferCmdSequence body_commands_;
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

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

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

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

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
// CuDnnCmd
//===----------------------------------------------------------------------===//

class CuDnnCmd : public TracedCommandBufferCmd {
 public:
  CuDnnCmd(ExecutionStreamId execution_stream_id,
           absl::Span<const BufferAllocation::Slice> args,
           std::shared_ptr<se::dnn::LazyDnnGraph> graph);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

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
  using Stream = CustomCallThunk::Stream;
  using CustomCallTarget = CustomCallThunk::CustomCallTarget;
  using AttributesMap = CustomCallThunk::AttributesMap;

  // This is a legacy custom call API that is discouraged, and will be
  // deprecated once XLA:FFI mechanism is ready.
  //
  // TODO(b/323534971): We have an ODR violation somewhere in Tensorflow/XLA and
  // include this header with different set of defines and CustomCallTarget
  // has different meaning in different translation units. We need to get rid of
  // GOOGLE_CUDA defines all over XLA to fix this! As a workaround just keep
  // constructor in a header file.
  CustomCallCmd(ExecutionStreamId execution_stream_id,
                CustomCallTarget call_target,
                std::vector<std::optional<Slice>> operands,
                std::vector<std::optional<Slice>> results,
                absl::string_view opaque)
      : CommandBufferCmd(execution_stream_id),
        call_target_(std::move(call_target)),
        opaque_(opaque),
        operands_(std::move(operands)),
        results_(std::move(results)) {}

  CustomCallCmd(ExecutionStreamId execution_stream_id, XLA_FFI_Handler* handler,
                std::vector<std::optional<Slice>> operands,
                std::vector<std::optional<Slice>> results,
                AttributesMap attributes,
                const HloComputation* called_computation)
      : CommandBufferCmd(execution_stream_id),
        handler_(handler),
        attributes_(std::move(attributes)),
        called_computation_(called_computation),
        operands_(std::move(operands)),
        results_(std::move(results)) {}

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;
  bool IsNestedCommandBuffer() const final { return true; }

 private:
  absl::Status RecordLegacyCustomCall(const Thunk::ExecuteParams& execute_param,
                                      const RecordParams& record_params,
                                      se::CommandBuffer* command_buffer);
  absl::Status RecordXlaFfiCall(const Thunk::ExecuteParams& execute_param,
                                const RecordParams& record_params,
                                se::CommandBuffer* command_buffer);

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
// BarrierCmd insert a barrier from the execution scope created from the
// 'from_stream_id' to the execution scope created from the
// 'execution_stream_id', e.g. Async operator lowered to command buffer requires
// a barrier from the launching stream to the async operator's execution stream.
//===----------------------------------------------------------------------===//

class BarrierCmd : public CommandBufferCmd {
 public:
  BarrierCmd(ExecutionStreamId execution_stream_id,
             ExecutionStreamId from_stream_id);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  ExecutionStreamId from_stream_id_;
};

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

class CollectiveCmd : public CommandBufferCmd {
 public:
  CollectiveCmd(ExecutionStreamId execution_stream_id,
                ExecutionStreamId async_from_stream_id, NcclApi* nccl_api,
                NcclCollectiveConfig config);

  absl::Status Prepare(const Thunk::PrepareParams& params,
                       Thunk::ResourceRequests& resource_requests) final;

  bool force_update() override { return true; }

  bool IsNestedCommandBuffer() const final { return true; }

  absl::Status AddTracedCommandBuffer(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, se::CommandBuffer* command_buffer,
      absl::FunctionRef<absl::Status(se::Stream*)> trace);

  virtual AsyncStreamKind GetAsyncStreamKind() = 0;

  bool IsAsync() const {
    return async_from_stream_id_ != execution_stream_id();
  }

  NcclStreamId nccl_stream_id() {
    return xla::gpu::GetStreamId(IsAsync(), GetAsyncStreamKind());
  }

  ExecutionStreamId async_from_stream_id() const {
    return async_from_stream_id_;
  }

  absl::Status BarrierIfAsync(
      se::CommandBuffer* command_buffer, se::StreamExecutor* executor,
      const CommandBufferCmd::RecordParams& record_params);

 protected:
  NcclApi* nccl_api() const { return nccl_api_; }
  const NcclCollectiveConfig& config() const { return config_; }

 private:
  ExecutionStreamId async_from_stream_id_;
  NcclApi* nccl_api_;
  NcclCollectiveConfig config_;
};

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

class AllReduceCmd : public CollectiveCmd {
 public:
  AllReduceCmd(ExecutionStreamId execution_stream_id,
               ExecutionStreamId async_from_stream_id, NcclApi* nccl_api,
               NcclCollectiveConfig config, ReductionKind reduction_kind,
               absl::Span<const NcclCollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

 private:
  ReductionKind reduction_kind_;
  std::vector<NcclCollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// ReduceScatterCmd
//===----------------------------------------------------------------------===//

class ReduceScatterCmd : public CollectiveCmd {
 public:
  ReduceScatterCmd(ExecutionStreamId execution_stream_id,
                   ExecutionStreamId async_from_stream_id, NcclApi* nccl_api,
                   NcclCollectiveConfig config, ReductionKind reduction_kind,
                   absl::Span<const NcclCollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

 private:
  ReductionKind reduction_kind_;
  std::vector<NcclCollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

class AllGatherCmd : public CollectiveCmd {
 public:
  AllGatherCmd(ExecutionStreamId execution_stream_id,
               ExecutionStreamId async_from_stream_id, NcclApi* nccl_api,
               NcclCollectiveConfig config,
               absl::Span<const NcclCollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

 private:
  std::vector<NcclCollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// CollectiveBroadcastCmd
//===----------------------------------------------------------------------===//

class CollectiveBroadcastCmd : public CollectiveCmd {
 public:
  CollectiveBroadcastCmd(ExecutionStreamId execution_stream_id,
                         ExecutionStreamId async_from_stream_id,
                         NcclApi* nccl_api, NcclCollectiveConfig config,
                         absl::Span<const NcclCollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  std::vector<NcclCollectiveThunk::Buffer> buffers_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_
