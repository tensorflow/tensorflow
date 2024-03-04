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
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/runtime/custom_call_thunk.h"
#include "xla/service/gpu/thunk.h"
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

// CommandBufferCmd is an abstract command that creates or updates command
// buffer by recording commands into it.
//
// Commands have the same execution stages as thunks as they are executed by a
// command buffer thunk: Prepare, Initialize and Record (Execute). See Thunk
// documentation for details.
//
// Commands must be thread safe as they can be recorded into multiple command
// buffers concurrently on different stream executors.
class CommandBufferCmd {
 public:
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

  // Records command into the command buffer.
  virtual absl::Status Record(const Thunk::ExecuteParams& params,
                              StateManager& state,
                              se::CommandBuffer* command_buffer) = 0;

  // Returns all buffers used by the cmd. These will be used to track cmd
  // updates, thus they need to be consistent across calls to the function.
  virtual BufferUsageVector buffers() = 0;

  // Returns true if command implemented as a nested command buffer.
  virtual bool IsNestedCommandBuffer() const { return false; }
};

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

// A sequence of command buffer commands that create or update a command buffer.
// You can think of CommandBufferCmdSequence as a mini interpreter whose sole
// purpose is to manipulate command buffers at run time.
class CommandBufferCmdSequence {
 public:
  explicit CommandBufferCmdSequence(bool force_barriers = false);

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
  absl::Status Record(const Thunk::ExecuteParams& params,
                      CommandBufferCmd::StateManager& state,
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

 private:
  struct Command {
    Command(std::unique_ptr<CommandBufferCmd> cmd, bool requires_barrier)
        : cmd(std::move(cmd)), requires_barrier(requires_barrier) {}

    std::unique_ptr<CommandBufferCmd> cmd;
    bool requires_barrier;
  };

  // Functions for tracking buffer usage of recorded commands and figuring out
  // when the next command requires a barrier for correctness.
  bool HasConflicts(const CommandBufferCmd::BufferUsageVector& buffers);
  void TrackBuffers(const CommandBufferCmd::BufferUsageVector& buffers);
  void ClearTrackedBuffers();

  bool force_barriers_;
  std::vector<Command> commands_;

  // Buffers referenced by commands in this sequence.
  absl::flat_hash_set<CommandBufferCmd::BufferUsage> buffers_;

  // Buffer allocations indices referenced by commands in this sequence.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices_;

  // We track read and write sets of commands recorded into the command
  // sequence to detect conflicts and insert explicit barriers. These are the
  // buffer allocation slices used by commands appended since the last barrier.
  absl::flat_hash_set<BufferAllocation::Slice> read_set_;
  absl::flat_hash_set<BufferAllocation::Slice> write_set_;
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
  // Creates a command buffer by calling a user-provided `trace` function and
  // adds it as a nested command to `command_buffer`. Traced command buffers
  // cached and reused in an instance of `TracedCommandBuffer` kept in `state`.
  absl::Status AddTracedCommandBuffer(
      const Thunk::ExecuteParams& params, StateManager& state,
      se::CommandBuffer* command_buffer,
      absl::FunctionRef<absl::Status(se::Stream*)> trace);
};

//===----------------------------------------------------------------------===//
// ComputationIdCmd (ReplicaId and PartitionId)
//===----------------------------------------------------------------------===//

class ComputationIdCmd : public CommandBufferCmd {
 public:
  enum class Kind { kReplica, kPartition };

  ComputationIdCmd(BufferAllocation::Slice dest, Kind kind);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
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
  LaunchCmd(std::string kernel_name,
            absl::Span<const BufferAllocation::Slice> args,
            absl::Span<const MemoryAccess> args_access, LaunchDimensions dims,
            int64_t shmem_bytes);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
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
  CustomKernelLaunchCmd(absl::Span<const BufferAllocation::Slice> args,
                        absl::Span<const MemoryAccess> args_access,
                        CustomKernel custom_kernel);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
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
  MemcpyDeviceToDeviceCmd(BufferAllocation::Slice dst,
                          BufferAllocation::Slice src, int64_t num_bytes);

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
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
  explicit MemzeroCmd(BufferAllocation::Slice dst);

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
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
  explicit Memset32Cmd(BufferAllocation::Slice dst, uint32_t bit_pattern);

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
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
  IfCmd(BufferAllocation::Slice pred, CommandBufferCmdSequence then_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
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
  IfElseCmd(BufferAllocation::Slice pred,
            CommandBufferCmdSequence then_commands,
            CommandBufferCmdSequence else_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
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
  CaseCmd(BufferAllocation::Slice index,
          std::vector<CommandBufferCmdSequence> branches_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
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
  ForCmd(int32_t num_iterations, BufferAllocation::Slice loop_counter,
         CommandBufferCmdSequence body_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
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
  WhileCmd(BufferAllocation::Slice pred, CommandBufferCmdSequence cond_commands,
           CommandBufferCmdSequence body_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  BufferAllocation::Slice pred_;
  CommandBufferCmdSequence cond_commands_;
  CommandBufferCmdSequence body_commands_;
};

//===----------------------------------------------------------------------===//
// AllocateCmd
//===----------------------------------------------------------------------===//

class AllocateCmd : public CommandBufferCmd {
 public:
  explicit AllocateCmd(BufferAllocation allocation);

  // After calling this function, the allocated memory is tracked in
  // CommandBuffer object.
  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  BufferAllocation allocation_;
};

//===----------------------------------------------------------------------===//
// FreeCmd
//===----------------------------------------------------------------------===//

class FreeCmd : public CommandBufferCmd {
 public:
  explicit FreeCmd(BufferAllocation allocation);

  // After calling this function, the allocated memory address for dst
  // BufferAllocation is freed, no update is required.
  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  BufferAllocation allocation_;
};

//===----------------------------------------------------------------------===//
// GemmCmd
//===----------------------------------------------------------------------===//

class GemmCmd : public TracedCommandBufferCmd {
 public:
  GemmCmd(GemmConfig config, const BufferAllocation::Slice& lhs_buffer,
          const BufferAllocation::Slice& rhs_buffer,
          const BufferAllocation::Slice& output_buffer,
          const BufferAllocation::Slice& workspace, bool deterministic);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
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
// CustomCallCmd
//===----------------------------------------------------------------------===//

class CustomCallCmd : public CommandBufferCmd {
 public:
  using Slice = CustomCallThunk::Slice;
  using Stream = CustomCallThunk::Stream;
  using CustomCallTarget = CustomCallThunk::CustomCallTarget;

  // This is a legacy custom call API that is discouraged, and will be
  // deprecated once XLA:FFI mechanism is ready.
  // TODO(anlunx): Support XLA:FFI calls as commands.
  CustomCallCmd(CustomCallTarget call_target,
                std::vector<std::optional<Slice>> operands,
                std::vector<std::optional<Slice>> results,
                absl::string_view opaque)
      : call_target_(std::move(call_target)),
        operands_(std::move(operands)),
        results_(std::move(results)),
        opaque_(opaque){};

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;
  bool IsNestedCommandBuffer() const final { return true; }

 private:
  CustomCallTarget call_target_;
  std::vector<std::optional<Slice>> operands_;
  std::vector<std::optional<Slice>> results_;
  std::string opaque_;
};

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

class CollectiveCmd : public TracedCommandBufferCmd {
 public:
  CollectiveCmd(NcclApi* nccl_api, NcclCollectiveConfig config);

  absl::Status Prepare(const Thunk::PrepareParams& params,
                       Thunk::ResourceRequests& resource_requests) final;

  bool IsNestedCommandBuffer() const final { return true; }

 protected:
  NcclApi* nccl_api() const { return nccl_api_; }
  const NcclCollectiveConfig& config() const { return config_; }

 private:
  NcclApi* nccl_api_;
  NcclCollectiveConfig config_;
};

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

class AllReduceCmd : public CollectiveCmd {
 public:
  AllReduceCmd(NcclApi* nccl_api, NcclCollectiveConfig config,
               ReductionKind reduction_kind,
               absl::Span<const NcclCollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  ReductionKind reduction_kind_;
  std::vector<NcclCollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// ReduceScatterCmd
//===----------------------------------------------------------------------===//

class ReduceScatterCmd : public CollectiveCmd {
 public:
  ReduceScatterCmd(NcclApi* nccl_api, NcclCollectiveConfig config,
                   ReductionKind reduction_kind,
                   absl::Span<const NcclCollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  ReductionKind reduction_kind_;
  std::vector<NcclCollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

class AllGatherCmd : public CollectiveCmd {
 public:
  AllGatherCmd(NcclApi* nccl_api, NcclCollectiveConfig config,
               absl::Span<const NcclCollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& params, StateManager& state,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  std::vector<NcclCollectiveThunk::Buffer> buffers_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_
