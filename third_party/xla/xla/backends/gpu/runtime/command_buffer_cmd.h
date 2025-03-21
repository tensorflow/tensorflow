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
#include <utility>
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

namespace xla::gpu {

// clang-format off
#define COMMAND_BUFFER_CMD_LIST(V)                       \
  V(kComputationIdCmd, "ComputationIdCmd")               \
  V(kLaunchCmd, "LaunchCmd")                             \
  V(kCustomKernelLaunchCmd, "CustomKernelLaunchCmd")     \
  V(kCublasLtCmd, "CublasLtCmd")                         \
  V(kCuDnnCmd, "CuDnnCmd")                               \
  V(kGemmCmd, "GemmCmd")                                 \
  V(kMemcpyDeviceToDeviceCmd, "MemcpyDeviceToDeviceCmd") \
  V(kMemzeroCmd, "MemzeroCmd")                           \
  V(kMemset32Cmd, "Memset32Cmd")                         \
  V(kIfCmd, "IfCmd")                                     \
  V(kIfElseCmd, "IfElseCmd")                             \
  V(kCaseCmd, "CaseCmd")                                 \
  V(kForCmd, "ForCmd")                                   \
  V(kWhileCmd, "WhileCmd")                               \
  V(kCustomCallCmd, "CustomCallCmd")                     \
  V(kBarrierCmd, "BarrierCmd")                           \
  V(kEmptyCmd, "EmptyCmd")                               \
  V(kCollectiveCmd, "CollectiveCmd")                     \
  V(kAllReduceCmd, "AllReduceCmd")                       \
  V(kReduceScatter, "ReduceScatterCmd")                  \
  V(kAllToAll, "AllToAllCmd")                            \
  V(kAllGatherCmd, "AllGatherCmd")                       \
  V(kCollectiveBroadcastCmd, "CollectiveBroadcastCmd")   \
  V(kDynamicSliceFusionCmd, "DynamicSliceFusionCmd")     \
  V(kTracedCommandBufferCmd, "TracedCommandBufferCmd")   \
  V(kChildCmd, "ChildCmd")                               \
  V(kSetForConditionCmd, "SetForConditionCmd")           \
  V(kSetWhileConditionCmd, "SetWhileConditionCmd")       \
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
  using CommandBufferNodeHandle = se::CommandBuffer::NodeHandle;
  using CommandBufferConditionalHandle = se::CommandBuffer::ConditionalHandle;
  using Dependencies = se::CommandBuffer::Dependencies;
  using DependencyCmdSet = absl::flat_hash_set<const CommandBufferCmd*>;

  static std::string DependencySetToString(const DependencyCmdSet& set) {
    std::string results;
    std::vector<std::string> strs;
    for (const auto& cmd : set) {
      strs.push_back(absl::StrFormat("%p", static_cast<const void*>(cmd)));
    }
    absl::StrAppend(&results, "DependencyCmdSet: {", absl::StrJoin(strs, ", "),
                    "}");
    return results;
  }

  explicit CommandBufferCmd(CommandBufferCmdType cmd_type)
      : cmd_type_(cmd_type) {}
  virtual ~CommandBufferCmd() = default;

  using BufferUseVector = absl::InlinedVector<BufferUse, 4>;

  std::string BufferUseVectorToString(const BufferUseVector& buffer_uses) {
    std::string results;
    for (const auto& buffer_use : buffer_uses) {
      absl::StrAppend(&results, buffer_use.ToString(), ", ");
    }
    return results;
  }

  virtual std::unique_ptr<CommandBufferCmd> Clone() const = 0;
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
  };

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

  // Records command into the command buffer using given execution scope.
  virtual absl::Status Record(const Thunk::ExecuteParams& execute_params,
                              const RecordParams& record_params,
                              se::CommandBuffer* command_buffer,
                              bool create) = 0;

  // Returns the leaf nodes of the command. A single command can create multiple
  // nodes in the command buffer graph, e.g. a WhileCmd will create a kernel
  // launch node (to set the initial value of the loop variable) followed by a
  // condition node. Leaf nodes are the nodes that no other nodes for current
  // command depends on.
  virtual std::vector<CommandBufferNodeHandle> leaf_nodes() const = 0;

  DependencyCmdSet dependencies() const { return dependencies_; }
  void add_dependency(const CommandBufferCmd* cmd) {
    dependencies_.insert(cmd);
  }

  Dependencies ToDependentNodes() const {
    Dependencies nodes;
    for (const CommandBufferCmd* cmd : dependencies_) {
      auto leaf_nodes = cmd->leaf_nodes();
      for (CommandBufferNodeHandle node : leaf_nodes) {
        CHECK(node != nullptr) << "Dependency node is null";
        nodes.push_back(node);
      }
    }
    return nodes;
  }

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

  bool IsBarrier() const {
    return cmd_type_ == CommandBufferCmdType::kBarrierCmd;
  }

  bool IsCollective() const {
    return (cmd_type_ == CommandBufferCmdType::kCollectiveCmd ||
            cmd_type_ == CommandBufferCmdType::kAllReduceCmd ||
            cmd_type_ == CommandBufferCmdType::kReduceScatter ||
            cmd_type_ == CommandBufferCmdType::kAllToAll ||
            cmd_type_ == CommandBufferCmdType::kAllGatherCmd ||
            cmd_type_ == CommandBufferCmdType::kCollectiveBroadcastCmd);
  }

  std::string ToString() const {
    return absl::StrFormat("%s (%p), dependencies: %s",
                           CommandBufferCmdString(cmd_type_), this,
                           DependencySetToString(dependencies_));
  }

 private:
  std::string profile_annotation_;

  // The set of command indexes that current command has dependency on.
  DependencyCmdSet dependencies_;

  CommandBufferCmdType cmd_type_;
};

//===----------------------------------------------------------------------===//
// CommandBufferCmdSequence
//===----------------------------------------------------------------------===//

// A sequence of command buffer commands that create or update a command buffer.
// You can think of CommandBufferCmdSequence as a mini interpreter whose sole
// purpose is to manipulate command buffers at run time.
class CommandBufferCmdSequence {
 public:
  enum class SynchronizationMode {
    // Adds barriers between all commands recorded into command buffer and
    // enforces completely serialized
    // execution order that matches what would happen in a ThunkSequence.
    kSerialize,

    // Relies on buffer use analysis to insert barriers only between commands
    // that have read-write conflicts into the same buffers. Conflicts are
    // detected only between commands using the same stream id, and inter-stream
    // synchronization is a user responsibility.
    kAutomatic
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
                       Thunk::ResourceRequestsInterface& resource_requests);

  // Initializes all commands added to a sequence.
  absl::Status Initialize(const Thunk::InitializeParams& params,
                          CommandBufferCmd::StateManager& state);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const CommandBufferCmd::RecordParams& record_params,
                      se::CommandBuffer* command_buffer);

  std::unique_ptr<CommandBufferCmdSequence> Clone() const;

  // Returns buffers referenced by commands in this sequence.
  const absl::flat_hash_set<BufferUse>& buffers() const;

  // Returns buffer allocations indices referenced by commands in this sequence.
  const absl::flat_hash_set<BufferAllocation::Index>& allocs_indices() const;

  bool empty() const { return commands_.empty(); }
  size_t size() const { return commands_.size(); }

  bool force_update() const {
    return absl::c_any_of(commands_,
                          [](const std::unique_ptr<CommandBufferCmd>& cmd) {
                            return cmd->force_update();
                          });
  }

  bool created() const { return created_; }

  CommandBufferCmd* get_command(size_t idx) const {
    CHECK_LT(idx, commands_.size());
    return commands_.at(idx).get();
  }

  std::string ToString() const;

 private:
  SynchronizationMode synchronization_mode_;
  std::vector<std::unique_ptr<CommandBufferCmd>> commands_;
  bool created_ = false;

  // Buffers referenced by commands in this sequence.
  absl::flat_hash_set<BufferUse> buffers_;

  // Buffer allocations indices referenced by commands in this sequence.
  absl::flat_hash_set<BufferAllocation::Index> allocs_indices_;

  // We track read and write sets of commands recorded into the command
  // sequence to detect conflicts and insert explicit barriers. These are the
  // buffer allocation slices used by commands appended since the last barrier.
  struct ReadWriteSet {
    absl::flat_hash_set<BufferAllocation::Slice> read;
    absl::flat_hash_set<BufferAllocation::Slice> write;
  };

  ReadWriteSet read_write_set_;
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
  explicit TracedCommandBufferCmd(CommandBufferCmdType cmd_type);

  // Creates a command buffer by calling a user-provided `trace` function and
  // adds it as a nested command to `command_buffer`. Traced command buffers
  // cached and reused in an instance of `TracedCommandBuffer` kept in `state`.
  absl::Status RecordTracedCommandBuffer(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, se::CommandBuffer* command_buffer,
      bool create, absl::FunctionRef<absl::Status(se::Stream*)> trace);

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

 private:
  CommandBufferNodeHandle node_;
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

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<ComputationIdCmd>(dest_, kind_);
  }

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice dest_;
  Kind kind_;

  // TODO(ezhulenev): This is a workaround for CUDA graphs + conditional nodes
  // bug that will be fixed in CUDA 12.4.1 release: currently it's impossible to
  // update a memset node inside a conditional graph. Instead of using memset
  // node we replace it with a kernel launch node of CUDA kernels doing 1D
  // memset. This should be removed when bug is fixed in CUDA.
  std::unique_ptr<se::Kernel> memset_kernel_;

  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// ChildCmd
//===----------------------------------------------------------------------===//

class ChildCmd : public CommandBufferCmd {
 public:
  ChildCmd(std::unique_ptr<CommandBufferCmdSequence> child_cmds);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<ChildCmd>(child_cmds_->Clone());
  }

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  BufferUseVector buffers() override;

  bool force_update() override { return child_cmds_->force_update(); }

 private:
  std::unique_ptr<CommandBufferCmdSequence> child_cmds_;
  std::unique_ptr<se::CommandBuffer> child_command_buffer_;
  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

class LaunchCmd : public CommandBufferCmd {
 public:
  LaunchCmd(std::string kernel_name,
            absl::Span<const BufferAllocation::Slice> args,
            absl::Span<const BufferUse::MemoryAccess> args_access,
            LaunchDimensions dims, int64_t shmem_bytes);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<LaunchCmd>(kernel_name_, args_, args_access_, dims_,
                                       shmem_bytes_);
  }

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  BufferUseVector buffers() override;

 private:
  std::string kernel_name_;
  std::vector<BufferAllocation::Slice> args_;
  std::vector<BufferUse::MemoryAccess> args_access_;
  LaunchDimensions dims_;
  int64_t shmem_bytes_;

  // Command sequence can be recorded concurrently for multiple command buffers
  // on different stream executors and we need to synchronize mutable state.
  std::unique_ptr<se::Kernel> kernel_;
  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// CustomKenelLaunchCmd
//===----------------------------------------------------------------------===//

class CustomKernelLaunchCmd : public CommandBufferCmd {
 public:
  CustomKernelLaunchCmd(absl::Span<const BufferAllocation::Slice> args,
                        absl::Span<const BufferUse::MemoryAccess> args_access,
                        CustomKernel custom_kernel);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<CustomKernelLaunchCmd>(args_, args_access_,
                                                   custom_kernel_);
  }

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  BufferUseVector buffers() override;

 private:
  std::vector<BufferAllocation::Slice> args_;
  std::vector<BufferUse::MemoryAccess> args_access_;
  CustomKernel custom_kernel_;
  std::unique_ptr<se::Kernel> kernel_;
  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// MemcpyDeviceToDeviceCmd
//===----------------------------------------------------------------------===//

class MemcpyDeviceToDeviceCmd : public CommandBufferCmd {
 public:
  MemcpyDeviceToDeviceCmd(BufferAllocation::Slice dst,
                          BufferAllocation::Slice src, int64_t num_bytes);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<MemcpyDeviceToDeviceCmd>(dst_, src_, num_bytes_);
  }

  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice dst_;
  BufferAllocation::Slice src_;
  int64_t num_bytes_;
  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

class MemzeroCmd : public CommandBufferCmd {
 public:
  explicit MemzeroCmd(BufferAllocation::Slice dst);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  BufferUseVector buffers() override;
  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<MemzeroCmd>(dst_);
  }

 private:
  BufferAllocation::Slice dst_;
  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

class Memset32Cmd : public CommandBufferCmd {
 public:
  Memset32Cmd(BufferAllocation::Slice dst, uint32_t bit_pattern);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  BufferUseVector buffers() override;
  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<Memset32Cmd>(dst_, bit_pattern_);
  }

 private:
  BufferAllocation::Slice dst_;
  uint32_t bit_pattern_;
  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// IfCmd
//===----------------------------------------------------------------------===//

// Adds a conditional command that will execute `then_commands` if `pred`
// value is `true`.
class IfCmd : public CommandBufferCmd {
 public:
  IfCmd(BufferAllocation::Slice pred,
        std::unique_ptr<CommandBufferCmdSequence> then_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<IfCmd>(pred_, then_commands_->Clone());
  }

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{then_cond_node_};
  }

  bool force_update() override;
  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice pred_;
  std::unique_ptr<CommandBufferCmdSequence> then_commands_;
  CommandBufferConditionalHandle then_cond_handle_;
  CommandBufferNodeHandle set_cond_handle_kernel_node_;
  CommandBufferNodeHandle then_cond_node_;
  std::unique_ptr<se::CommandBuffer> then_command_buffer_;
};

//===----------------------------------------------------------------------===//
// IfElseCmd
//===----------------------------------------------------------------------===//

// Adds a conditional command that will execute `then_commands` if `pred`
// value is `true`, or `else_commands` if `pred` is `false`.
class IfElseCmd : public CommandBufferCmd {
 public:
  IfElseCmd(BufferAllocation::Slice pred,
            std::unique_ptr<CommandBufferCmdSequence> then_commands,
            std::unique_ptr<CommandBufferCmdSequence> else_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<IfElseCmd>(pred_, then_commands_->Clone(),
                                       else_commands_->Clone());
  }

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{then_cond_node_,
                                                else_cond_node_};
  }

  bool force_update() override;
  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice pred_;
  std::unique_ptr<CommandBufferCmdSequence> then_commands_;
  std::unique_ptr<CommandBufferCmdSequence> else_commands_;

  CommandBufferConditionalHandle then_cond_handle_;
  CommandBufferConditionalHandle else_cond_handle_;
  CommandBufferNodeHandle set_cond_handle_kernel_node_;

  CommandBufferNodeHandle then_cond_node_;
  CommandBufferNodeHandle else_cond_node_;

  std::unique_ptr<se::CommandBuffer> then_command_buffer_;
  std::unique_ptr<se::CommandBuffer> else_command_buffer_;
};

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

// Adds a conditional command that will execute the commands in
// `branches_commands` at `index`. If `index` is out of range, then it will run
// the commands in `branches_commands.back()`.
//
// See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#case
class CaseCmd : public CommandBufferCmd {
 public:
  static constexpr size_t kBranchBatchSize = 8;
  CaseCmd(
      BufferAllocation::Slice cond_alloc_slice, bool index_is_bool,
      std::vector<std::unique_ptr<CommandBufferCmdSequence>> branches_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    std::vector<std::unique_ptr<CommandBufferCmdSequence>>
        cloned_branches_commands;
    for (const auto& branch : branches_commands_) {
      cloned_branches_commands.push_back(branch->Clone());
    }
    return std::make_unique<CaseCmd>(index_, index_is_bool_,
                                     std::move(cloned_branches_commands));
  }

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return cond_nodes_;
  }

  bool force_update() override;

  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice index_;
  bool index_is_bool_;
  std::vector<std::unique_ptr<CommandBufferCmdSequence>> branches_commands_;

  std::vector<CommandBufferConditionalHandle> case_branch_handles_;
  std::vector<CommandBufferNodeHandle> set_case_handle_kernel_nodes_;
  std::vector<CommandBufferNodeHandle> cond_nodes_;
  std::vector<std::unique_ptr<se::CommandBuffer>> branch_command_buffers_;
};

//===----------------------------------------------------------------------===//
// SetForConditionCmd
// Internal command, not used externally.
// Set the condition handle before running conditional node
//===----------------------------------------------------------------------===//

class SetForConditionCmd : public CommandBufferCmd {
 public:
  SetForConditionCmd(CommandBufferConditionalHandle* cond_handle,
                     BufferAllocation::Slice loop_counter,
                     int32_t num_iterations);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }
  BufferUseVector buffers() override;

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<SetForConditionCmd>(cond_handle_, loop_counter_,
                                                num_iterations_);
  }

 private:
  CommandBufferConditionalHandle* cond_handle_;
  BufferAllocation::Slice loop_counter_;
  int32_t num_iterations_;

  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// ForCmd
//===----------------------------------------------------------------------===//

// Adds a conditional command that will execute `body_commands` exactly
// `num_iterations` times. This means the condition is known at compile time
// (`num_iterations` < `loop_counter`), and does not require a condition.
class ForCmd : public CommandBufferCmd {
 public:
  ForCmd(int32_t num_iterations, BufferAllocation::Slice loop_counter,
         std::unique_ptr<CommandBufferCmdSequence> body_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<ForCmd>(num_iterations_, loop_counter_,
                                    body_commands_->Clone());
  }

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{cond_node_};
  }
  bool force_update() override;
  BufferUseVector buffers() override;

 private:
  int32_t num_iterations_;
  BufferAllocation::Slice loop_counter_;
  std::unique_ptr<CommandBufferCmdSequence> body_commands_;
  std::unique_ptr<CommandBufferCmdSequence> body_and_predict_commands_;

  // First node: memset loop counter to 0
  CommandBufferNodeHandle initialize_counter_node_;

  // Second node: set the condition handle before running for loop
  CommandBufferConditionalHandle cond_handle_;
  CommandBufferCmd::CommandBufferNodeHandle set_cond_handle_node_;

  // body_commands appended with LaunchSetForConditionKernelCmd
  // CommandBufferCmdSequence body_and_predict_commands_;

  // Third node: the conditional node that will run the for loop
  CommandBufferNodeHandle cond_node_;

  std::unique_ptr<se::CommandBuffer> body_command_buffer_;
};

//===----------------------------------------------------------------------===//
// SetWhileConditionCmd
//===----------------------------------------------------------------------===//

class SetWhileConditionCmd : public CommandBufferCmd {
 public:
  SetWhileConditionCmd(CommandBufferConditionalHandle* cond_handle,
                       BufferAllocation::Slice pred);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create);

  BufferUseVector buffers() override;
  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<SetWhileConditionCmd>(cond_handle_, pred_);
  }

 private:
  CommandBufferConditionalHandle* cond_handle_;
  BufferAllocation::Slice pred_;
  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

// Adds a conditional command that will execute `body_commands` while `pred`
// value is `true`.
//
// The condition is updated by `cond_commands` and the value of `pred` is
// continuously updated by `cond_commands`.
//
// See: https://github.com/openxla/stablehlo/blob/main/docs/spec.md#while
// In pseudocode:
//
//   cond_commands()
//   while(pred):
//     body_commands()
//     cond_commands()
//
class WhileCmd : public CommandBufferCmd {
 public:
  WhileCmd(BufferAllocation::Slice pred,
           std::unique_ptr<CommandBufferCmdSequence> cond_commands,
           std::unique_ptr<CommandBufferCmdSequence> body_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<WhileCmd>(pred_, cond_commands_->Clone(),
                                      body_commands_->Clone());
  }

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{cond_node_};
  }
  bool force_update() override;

  BufferUseVector buffers() override;

 private:
  BufferAllocation::Slice pred_;
  std::unique_ptr<CommandBufferCmdSequence> cond_commands_;
  std::unique_ptr<CommandBufferCmdSequence> body_commands_;

  // cond_commands_ +  set_while_condition_kernel command;
  CommandBufferCmdSequence initialize_commands_;

  // body_commands_ + cond_commands_ +  set_while_condition_kernel command;
  CommandBufferCmdSequence loop_commands_;

  CommandBufferConditionalHandle cond_handle_;
  CommandBufferNodeHandle initialize_while_handle_node_;

  CommandBufferNodeHandle cond_node_;

  // created from predict_commands_
  std::unique_ptr<se::CommandBuffer> initialize_command_buffer_;

  // created from loop_commands_
  std::unique_ptr<se::CommandBuffer> loop_command_buffer_;
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

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<GemmCmd>(config_, lhs_buffer_, rhs_buffer_,
                                     output_buffer_, workspace_,
                                     deterministic_);
  }

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
  CublasLtCmd(GemmConfig gemm_config, se::gpu::BlasLt::Epilogue epilogue,
              int64_t algorithm_idx, BufferAllocation::Slice a_buffer,
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

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<CublasLtCmd>(
        gemm_config_, epilogue_, algorithm_idx_, a_buffer_, b_buffer_,
        c_buffer_, d_buffer_, bias_buffer_, aux_buffer_, a_scale_buffer_,
        b_scale_buffer_, c_scale_buffer_, d_scale_buffer_, d_amax_buffer_,
        workspace_buffer_);
  }

  BufferUseVector buffers() override;

  bool IsNestedCommandBuffer() const final { return true; }

 private:
  absl::StatusOr<se::gpu::BlasLt::MatmulPlan*> GetMatmulPlan(
      const se::Stream* stream);

  absl::StatusOr<se::gpu::BlasLt::MatmulAlgorithm> GetMatmulAlgorithm(
      const se::Stream* stream, const se::gpu::BlasLt::MatmulPlan* plan,
      int64_t max_workspace);

  absl::flat_hash_map<const se::Stream*, se::gpu::BlasLt::MatmulPlanPtr>
      matmul_plans_cache_;

  absl::flat_hash_map<const se::gpu::BlasLt::MatmulPlan*,
                      se::gpu::BlasLt::MatmulAlgorithm>
      matmul_algorithm_cache_;

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
  CuDnnCmd(absl::Span<const BufferAllocation::Slice> args,
           std::shared_ptr<se::dnn::LazyDnnGraph> graph);

  absl::Status Initialize(const Thunk::InitializeParams& params,
                          StateManager& state) override;

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  BufferUseVector buffers() override;

  bool IsNestedCommandBuffer() const final { return true; }

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<CuDnnCmd>(args_, graph_);
  }

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
  CustomCallCmd(std::string target_name, CustomCallTarget call_target,
                std::vector<std::optional<Slice>> operands,
                std::vector<std::optional<Slice>> results,
                absl::string_view opaque)
      : CommandBufferCmd(CommandBufferCmdType::kCustomCallCmd),
        target_name_(target_name),
        call_target_(call_target),
        opaque_(opaque),
        operands_(operands),
        results_(results) {}

  CustomCallCmd(std::string target_name, XLA_FFI_Handler* handler,
                std::vector<std::optional<Slice>> operands,
                std::vector<std::optional<Slice>> results,
                AttributesMap attributes,
                const HloComputation* called_computation)
      : CommandBufferCmd(CommandBufferCmdType::kCustomCallCmd),
        target_name_(target_name),
        handler_(handler),
        attributes_(attributes),
        called_computation_(called_computation),
        operands_(operands),
        results_(results) {}

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  BufferUseVector buffers() override;
  bool IsNestedCommandBuffer() const final { return true; }
  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    if (handler_ == nullptr) {
      return std::make_unique<CustomCallCmd>(target_name_, call_target_,
                                             operands_, results_, opaque_);
    }
    return std::make_unique<CustomCallCmd>(target_name_, handler_, operands_,
                                           results_, attributes_,
                                           called_computation_);
  }

 private:
  absl::Status RecordLegacyCustomCall(const Thunk::ExecuteParams& execute_param,
                                      const RecordParams& record_params,
                                      se::CommandBuffer* command_buffer,
                                      bool create);
  absl::Status RecordXlaFfiCall(const Thunk::ExecuteParams& execute_param,
                                const RecordParams& record_params,
                                se::CommandBuffer* command_buffer, bool create);

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

  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// BarrierCmd
//===----------------------------------------------------------------------===//
class BarrierCmd : public CommandBufferCmd {
 public:
  // Creates a barrier that will synchronize with commands specified
  // by dependencies.
  explicit BarrierCmd();

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  BufferUseVector buffers() override;
  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<BarrierCmd>();
  }

 private:
  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// EmptyCmd insert an empty node that will act as dependency node
//===----------------------------------------------------------------------===//
class EmptyCmd : public CommandBufferCmd {
 public:
  // Creates a barrier that will synchronize with commands specified
  // by dependencies.
  explicit EmptyCmd(DependencyCmdSet dependencies);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  BufferUseVector buffers() override;
  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<EmptyCmd>(dependencies());
  }

 private:
  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

class CollectiveCmd : public CommandBufferCmd {
 public:
  CollectiveCmd(CommandBufferCmdType cmd_type, CollectiveConfig config);

  absl::Status Prepare(
      const Thunk::PrepareParams& params,
      Thunk::ResourceRequestsInterface& resource_requests) final;

  bool force_update() override { return true; }

  bool IsNestedCommandBuffer() const final { return true; }

  absl::Status RecordTracedCommandBuffer(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, se::CommandBuffer* command_buffer,
      bool create, absl::FunctionRef<absl::Status(se::Stream*)> trace);

  virtual AsyncStreamKind GetAsyncStreamKind() = 0;

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  CollectiveStreamId nccl_stream_id() {
    return xla::gpu::GetCollectiveStreamId(false, GetAsyncStreamKind());
  }

 protected:
  CollectiveConfig config_;

 private:
  CommandBufferNodeHandle node_;
};

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

class AllReduceCmd : public CollectiveCmd {
 public:
  AllReduceCmd(CollectiveConfig config, ReductionKind reduction_kind,
               absl::Span<const CollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  BufferUseVector buffers() override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<AllReduceCmd>(config_, reduction_kind_, buffers_);
  }

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
                   absl::Span<const CollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  BufferUseVector buffers() override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<ReduceScatterCmd>(config_, reduction_kind_,
                                              buffers_);
  }

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
              absl::Span<const CollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  BufferUseVector buffers() override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<AllToAllCmd>(config_, has_split_dimension_,
                                         buffers_);
  }

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
               absl::Span<const CollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  BufferUseVector buffers() override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<AllGatherCmd>(config_, buffers_);
  }

 private:
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// CollectiveBroadcastCmd
//===----------------------------------------------------------------------===//

class CollectiveBroadcastCmd : public CollectiveCmd {
 public:
  CollectiveBroadcastCmd(CollectiveConfig config,
                         absl::Span<const CollectiveThunk::Buffer> buffers);

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  AsyncStreamKind GetAsyncStreamKind() override {
    return AsyncStreamKind::kCollective;
  };

  BufferUseVector buffers() override;

  std::unique_ptr<CommandBufferCmd> Clone() const override {
    return std::make_unique<CollectiveBroadcastCmd>(config_, buffers_);
  }

 private:
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// DynamicSliceFusionCmd
//===----------------------------------------------------------------------===//

class DynamicSliceFusionCmd : public CommandBufferCmd {
 public:
  DynamicSliceFusionCmd(
      std::unique_ptr<CommandBufferCmdSequence> embedded_commands,
      std::vector<std::optional<BufferAllocation::Slice>> arguments,
      std::vector<std::unique_ptr<BufferAllocation>> fake_allocations,
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

  absl::Status Record(const Thunk::ExecuteParams& execute_params,
                      const RecordParams& record_params,
                      se::CommandBuffer* command_buffer, bool create) override;

  BufferUseVector buffers() override;

  bool force_update() override;

  bool IsNestedCommandBuffer() const final { return true; }

  std::vector<CommandBufferNodeHandle> leaf_nodes() const override {
    return std::vector<CommandBufferNodeHandle>{node_};
  }

  std::unique_ptr<CommandBufferCmd> Clone() const override;

 private:
  std::unique_ptr<CommandBufferCmdSequence> embedded_commands_;
  std::vector<std::optional<BufferAllocation::Slice>> arguments_;
  std::vector<std::unique_ptr<BufferAllocation>> fake_allocations_;
  std::vector<std::optional<std::vector<DynamicSliceThunk::Offset>>> offsets_;
  std::vector<std::optional<Shape>> orig_shapes_;
  std::vector<std::optional<Shape>> sliced_shapes_;
  std::vector<std::optional<uint64_t>> offset_byte_sizes_;

  std::vector<DynamicSliceThunk::SliceDef> slices_;

  // Pinned host memory for transferring offset values from device to host.

  std::unique_ptr<se::MemoryAllocation> offsets_alloc_;

  // Pre-computed size requirement for `offsets_allocs_`.
  int64_t offsets_allocs_size_ = 0;

  // A mapping from argument index to the base offset in the `offsets_allocs_`.
  std::vector<int64_t> offsets_allocs_base_;

  // mapping from original allocation index to allocation index of embedded
  // command sequences.
  absl::flat_hash_map<int64_t, std::optional<BufferAllocation::Slice>>
      embeded_to_origin_slice_map_;

  CommandBufferNodeHandle node_;
  std::unique_ptr<se::CommandBuffer> child_command_buffer_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_
