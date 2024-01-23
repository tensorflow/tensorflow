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

#ifndef XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_CMD_H_
#define XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_CMD_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/kernels/custom_kernel.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/nccl_all_reduce_thunk.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/thunk.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream_executor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_types.h"
#endif

namespace xla::gpu {

using OwnedKernel = std::unique_ptr<se::Kernel>;

//===----------------------------------------------------------------------===//
// CommandBufferCmd
//===----------------------------------------------------------------------===//

// CommandBufferCmd is an abstract command that creates or updates command
// buffer by recording commands into it.
//
// Command initialization and recording must be thread safe as commands can be
// recorded concurrently for multiple command buffers on different stream
// executors.
class CommandBufferCmd {
 public:
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
  using CollectiveExecuteParams = Thunk::CollectiveExecuteParams;
  using ExecutableSource = Thunk::ExecutableSource;

  // Run time parameters required for recording commands into the command
  // buffer. For example when we emit command buffer cmd sequence from an HLO
  // module, we only know the buffer slices required for HLO operations, but the
  // concrete device pointers become available only at run time.
  //
  // For allocations that performed through command buffer Allocate command, the
  // target addresses are tracked by command buffer runtime. To record command
  // that consumes buffers allocated inside command buffer, user should specify
  // the target address as se::DeviceMemoryBase{nullptr, size}.
  struct RecordParams {
    se::StreamExecutor* executor = nullptr;
    se::Stream* stream = nullptr;
    se::Stream* trace_stream = nullptr;
    const BufferAllocations* buffer_allocations = nullptr;
    const CollectiveExecuteParams* collective_params = nullptr;
  };

  // Prepares a command for recording on a given executor. We split it into a
  // separate function to allow expensive initialization (e.g. device kernel
  // loading) to happen before a command buffer thunk execution.
  virtual absl::Status Initialize(se::StreamExecutor* executor,
                                  ExecutableSource source) {
    return absl::OkStatus();
  }

  // Records command into the command buffer.
  virtual absl::Status Record(const RecordParams& params,
                              se::CommandBuffer* command_buffer) = 0;

  // Returns all buffers used by the cmd. These will be used to track cmd
  // updates, thus they need to be consistent across calls to the function.
  virtual BufferUsageVector buffers() = 0;

  // Returns true if command implemented as a nested command buffer.
  virtual bool IsNestedCommandBuffer() const { return false; }

  virtual ~CommandBufferCmd() = default;
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

  // Initialized all commands added to a sequence.
  absl::Status Initialize(se::StreamExecutor* executor,
                          CommandBufferCmd::ExecutableSource source);

  // Records all commands added to a sequence into the given command buffer.
  absl::Status Record(const CommandBufferCmd::RecordParams& params,
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
// LaunchCmd
//===----------------------------------------------------------------------===//

class LaunchCmd : public CommandBufferCmd {
 public:
  LaunchCmd(std::string kernel_name,
            absl::Span<const BufferAllocation::Slice> args,
            absl::Span<const MemoryAccess> args_access, LaunchDimensions dims,
            int64_t shmem_bytes);

  absl::Status Initialize(se::StreamExecutor* executor,
                          ExecutableSource source) override;

  absl::Status Record(const RecordParams& params,
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
  absl::flat_hash_map<se::StreamExecutor*, OwnedKernel> kernels_
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

  absl::Status Initialize(se::StreamExecutor* executor,
                          ExecutableSource source) override;

  absl::Status Record(const RecordParams& params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  std::vector<BufferAllocation::Slice> args_;
  std::vector<MemoryAccess> args_access_;
  CustomKernel custom_kernel_;

  // Command sequence can be recorded concurrently for multiple command buffers
  // on different stream executors and we need to synchronize mutable state.
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, OwnedKernel> kernels_
      ABSL_GUARDED_BY(mutex_);
};

//===----------------------------------------------------------------------===//
// MemcpyDeviceToDeviceCmd
//===----------------------------------------------------------------------===//

class MemcpyDeviceToDeviceCmd : public CommandBufferCmd {
 public:
  MemcpyDeviceToDeviceCmd(BufferAllocation::Slice dst,
                          BufferAllocation::Slice src, int64_t num_bytes);

  absl::Status Record(const RecordParams& params,
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

  absl::Status Record(const RecordParams& params,
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

  absl::Status Record(const RecordParams& params,
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

  absl::Status Initialize(se::StreamExecutor* executor,
                          ExecutableSource source) override;

  absl::Status Record(const RecordParams& params,
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

  absl::Status Initialize(se::StreamExecutor* executor,
                          ExecutableSource source) override;

  absl::Status Record(const RecordParams& params,
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

  absl::Status Initialize(se::StreamExecutor* executor,
                          ExecutableSource source) override;

  absl::Status Record(const RecordParams& params,
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

  absl::Status Initialize(se::StreamExecutor* executor,
                          ExecutableSource source) override;

  absl::Status Record(const RecordParams& params,
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

  absl::Status Initialize(se::StreamExecutor* executor,
                          ExecutableSource source) override;

  absl::Status Record(const RecordParams& params,
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
  absl::Status Record(const RecordParams& params,
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
  absl::Status Record(const RecordParams& params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

 private:
  BufferAllocation allocation_;
};

//===----------------------------------------------------------------------===//
// GemmCmd
//===----------------------------------------------------------------------===//

class GemmCmd : public CommandBufferCmd {
 public:
  GemmCmd(GemmConfig config, const BufferAllocation::Slice& lhs_buffer,
          const BufferAllocation::Slice& rhs_buffer,
          const BufferAllocation::Slice& output_buffer,
          const BufferAllocation::Slice& workspace, bool deterministic);

  absl::Status Initialize(se::StreamExecutor* executor,
                          ExecutableSource source) override;

  absl::Status Record(const RecordParams& params,
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
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  using Stream = stream_executor::gpu::GpuStreamHandle;
#else   //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  using Stream = void*;
#endif  //  GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  using CustomCallTarget = std::function<void(Stream, void**, const char*,
                                              size_t, XlaCustomCallStatus*)>;
  struct Slice {
    BufferAllocation::Slice slice;
    Shape shape;
  };

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

  Status Record(const RecordParams& params,
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
// AllReduceCmd
//===----------------------------------------------------------------------===//

class AllReduceCmd : public CommandBufferCmd {
 public:
  AllReduceCmd(NcclApi* nccl_api, NcclCollectiveConfig config,
               ReductionKind reduction_kind,
               absl::Span<const NcclCollectiveThunk::Buffer> buffers);

  absl::Status Record(const RecordParams& params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

  bool IsNestedCommandBuffer() const final { return true; }

 private:
  NcclApi* nccl_api_;
  NcclCollectiveConfig config_;
  ReductionKind reduction_kind_;
  std::vector<NcclCollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// ReduceScatterCmd
//===----------------------------------------------------------------------===//

class ReduceScatterCmd : public CommandBufferCmd {
 public:
  ReduceScatterCmd(NcclApi* nccl_api, NcclCollectiveConfig config,
                   ReductionKind reduction_kind,
                   absl::Span<const NcclCollectiveThunk::Buffer> buffers);

  absl::Status Record(const RecordParams& params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

  bool IsNestedCommandBuffer() const final { return true; }

 private:
  NcclApi* nccl_api_;
  NcclCollectiveConfig config_;
  ReductionKind reduction_kind_;
  std::vector<NcclCollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

class AllGatherCmd : public CommandBufferCmd {
 public:
  AllGatherCmd(NcclApi* nccl_api, NcclCollectiveConfig config,
               absl::Span<const NcclCollectiveThunk::Buffer> buffers);

  absl::Status Record(const RecordParams& params,
                      se::CommandBuffer* command_buffer) override;

  BufferUsageVector buffers() override;

  bool IsNestedCommandBuffer() const final { return true; }

 private:
  NcclApi* nccl_api_;
  NcclCollectiveConfig config_;
  std::vector<NcclCollectiveThunk::Buffer> buffers_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_RUNTIME3_COMMAND_BUFFER_CMD_H_
