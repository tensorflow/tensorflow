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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_memcpy_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/attribute_map.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_state.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/object_pool.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/shaped_slice.h"
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
#include "xla/xla_data.pb.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// TracedCommandBuffer
//===----------------------------------------------------------------------===//

// A cache for traced command buffers that will re-trace on change in buffer
// allocations that are relevant for `buffers` passed to constructor. We use a
// very simple most-recently-used cache of traced command buffers as in practice
// subsequent calls to XLA executable tend to reuse the same allocations.
class TracedCommandBuffer : public CommandState {
 public:
  explicit TracedCommandBuffer(const Command* trace_cmd,
                               Command::BufferUses buffers,
                               int64_t capacity = 16);

  // Returns cached command buffer traced using the same buffer addresses or
  // traces and caches a new command buffer using user provided callback.
  absl::StatusOr<se::CommandBuffer*> GetOrTraceCommandBuffer(
      const BufferAllocations* buffer_allocation, se::StreamExecutor* executor,
      se::Stream* stream, absl::FunctionRef<absl::Status(se::Stream*)> trace,
      se::StreamPriority priority = se::StreamPriority::Default);

 private:
  std::vector<BufferAllocation::Index> allocs_indices_;

  struct Entry {
    std::vector<se::DeviceAddressBase> recorded_allocs;
    std::unique_ptr<se::CommandBuffer> command_buffer;
  };
  const Command* trace_cmd_;
  int64_t capacity_;
  std::vector<Entry> entries_;
};

//===----------------------------------------------------------------------===//
// TracedCommandBufferCmd
//===----------------------------------------------------------------------===//

// A base class for commands implemented as tracing of stream activities.
class TracedCommandBufferCmd : public Command {
 protected:
  explicit TracedCommandBufferCmd(CommandType cmd_type);

  // Creates a command buffer by calling a user-provided `trace` function and
  // adds it as a nested command to `command_buffer`. Traced command buffers
  // cached and reused in an instance of `TracedCommandBuffer` kept in `state`.
  absl::StatusOr<const se::CommandBuffer::Command*> RecordTracedCommand(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer,
      absl::FunctionRef<absl::Status(se::Stream*)> trace);
};

//===----------------------------------------------------------------------===//
// EmptyCmd
//===----------------------------------------------------------------------===//

class EmptyCmd : public Command {
 public:
  explicit EmptyCmd();

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override { return {}; }
};

//===----------------------------------------------------------------------===//
// ComputationIdCmd (ReplicaId and PartitionId)
//===----------------------------------------------------------------------===//

class ComputationIdCmd : public Command {
 public:
  enum class Kind { kReplica, kPartition };

  ComputationIdCmd(BufferAllocation::Slice dest, Kind kind);

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

 private:
  BufferAllocation::Slice dest_;
  Kind kind_;
};

//===----------------------------------------------------------------------===//
// LaunchCmd
//===----------------------------------------------------------------------===//

class LaunchCmd : public Command {
 public:
  LaunchCmd(std::string kernel_name, absl::Span<const ShapedSlice> args,
            absl::Span<const BufferUse::MemoryAccess> args_access,
            LaunchDimensions dims, int64_t shmem_bytes,
            std::optional<stream_executor::gpu::TmaMetadata> tma_metadata =
                std::nullopt);

  absl::Status Initialize(const Thunk::InitializeParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

 private:
  std::string kernel_name_;
  std::vector<ShapedSlice> args_;
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

class CustomKernelLaunchCmd : public Command {
 public:
  CustomKernelLaunchCmd(absl::Span<const ShapedSlice> args,
                        absl::Span<const BufferUse::MemoryAccess> args_access,
                        CustomKernel custom_kernel);

  absl::Status Initialize(const Thunk::InitializeParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

 private:
  std::vector<ShapedSlice> args_;
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

class MemcpyDeviceToDeviceCmd : public Command {
 public:
  MemcpyDeviceToDeviceCmd(ShapedSlice dst, ShapedSlice src, int64_t num_bytes);

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

 private:
  ShapedSlice dst_;
  ShapedSlice src_;
  uint64_t num_bytes_;
};

//===----------------------------------------------------------------------===//
// MemzeroCmd
//===----------------------------------------------------------------------===//

class MemzeroCmd : public Command {
 public:
  explicit MemzeroCmd(ShapedSlice dst);

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

 private:
  ShapedSlice dst_;
};

//===----------------------------------------------------------------------===//
// Memset32Cmd
//===----------------------------------------------------------------------===//

class Memset32Cmd : public Command {
 public:
  Memset32Cmd(BufferAllocation::Slice dst, uint32_t bit_pattern);

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

 private:
  BufferAllocation::Slice dst_;
  uint32_t bit_pattern_;
};

//===----------------------------------------------------------------------===//
// ChildCmd
//===----------------------------------------------------------------------===//

class ChildCmd : public Command {
 public:
  explicit ChildCmd(CommandExecutor child_commands);

  absl::Status Initialize(const Thunk::InitializeParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  bool requires_initialization() override;

  bool force_update() override;

  bool support_loop_unroll() override { return false; }

  BufferUses buffer_uses() const override;

  absl::Status WalkNested(
      absl::FunctionRef<absl::Status(Command*)> callback) override;

 private:
  CommandExecutor child_commands_;
};

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

class CaseCmd : public Command {
 public:
  CaseCmd(ShapedSlice index, std::vector<CommandExecutor> branches);

  absl::Status Initialize(const Thunk::InitializeParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  bool requires_initialization() override;

  bool force_update() override;

  bool support_loop_unroll() override { return false; }

  BufferUses buffer_uses() const override;

  absl::Status WalkNested(
      absl::FunctionRef<absl::Status(Command*)> callback) override;

 private:
  ShapedSlice index_;
  bool index_is_bool_;
  std::vector<CommandExecutor> branches_;
};

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

class WhileCmd : public Command {
 public:
  WhileCmd(BufferAllocation::Slice pred, CommandExecutor cond_commands,
           CommandExecutor body_commands,
           std::optional<int64_t> trip_count = std::nullopt,
           bool enable_loop_unroll = false);

  absl::Status Initialize(const Thunk::InitializeParams& params) override;

  absl::Status Prepare(const Thunk::PrepareParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  bool requires_initialization() override;

  bool force_update() override;

  // We have not tried unrolling the loop inside another loop, so marking it
  // unsupported for now.
  bool support_loop_unroll() override { return false; }

  BufferUses buffer_uses() const override;

  absl::Status WalkNested(
      absl::FunctionRef<absl::Status(Command*)> callback) override;

 private:
  BufferAllocation::Slice pred_;

  CommandExecutor cond_commands_;
  CommandExecutor body_commands_;

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

  absl::Status Initialize(const Thunk::InitializeParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

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

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

  bool IsNestedCommandBuffer() const final { return true; }
};

//===----------------------------------------------------------------------===//
// CuDnnCmd
//===----------------------------------------------------------------------===//

class CuDnnCmd : public TracedCommandBufferCmd {
 public:
  CuDnnCmd(absl::Span<const ShapedSlice> args,
           std::shared_ptr<se::dnn::LazyDnnGraph> graph);

  absl::Status Initialize(const Thunk::InitializeParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

  bool IsNestedCommandBuffer() const final { return true; }

 private:
  std::vector<ShapedSlice> args_;
  const std::shared_ptr<se::dnn::LazyDnnGraph> graph_;
};

//===----------------------------------------------------------------------===//
// CustomCallCmd
//===----------------------------------------------------------------------===//

class CustomCallCmd : public Command {
 public:
  using CustomCallTarget = CustomCallThunk::CustomCallTarget;
  using AttributesMap = ffi::AttributesMap;

  // This is a legacy custom call API that is discouraged, and will be
  // deprecated once XLA:FFI mechanism is ready.
  CustomCallCmd(std::string target_name, CustomCallTarget call_target,
                std::vector<NullableShapedSlice> operands,
                std::vector<NullableShapedSlice> results,
                absl::string_view opaque)
      : Command(CommandType::kCustomCallCmd),
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
      : Command(CommandType::kCustomCallCmd),
        target_name_(std::move(target_name)),
        handler_(handler),
        call_frame_(std::move(call_frame)),
        execution_state_(std::move(execution_state)),
        call_frames_([this] { return call_frame_->Copy(); }),
        called_computation_(called_computation),
        operands_(std::move(operands)),
        results_(std::move(results)) {}

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;
  bool IsNestedCommandBuffer() const final { return true; }

 private:
  absl::StatusOr<const se::CommandBuffer::Command*> RecordLegacyCustomCall(
      const Thunk::ExecuteParams& execute_param,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer);

  absl::StatusOr<const se::CommandBuffer::Command*> RecordXlaFfiCall(
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

class CollectiveCmd : public AsyncStartCommand {
 public:
  CollectiveCmd(CommandType cmd_type, CollectiveConfig config,
                std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::Status Prepare(const Thunk::PrepareParams& params) final;

  bool requires_initialization() override { return true; }

  bool IsNestedCommandBuffer() const final { return true; }

  absl::StatusOr<const se::CommandBuffer::Command*> RecordTracedCommand(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer,
      absl::FunctionRef<absl::Status(se::Stream*)> trace);

  bool IsAsync() const final { return async_events_ != nullptr; }
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
// CollectiveDoneCmd
//===----------------------------------------------------------------------===//

class CollectiveDoneCmd : public AsyncDoneCommand {
 public:
  explicit CollectiveDoneCmd(
      const AsyncStartCommand* async_start,
      std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override { return {}; }

  std::shared_ptr<CollectiveThunk::AsyncEvents> async_events() const {
    return async_events_;
  }

 private:
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

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

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

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

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

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

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

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

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

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

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

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

 private:
  P2PConfig p2p_config_;
  std::vector<CollectiveThunk::Buffer> buffers_;
};

//===----------------------------------------------------------------------===//
// RecvCmd
//===----------------------------------------------------------------------===//

class RecvCmd : public CollectiveCmd {
 public:
  RecvCmd(CollectiveConfig config, P2PConfig p2p_config,
          const CollectiveThunk::Buffer& buffer,
          std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

 private:
  P2PConfig p2p_config_;
  CollectiveThunk::Buffer buffer_;
};

//===----------------------------------------------------------------------===//
// SendCmd
//===----------------------------------------------------------------------===//

class SendCmd : public CollectiveCmd {
 public:
  SendCmd(CollectiveConfig config, P2PConfig p2p_config,
          const CollectiveThunk::Buffer& buffer,
          std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

 private:
  P2PConfig p2p_config_;
  CollectiveThunk::Buffer buffer_;
};

//===----------------------------------------------------------------------===//
// DynamicSliceFusionCmd
//===----------------------------------------------------------------------===//

class DynamicSliceFusionCmd : public Command {
 public:
  DynamicSliceFusionCmd(
      CommandExecutor embedded_commands,
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

  absl::Status Initialize(const Thunk::InitializeParams& params) override;

  absl::Status Prepare(const Thunk::PrepareParams& params) final;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

  bool force_update() override { return true; }

  bool requires_initialization() override;

  bool support_loop_unroll() override { return true; }

  bool IsNestedCommandBuffer() const final { return true; }

  absl::Status WalkNested(
      absl::FunctionRef<absl::Status(Command*)> callback) override;

 private:
  CommandExecutor embedded_commands_;
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
      embedded_to_origin_slice_map_;

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
class DynamicSliceCopyFusionCmd : public Command {
 public:
  DynamicSliceCopyFusionCmd(const ShapedSlice& source_buffer,
                            const ShapedSlice& destination_buffer,
                            uint64_t mem_size,
                            DynamicMemcpyThunk::Offsets offsets);

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  bool force_update() override { return offsets_.depends_on_loop; }

  bool support_loop_unroll() override { return true; }

  BufferUses buffer_uses() const override;

 private:
  const ShapedSlice source_buffer_;
  const ShapedSlice destination_buffer_;
  uint64_t mem_size_;
  DynamicMemcpyThunk::Offsets offsets_;
};

//===----------------------------------------------------------------------===//
// RaggedAllToAllCmd
//===----------------------------------------------------------------------===//

class RaggedAllToAllCmd : public CollectiveCmd {
 public:
  RaggedAllToAllCmd(RaggedAllToAllConfig ragged_all_to_all_config,
                    absl::Span<const CollectiveThunk::Buffer> buffers,
                    std::shared_ptr<CollectiveThunk::AsyncEvents> async_events);

  absl::Status Initialize(const Thunk::InitializeParams& params) override;

  absl::StatusOr<const se::CommandBuffer::Command*> Record(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer) override;

  BufferUses buffer_uses() const override;

 private:
  RaggedAllToAllConfig ragged_all_to_all_config_;
  std::vector<CollectiveThunk::Buffer> buffers_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CMD_H_
