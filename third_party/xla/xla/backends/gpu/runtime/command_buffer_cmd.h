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
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/traced_command.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

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

  BufferUses buffer_uses() const override;

  absl::Status WalkNested(
      absl::FunctionRef<absl::Status(Thunk*)> callback) override;

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

  BufferUses buffer_uses() const override;

  absl::Status WalkNested(
      absl::FunctionRef<absl::Status(Thunk*)> callback) override;

 private:
  BufferAllocation::Slice pred_;

  CommandExecutor cond_commands_;
  CommandExecutor body_commands_;

  std::optional<int64_t> trip_count_;
  bool enable_loop_unroll_ = false;
  bool is_unrolled_loop_ = false;
};

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

class CollectiveCmd : public Command {
 public:
  CollectiveCmd(CommandType cmd_type, CollectiveConfig config,
                CommunicationId communication_id = CommunicationId(0));

  absl::Status Prepare(const Thunk::PrepareParams& params) final;

  bool IsTracedCommand() const override { return true; }

  bool requires_initialization() const final { return true; }

  absl::StatusOr<const se::CommandBuffer::Command*> RecordTracedCommand(
      const Thunk::ExecuteParams& execute_params,
      const RecordParams& record_params, RecordAction record_action,
      se::CommandBuffer* command_buffer,
      absl::FunctionRef<absl::Status(se::Stream*)> trace);

 protected:
  const CollectiveConfig& config() const { return config_; }
  CommunicationId communication_id() const { return communication_id_; }

 private:
  CollectiveConfig config_;
  CommunicationId communication_id_;
};

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

class AllReduceCmd : public CollectiveCmd {
 public:
  AllReduceCmd(CollectiveConfig config, ReductionKind reduction_kind,
               absl::Span<const CollectiveThunk::Buffer> buffers);

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
                   absl::Span<const CollectiveThunk::Buffer> buffers);

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
              absl::Span<const CollectiveThunk::Buffer> buffers);

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
               absl::Span<const CollectiveThunk::Buffer> buffers);

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
  CollectiveBroadcastCmd(CollectiveConfig config,
                         absl::Span<const CollectiveThunk::Buffer> buffers);

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
  CollectivePermuteCmd(CollectiveConfig config, P2PConfig p2p_config,
                       absl::Span<const CollectiveThunk::Buffer> buffers);

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
          const CollectiveThunk::Buffer& buffer);

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
          const CollectiveThunk::Buffer& buffer);

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
// RaggedAllToAllCmd
//===----------------------------------------------------------------------===//

class RaggedAllToAllCmd : public CollectiveCmd {
 public:
  RaggedAllToAllCmd(RaggedAllToAllConfig ragged_all_to_all_config,
                    absl::Span<const CollectiveThunk::Buffer> buffers);

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
