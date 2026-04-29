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

#include "xla/backends/gpu/runtime/command_buffer_cmd.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/annotation.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_execution.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_loop.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/runtime/execution_graph.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/gpu/multi_gpu_barrier_kernel.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/tensor_map.h"
#include "xla/stream_executor/trace_command_buffer_factory.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/scoped_annotation.h"

namespace xla::gpu {

static absl::string_view ReductionKindString(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::MAX:
      return "max";
    case ReductionKind::MIN:
      return "min";
    case ReductionKind::PRODUCT:
      return "product";
    case ReductionKind::SUM:
      return "sum";
  }
}

// Create a callback to create a command buffer from a command sequence.
static se::CommandBuffer::CreateCommands CreateCommands(
    const CommandExecutor* commands, const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer,
             absl::Span<const se::CommandBuffer::Command* const> dependencies) {
    return commands->RecordCreate(*execute_params, *record_params,
                                  command_buffer, dependencies);
  };
}

// Create callbacks to create a command buffer from command sequences.
static std::vector<se::CommandBuffer::CreateCommands> CreateCommands(
    absl::Span<const CommandExecutor> commands,
    const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  std::vector<se::CommandBuffer::CreateCommands> create_commands;
  for (const CommandExecutor& cmd : commands) {
    create_commands.push_back(
        CreateCommands(&cmd, execute_params, record_params));
  }
  return create_commands;
}

// Create a callback to update a command buffer with command sequence.
static se::CommandBuffer::UpdateCommands UpdateCommands(
    const CommandExecutor* commands, const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  return [=](se::CommandBuffer* command_buffer) {
    return commands->RecordUpdate(*execute_params, *record_params,
                                  command_buffer);
  };
}

// Create callbacks to update a command buffer with command sequence.
static std::vector<se::CommandBuffer::UpdateCommands> UpdateCommands(
    absl::Span<const CommandExecutor> commands,
    const Thunk::ExecuteParams* execute_params,
    const Command::RecordParams* record_params) {
  std::vector<se::CommandBuffer::UpdateCommands> update_commands;
  for (const CommandExecutor& cmd : commands) {
    update_commands.push_back(
        UpdateCommands(&cmd, execute_params, record_params));
  }
  return update_commands;
}

//===----------------------------------------------------------------------===//
// Command::RecordAction helpers.
//===----------------------------------------------------------------------===//

using CreateCommand =
    absl::FunctionRef<absl::StatusOr<const se::CommandBuffer::Command*>(
        absl::Span<const se::CommandBuffer::Command* const> dependencies)>;

using UpdateCommand =
    absl::FunctionRef<absl::Status(const se::CommandBuffer::Command* command)>;

// Handles a record action by calling one of the user-provided functions.
static absl::StatusOr<const se::CommandBuffer::Command*> Handle(
    Command::RecordAction action, CreateCommand create_command,
    UpdateCommand update_command) {
  if (auto* create = std::get_if<Command::RecordCreate>(&action)) {
    return create_command(create->dependencies);
  }

  if (auto* update = std::get_if<Command::RecordUpdate>(&action)) {
    TF_RETURN_IF_ERROR(update_command(update->command));
    return update->command;
  }

  return Internal("Invalid record action");
}

//===----------------------------------------------------------------------===//
// CaseCmd
//===----------------------------------------------------------------------===//

CaseCmd::CaseCmd(ShapedSlice index, std::vector<CommandExecutor> branches)
    : Command(CommandType::kCaseCmd),
      index_(index),
      index_is_bool_(index.shape.element_type() == PRED),
      branches_(std::move(branches)) {}

absl::Status CaseCmd::Initialize(const Thunk::InitializeParams& params) {
  for (auto& branch : branches_) {
    TF_RETURN_IF_ERROR(branch.Initialize(params));
  }
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> CaseCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase index =
      execute_params.buffer_allocations->GetDeviceAddress(index_.slice);

  VLOG(5) << "CaseCmd:";
  VLOG(5) << "  index: " << index_ << " (" << index.opaque() << ")";

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        if (index_is_bool_) {
          return command_buffer->CreateCase(
              se::DeviceAddress<bool>(index),
              CreateCommands(branches_, &execute_params, &record_params),
              dependencies);
        }
        return command_buffer->CreateCase(
            se::DeviceAddress<int32_t>(index),
            CreateCommands(branches_, &execute_params, &record_params),
            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        if (index_is_bool_) {
          return command_buffer->UpdateCase(
              command, se::DeviceAddress<bool>(index),
              UpdateCommands(branches_, &execute_params, &record_params));
        }
        return command_buffer->UpdateCase(
            command, se::DeviceAddress<int32_t>(index),
            UpdateCommands(branches_, &execute_params, &record_params));
      });
}

Command::BufferUses CaseCmd::buffer_uses() const {
  return {BufferUse::Read(index_.slice, index_.shape)};
}

absl::Status CaseCmd::WalkNested(
    absl::FunctionRef<absl::Status(Thunk*)> callback) {
  for (auto& branch : branches_) {
    RETURN_IF_ERROR(branch.Walk(
        [&](Command* cmd) -> absl::Status { return callback(cmd); }));
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// WhileCmd
//===----------------------------------------------------------------------===//

WhileCmd::WhileCmd(BufferAllocation::Slice pred, CommandExecutor cond_commands,
                   CommandExecutor body_commands,
                   std::optional<int64_t> trip_count, bool enable_loop_unroll)
    : Command(CommandType::kWhileCmd),
      pred_(pred),
      cond_commands_(std::move(cond_commands)),
      body_commands_(std::move(body_commands)),
      trip_count_(trip_count),
      enable_loop_unroll_(enable_loop_unroll) {}

absl::Status WhileCmd::Initialize(const Thunk::InitializeParams& params) {
  TF_RETURN_IF_ERROR(cond_commands_.Initialize(params));
  TF_RETURN_IF_ERROR(body_commands_.Initialize(params));
  if (enable_loop_unroll_ && body_commands_.support_loop_unroll() &&
      cond_commands_.support_loop_unroll() && trip_count_.has_value()) {
    is_unrolled_loop_ = true;
  }
  VLOG(3) << "WhileCmd::Initialize: enable_loop_unroll_=" << enable_loop_unroll_
          << ", body_support=" << body_commands_.support_loop_unroll()
          << ", cond_support=" << cond_commands_.support_loop_unroll()
          << ", trip_count=" << trip_count_.value_or(-1)
          << ", is_unrolled_loop_=" << is_unrolled_loop_;
  return absl::OkStatus();
}

absl::Status WhileCmd::Prepare(const Thunk::PrepareParams& params) {
  TF_RETURN_IF_ERROR(cond_commands_.Prepare(params));
  TF_RETURN_IF_ERROR(body_commands_.Prepare(params));
  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> WhileCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase pred =
      execute_params.buffer_allocations->GetDeviceAddress(pred_);

  VLOG(5) << "WhileCmd: cond_commands=" << cond_commands_.size()
          << " body_commands=" << body_commands_.size();
  VLOG(5) << "  pred: " << pred_ << " (" << pred.opaque() << ")";
  VLOG(5) << "  trip_count: " << trip_count_.value_or(-1)
          << " (unroll: " << is_unrolled_loop_ << ")";
  if (is_unrolled_loop_) {
    // When the loop is unrolled, we need to record the body commands for
    // `trip_count` times into child_command_buffer, and implement the While
    // command as a child command.
    //
    // Unroll the while loop body for `trip_count` times.
    // Unrolled execution sequence: cond -> body -> cond -> body -> ...
    // In the unrolled pattern, we still need to run the cond commands because
    // body commands might depends on the value of index variable that is
    // updated by condition commands.

    auto record_fn =
        [&](se::CommandBuffer* child_command_buffer) -> absl::Status {
      VLOG(3) << "Recording unrolled loop with trip_count: "
              << trip_count_.value();

      Command::RecordParams new_record_params = record_params;
      std::vector<const se::CommandBuffer::Command*> dependencies;

      ScopedWhileLoop loop("record_fn", trip_count_);
      for (int64_t i = 0; i < *trip_count_; loop.IncLoopIteration(), ++i) {
        CommandExecutor::RecordId record_id(i);
        TF_ASSIGN_OR_RETURN(dependencies,
                            cond_commands_.RecordCreate(
                                execute_params, new_record_params,
                                child_command_buffer, dependencies, record_id));
        TF_ASSIGN_OR_RETURN(dependencies,
                            body_commands_.RecordCreate(
                                execute_params, new_record_params,
                                child_command_buffer, dependencies, record_id));
      }

      return absl::OkStatus();
    };

    auto update_fn =
        [&](se::CommandBuffer* child_command_buffer) -> absl::Status {
      VLOG(3) << "Updating unrolled loop with trip_count: "
              << trip_count_.value();

      Command::RecordParams new_record_params = record_params;

      ScopedWhileLoop loop("record_fn", trip_count_);
      for (int64_t i = 0; i < *trip_count_; loop.IncLoopIteration(), ++i) {
        CommandExecutor::RecordId record_id(i);
        TF_RETURN_IF_ERROR(
            cond_commands_.RecordUpdate(execute_params, new_record_params,
                                        child_command_buffer, record_id));
        TF_RETURN_IF_ERROR(
            body_commands_.RecordUpdate(execute_params, new_record_params,
                                        child_command_buffer, record_id));
      }

      return absl::OkStatus();
    };

    return Handle(
        std::move(record_action),
        [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
          return command_buffer->CreateChildCommand(record_fn, dependencies);
        },
        [&](const se::CommandBuffer::Command* command) {
          return command_buffer->UpdateChildCommand(command, update_fn);
        });
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateWhile(
            se::DeviceAddress<bool>(pred),
            CreateCommands(&cond_commands_, &execute_params, &record_params),
            CreateCommands(&body_commands_, &execute_params, &record_params),
            dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateWhile(
            command, se::DeviceAddress<bool>(pred),
            UpdateCommands(&cond_commands_, &execute_params, &record_params),
            UpdateCommands(&body_commands_, &execute_params, &record_params));
      });
}

Command::BufferUses WhileCmd::buffer_uses() const {
  return {BufferUse::Read(pred_, ShapeUtil::MakeShape(PRED, {}))};
}

absl::Status WhileCmd::WalkNested(
    absl::FunctionRef<absl::Status(Thunk*)> callback) {
  RETURN_IF_ERROR(cond_commands_.Walk(
      [&](Command* cmd) -> absl::Status { return callback(cmd); }));
  return body_commands_.Walk(
      [&](Command* cmd) -> absl::Status { return callback(cmd); });
}

//===----------------------------------------------------------------------===//
// CollectiveCmd
//===----------------------------------------------------------------------===//

CollectiveCmd::CollectiveCmd(CommandType cmd_type, CollectiveConfig config,
                             CommunicationId communication_id)
    : Command(cmd_type, se::StreamPriority::Highest),
      config_(std::move(config)),
      communication_id_(communication_id) {}

absl::Status CollectiveCmd::Prepare(const Thunk::PrepareParams& params) {
  TF_RET_CHECK(params.collective_params &&
               params.collective_params->device_assn);

  TF_ASSIGN_OR_RETURN(
      GpuCliqueKey clique_key,
      GetGpuCliqueKey(*params.collective_params, config().replica_groups,
                      config().group_mode, communication_id_));

  TF_ASSIGN_OR_RETURN(std::vector<std::vector<GlobalDeviceId>> device_groups,
                      GetParticipatingDevicesGroups(
                          *params.collective_params->device_assn,
                          config().replica_groups, config().group_mode));

  // Sort device groups: RequestClique expects pre-sorted groups.
  absl::c_for_each(device_groups, [](auto& group) { absl::c_sort(group); });
  absl::c_sort(device_groups);

  return params.collective_clique_requests->RequestClique(clique_key,
                                                          device_groups);
}

absl::StatusOr<const se::CommandBuffer::Command*>
CollectiveCmd::RecordTracedCommand(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer,
    absl::FunctionRef<absl::Status(se::Stream*)> trace) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::CommandBuffer> nested_cmd,
                      se::TraceCommandBufferFactory::Create(
                          execute_params.stream->parent(),
                          execute_params.command_buffer_trace_stream, trace));

  if (priority() != se::StreamPriority::Default) {
    TF_RETURN_IF_ERROR(nested_cmd->SetPriority(priority()));
  }

  return Handle(
      std::move(record_action),
      [&](absl::Span<const se::CommandBuffer::Command* const> dependencies) {
        return command_buffer->CreateChildCommand(*nested_cmd, dependencies);
      },
      [&](const se::CommandBuffer::Command* command) {
        return command_buffer->UpdateChildCommand(command, *nested_cmd);
      });
}

//===----------------------------------------------------------------------===//
// AllReduceCmd
//===----------------------------------------------------------------------===//

AllReduceCmd::AllReduceCmd(CollectiveConfig config,
                           ReductionKind reduction_kind,
                           absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandType::kAllReduceCmd, std::move(config)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> AllReduceCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "AllReduceCmd: reduction=" << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllReduceCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                      GetGpuCliqueKey(*execute_params.collective_params,
                                      config().replica_groups,
                                      config().group_mode, communication_id()));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunAllReduce(reduction_kind_, device_buffers, *stream, *comm,
                            config().use_symmetric_buffer);
      });
}

Command::BufferUses AllReduceCmd::buffer_uses() const {
  BufferUses buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// ReduceScatterCmd
//===----------------------------------------------------------------------===//

ReduceScatterCmd::ReduceScatterCmd(
    CollectiveConfig config, ReductionKind reduction_kind,
    absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandType::kReduceScatterCmd, std::move(config)),
      reduction_kind_(reduction_kind),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> ReduceScatterCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "ReduceScatterCmd: reduction=" << ReductionKindString(reduction_kind_);

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "ReduceScatterCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                      GetGpuCliqueKey(*execute_params.collective_params,
                                      config().replica_groups,
                                      config().group_mode, communication_id()));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(execute_params, record_params, record_action,
                             command_buffer, [&](se::Stream* stream) {
                               return RunReduceScatter(
                                   reduction_kind_, device_buffers, *stream,
                                   *comm, config().use_symmetric_buffer);
                             });
}

Command::BufferUses ReduceScatterCmd::buffer_uses() const {
  BufferUses buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// AllToAllCmd
//===----------------------------------------------------------------------===//

AllToAllCmd::AllToAllCmd(CollectiveConfig config, bool has_split_dimension,
                         absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandType::kAllToAllCmd, std::move(config)),
      has_split_dimension_(has_split_dimension),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> AllToAllCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "AllToAllCmd, has_split_dimension=" << has_split_dimension_;

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllToAllCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                      GetGpuCliqueKey(*execute_params.collective_params,
                                      config().replica_groups,
                                      config().group_mode, communication_id()));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  // Memcpy case is not currently supported in CommandBuffer.
  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunAllToAll(has_split_dimension_, device_buffers, *stream, *comm,
                           config().use_symmetric_buffer);
      });
}

Command::BufferUses AllToAllCmd::buffer_uses() const {
  BufferUses buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// AllGatherCmd
//===----------------------------------------------------------------------===//

AllGatherCmd::AllGatherCmd(CollectiveConfig config,
                           absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandType::kAllGatherCmd, std::move(config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> AllGatherCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "AllGatherCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "AllGatherCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                      GetGpuCliqueKey(*execute_params.collective_params,
                                      config().replica_groups,
                                      config().group_mode, communication_id()));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunAllGather(device_buffers, *stream, *comm,
                            config().use_symmetric_buffer);
      });
}

Command::BufferUses AllGatherCmd::buffer_uses() const {
  BufferUses buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectiveBroadcastCmd
//===----------------------------------------------------------------------===//

CollectiveBroadcastCmd::CollectiveBroadcastCmd(
    CollectiveConfig config, absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandType::kCollectiveBroadcastCmd, std::move(config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*>
CollectiveBroadcastCmd::Record(const Thunk::ExecuteParams& execute_params,
                               const RecordParams& record_params,
                               RecordAction record_action,
                               se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "CollectiveBroadcastCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "CollectiveBroadcastCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                      GetGpuCliqueKey(*execute_params.collective_params,
                                      config().replica_groups,
                                      config().group_mode, communication_id()));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunCollectiveBroadcast(device_buffers, *stream, *comm);
      });
}

Command::BufferUses CollectiveBroadcastCmd::buffer_uses() const {
  BufferUses buffer_usage;
  for (auto& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// RecvCmd
//===----------------------------------------------------------------------===//

RecvCmd::RecvCmd(CollectiveConfig config, P2PConfig p2p_config,
                 const CollectiveThunk::Buffer& buffer)
    : CollectiveCmd(CommandType::kRecvCmd, std::move(config),
                    CommunicationId(1)),
      p2p_config_(std::move(p2p_config)),
      buffer_(buffer) {}

absl::StatusOr<const se::CommandBuffer::Command*> RecvCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  DeviceBufferPair device_buffer_pair{
      config().operand_element_type[0],
      buffer_.element_count,
      execute_params.buffer_allocations->GetDeviceAddress(
          buffer_.source_buffer.slice),
      execute_params.buffer_allocations->GetDeviceAddress(
          buffer_.destination_buffer.slice),
      buffer_.source_memory_space,
      buffer_.destination_memory_space};

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "RecvCmd:";

  XLA_VLOG_DEVICE(5, device_ordinal)
      << "  Src: " << buffer_.source_buffer << " ("
      << device_buffer_pair.source_buffer.opaque() << ")";
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "  Dst: " << buffer_.destination_buffer << " ("
      << device_buffer_pair.destination_buffer.opaque() << ")";

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "RecvCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                      GetGpuCliqueKey(*execute_params.collective_params,
                                      config().replica_groups,
                                      config().group_mode, communication_id()));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  GlobalDeviceId global_device_id =
      execute_params.collective_params->global_device_id;

  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID current_logical_id,
      execute_params.collective_params->device_assn->LogicalIdForDevice(
          global_device_id));

  const int64_t current_id =
      config().group_mode ==
              CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;
  std::string device_string =
      CollectiveThunk::GetDeviceString(*execute_params.collective_params);

  P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(p2p_config_.id_to_source_target, current_id);

  const std::optional<int64_t> source_id = source_target.source;
  std::function<absl::Status(se::Stream*)> trace = [&](se::Stream* stream) {
    return RunRecv(device_buffer_pair, *stream, *comm, current_id, source_id,
                   device_string);
  };

  return RecordTracedCommand(execute_params, record_params,
                             std::move(record_action), command_buffer, trace);
}

Command::BufferUses RecvCmd::buffer_uses() const {
  BufferUses buffer_usage;
  buffer_usage.emplace_back(BufferUse::Read(buffer_.source_buffer.slice,
                                            buffer_.source_buffer.shape));
  buffer_usage.emplace_back(BufferUse::Write(buffer_.destination_buffer.slice,
                                             buffer_.destination_buffer.shape));
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// SendCmd
//===----------------------------------------------------------------------===//

SendCmd::SendCmd(CollectiveConfig config, P2PConfig p2p_config,
                 const CollectiveThunk::Buffer& buffer)
    : CollectiveCmd(CommandType::kSendCmd, std::move(config),
                    CommunicationId(1)),
      p2p_config_(std::move(p2p_config)),
      buffer_(buffer) {}

absl::StatusOr<const se::CommandBuffer::Command*> SendCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  DeviceBufferPair device_buffer_pair{
      config().operand_element_type[0],
      buffer_.element_count,
      execute_params.buffer_allocations->GetDeviceAddress(
          buffer_.source_buffer.slice),
      execute_params.buffer_allocations->GetDeviceAddress(
          buffer_.destination_buffer.slice),
      buffer_.source_memory_space,
      buffer_.destination_memory_space};

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "SendCmd:";

  XLA_VLOG_DEVICE(5, device_ordinal)
      << "  Src: " << buffer_.source_buffer << " ("
      << device_buffer_pair.source_buffer.opaque() << ")";
  XLA_VLOG_DEVICE(5, device_ordinal)
      << "  Dst: " << buffer_.destination_buffer << " ("
      << device_buffer_pair.destination_buffer.opaque() << ")";

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "SendCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                      GetGpuCliqueKey(*execute_params.collective_params,
                                      config().replica_groups,
                                      config().group_mode, communication_id()));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  GlobalDeviceId global_device_id =
      execute_params.collective_params->global_device_id;

  TF_ASSIGN_OR_RETURN(
      const DeviceAssignment::LogicalID current_logical_id,
      execute_params.collective_params->device_assn->LogicalIdForDevice(
          global_device_id));

  const int64_t current_id =
      config().group_mode ==
              CollectiveOpGroupMode::COLLECTIVE_OP_GROUP_MODE_CROSS_REPLICA
          ? current_logical_id.replica_id
          : current_logical_id.computation_id;
  std::string device_string =
      CollectiveThunk::GetDeviceString(*execute_params.collective_params);

  P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(p2p_config_.id_to_source_target, current_id);

  std::optional<int64_t> target_id = source_target.target;
  if (!target_id) {
    VLOG(3) << "[" << device_ordinal << "] Skipping Send";
    return nullptr;
  }

  std::function<absl::Status(se::Stream*)> trace = [&](se::Stream* stream) {
    return RunSend(device_buffer_pair, *stream, *comm, current_id, *target_id,
                   device_string);
  };

  return RecordTracedCommand(execute_params, record_params,
                             std::move(record_action), command_buffer, trace);
}

Command::BufferUses SendCmd::buffer_uses() const {
  BufferUses buffer_usage;
  buffer_usage.emplace_back(BufferUse::Read(buffer_.source_buffer.slice,
                                            buffer_.source_buffer.shape));
  buffer_usage.emplace_back(BufferUse::Write(buffer_.destination_buffer.slice,
                                             buffer_.destination_buffer.shape));
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// CollectivePermuteCmd
//===----------------------------------------------------------------------===//

CollectivePermuteCmd::CollectivePermuteCmd(
    CollectiveConfig config, P2PConfig p2p_config,
    absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandType::kCollectivePermuteCmd, std::move(config),
                    CommunicationId(1)),
      p2p_config_(std::move(p2p_config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::StatusOr<const se::CommandBuffer::Command*> CollectivePermuteCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  int device_ordinal = execute_params.stream->parent()->device_ordinal();
  XLA_VLOG_DEVICE(5, device_ordinal) << "CollectivePermuteCmd:";

  for (size_t i = 0; i < device_buffers.size(); ++i) {
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Src: " << buffers_[i].source_buffer << " ("
        << device_buffers[i].source_buffer.opaque() << ")";
    XLA_VLOG_DEVICE(5, device_ordinal)
        << "  Dst: " << buffers_[i].destination_buffer << " ("
        << device_buffers[i].destination_buffer.opaque() << ")";
  }

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "CollectivePermuteCmd requires collective parameters and cliques");
  }

  TF_ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                      GetGpuCliqueKey(*execute_params.collective_params,
                                      config().replica_groups,
                                      config().group_mode, communication_id()));

  TF_ASSIGN_OR_RETURN(
      Communicator * comm,
      execute_params.collective_cliques->GetComm(
          clique_key, execute_params.collective_params->global_device_id));

  std::string device_string =
      CollectiveThunk::GetDeviceString(*execute_params.collective_params);
  bool use_symmetric_buffer = config().use_symmetric_buffer;

  TF_ASSIGN_OR_RETURN(
      const int64_t current_id,
      GetCollectiveCurrentId(execute_params.collective_params, p2p_config_));

  const P2PConfig::SourceTargetMapEntry source_target =
      P2PConfig::GetSourceTarget(p2p_config_.id_to_source_target, current_id);

  // Convert logical source/target IDs to communicator-local ranks.
  P2PConfig::SourceTargetRanks source_target_ranks;
  if (source_target.source) {
    source_target_ranks.source = RankId(*source_target.source);
  }
  if (source_target.target) {
    source_target_ranks.target = RankId(*source_target.target);
  }

  // MemCpy case is not currently supported in CommandBuffer.
  return RecordTracedCommand(
      execute_params, record_params, std::move(record_action), command_buffer,
      [&](se::Stream* stream) {
        return RunCollectivePermute(source_target_ranks, device_buffers,
                                    *stream, *comm, device_string, current_id,
                                    use_symmetric_buffer);
      });
}

Command::BufferUses CollectivePermuteCmd::buffer_uses() const {
  BufferUses buffer_usage;
  for (const CollectiveThunk::Buffer& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

//===----------------------------------------------------------------------===//
// RaggedAllToAllCmd
//===----------------------------------------------------------------------===//

namespace {
struct RaggedAllToAllCmdState : CommandState {
  // MultiGpuBarrier: Device memory buffer for signal values (one per peer).
  // Peers write specific slots in this array to signal this device.
  se::DeviceAddressHandle barrier_signal_buffer;

  // MultiGpuBarrier: Device memory for the current local step counter.
  // This value is incremented locally by the kernel after every barrier.
  se::DeviceAddressHandle barrier_signal_value;
};
}  // namespace

RaggedAllToAllCmd::RaggedAllToAllCmd(
    RaggedAllToAllConfig ragged_all_to_all_config,
    absl::Span<const CollectiveThunk::Buffer> buffers)
    : CollectiveCmd(CommandType::kRaggedAllToAllCmd,
                    ragged_all_to_all_config.config),
      ragged_all_to_all_config_(std::move(ragged_all_to_all_config)),
      buffers_(buffers.begin(), buffers.end()) {}

absl::Status RaggedAllToAllCmd::Initialize(
    const Thunk::InitializeParams& params) {
  // Safety Checks
  // Check if all replicas are local.
  TF_RET_CHECK(IsAllReplicasLocal(params.local_device_count, config()))
      << "RaggedAllToAllCmd: All replicas must be local for the one-shot "
         "kernel to work";

  return absl::OkStatus();
}

absl::StatusOr<const se::CommandBuffer::Command*> RaggedAllToAllCmd::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::StreamExecutor* executor = execute_params.stream->parent();

  if (!execute_params.collective_params || !execute_params.collective_cliques) {
    return absl::InvalidArgumentError(
        "RaggedAllToAllCmd requires collective parameters and cliques");
  }

  // 1. Resolve Clique Key
  TF_ASSIGN_OR_RETURN(GpuCliqueKey clique_key,
                      GetGpuCliqueKey(*execute_params.collective_params,
                                      config().replica_groups,
                                      config().group_mode, communication_id()));

  // 2. Prepare Local Data
  auto device_ordinal = execute_params.stream->parent()->device_ordinal();
  const std::optional<RankId> rank_opt =
      clique_key.rank(execute_params.collective_params->global_device_id);
  TF_RET_CHECK(rank_opt.has_value())
      << "RaggedAllToAllCmd::Record: Current device is not part of the clique";
  RankId rank = rank_opt.value();

  // 3. Safety Checks
  // Check if peer access is enabled.
  TF_ASSIGN_OR_RETURN(
      bool peer_access_enabled,
      execute_params.collective_cliques->peer_access_enabled(clique_key));

  TF_RET_CHECK(peer_access_enabled)
      << "RaggedAllToAllCmd: Peer access must be enabled.";

  // 4. Manage State (Allocations)
  absl::Status state_status = absl::OkStatus();

  RaggedAllToAllCmdState* cmd_state =
      record_params.state.GetOrCreate<RaggedAllToAllCmdState>(
          this, command_buffer,
          [&]() -> std::unique_ptr<RaggedAllToAllCmdState> {
            auto state = std::make_unique<RaggedAllToAllCmdState>();

            // 1. Allocate Signal Buffer (Array of uint32_t)
            int64_t signal_buf_bytes =
                se::gpu::MultiGpuBarrierKernel::kMaxPeers * sizeof(uint32_t);
            state->barrier_signal_buffer = se::DeviceAddressHandle{
                executor, executor->Allocate(signal_buf_bytes)};

            // 2. Allocate Counter (Scalar uint32_t)
            state->barrier_signal_value = se::DeviceAddressHandle{
                executor, executor->Allocate(sizeof(uint32_t))};

            // Error Checking
            if (state->barrier_signal_buffer.address().is_null() ||
                state->barrier_signal_value.address().is_null()) {
              state_status = absl::ResourceExhaustedError(
                  "Failed to allocate RaggedAllToAll barrier buffers");
              return nullptr;
            }

            // Zero buffers synchronously (Safe during graph construction)
            state_status = execute_params.stream->MemZero(
                state->barrier_signal_buffer.address_ptr(), signal_buf_bytes);
            if (!state_status.ok()) {
              return nullptr;
            }

            state_status = execute_params.stream->MemZero(
                state->barrier_signal_value.address_ptr(), sizeof(uint32_t));
            if (!state_status.ok()) {
              return nullptr;
            }

            state_status = execute_params.stream->BlockHostUntilDone();
            if (!state_status.ok()) {
              return nullptr;
            }

            return state;
          });

  // Check the captured state_status *after* GetOrCreate returns.
  TF_RETURN_IF_ERROR(state_status);
  TF_RET_CHECK(cmd_state != nullptr)
      << "Failed to get or create RaggedAllToAllCmdState";

  // 5. Resolve Buffer Addresses (For the current run/capture)
  TF_ASSIGN_OR_RETURN(
      std::vector<DeviceBufferPair> device_buffers,
      ConvertToDeviceBuffers(execute_params.buffer_allocations, buffers_,
                             config().operand_element_type));

  // 6. Define the Trace Callback
  auto trace = [&](se::Stream* stream) -> absl::Status {
    const se::DeviceAddressBase& output_buffer =
        device_buffers[1].destination_buffer;

    // A. Rendezvous
    // Exchanges *current* buffer addresses to bake into the graph.
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<std::vector<RaggedAllToAllRendezvousValue>>
            participants,
        RendezvousResources(device_ordinal, rank, clique_key, output_buffer,
                            cmd_state->barrier_signal_buffer.address()));

    // B. Record Kernel Launches
    return RunOneShotRaggedAllToAll(clique_key, *stream, rank,
                                    cmd_state->barrier_signal_buffer.address(),
                                    cmd_state->barrier_signal_value.address(),
                                    ragged_all_to_all_config_.num_total_updates,
                                    ragged_all_to_all_config_.num_input_rows,
                                    ragged_all_to_all_config_.num_row_elements,
                                    device_buffers, *participants);
  };

  return RecordTracedCommand(execute_params, record_params,
                             std::move(record_action), command_buffer,
                             std::move(trace));
}

Command::BufferUses RaggedAllToAllCmd::buffer_uses() const {
  BufferUses buffer_usage;
  for (const CollectiveThunk::Buffer& buffer : buffers_) {
    buffer_usage.emplace_back(BufferUse::Read(buffer.source_buffer.slice,
                                              buffer.source_buffer.shape));
    buffer_usage.emplace_back(BufferUse::Write(
        buffer.destination_buffer.slice, buffer.destination_buffer.shape));
  }
  return buffer_usage;
}

}  // namespace xla::gpu
