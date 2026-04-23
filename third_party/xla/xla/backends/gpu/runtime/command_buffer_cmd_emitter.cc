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

#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/all_reduce_thunk.h"
#include "xla/backends/gpu/runtime/all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_broadcast_thunk.h"
#include "xla/backends/gpu/runtime/collective_permute_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_cmd.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/cudnn_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/gpublas_lt_matmul_thunk.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/legacy_custom_call_thunk.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/ragged_all_to_all_thunk.h"
#include "xla/backends/gpu/runtime/recv_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/send_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"

namespace xla::gpu {

namespace {
// A context for tracking thunks to commands conversion details.
struct ConversionContext {
  std::vector<Command::ResourceUses> extra_resources;
};
}  // namespace

// Appends command(s) converted from `sequence` to `cmd_sequence`.
static absl::Status AppendCommands(ConversionContext& ctx,
                                   CommandSequence& cmd_sequence,
                                   const ThunkSequence& sequence,
                                   const ConvertToCommandsOptions& options);

//===----------------------------------------------------------------------===//
// Conversions from Thunk to Command
//===----------------------------------------------------------------------===//

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const WhileThunk& thunk, const ConvertToCommandsOptions& options) {
  VLOG(1) << "WhileThunk: " << thunk.profile_annotation();
  ASSIGN_OR_RETURN(
      CommandExecutor cond_cmds,
      ConvertToCommands(thunk.condition_executor().thunks(), options));
  ASSIGN_OR_RETURN(CommandExecutor body_cmds,
                   ConvertToCommands(thunk.body_executor().thunks(), options));

  return std::make_unique<WhileCmd>(
      thunk.condition_result_buffer(), std::move(cond_cmds),
      std::move(body_cmds), thunk.trip_count(), options.enable_loop_unroll);
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const ConditionalThunk& thunk, const ConvertToCommandsOptions& options) {
  std::vector<CommandExecutor> branch_cmds;
  branch_cmds.reserve(thunk.branch_executors().size());
  if (thunk.branch_index_is_bool()) {
    // For boolean predicates, we need to convert the branches in reverse order
    // because the first branch is the "false" branch and the second is "true"
    CHECK_EQ(thunk.branch_executors().size(), 2);
    ASSIGN_OR_RETURN(
        branch_cmds.emplace_back(),
        ConvertToCommands(thunk.branch_executors()[1].thunks(), options));
    ASSIGN_OR_RETURN(
        branch_cmds.emplace_back(),
        ConvertToCommands(thunk.branch_executors()[0].thunks(), options));
  } else {
    for (auto& branch_thunk : thunk.branch_executors()) {
      ASSIGN_OR_RETURN(CommandExecutor cmds,
                       ConvertToCommands(branch_thunk.thunks(), options));
      branch_cmds.emplace_back(std::move(cmds));
    }
  }
  return std::make_unique<CaseCmd>(thunk.branch_index_buffer(),
                                   std::move(branch_cmds));
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const AllReduceThunk& thunk) {
  return std::make_unique<AllReduceCmd>(thunk.config(), thunk.reduction_kind(),
                                        thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const ReduceScatterThunk& thunk) {
  return std::make_unique<ReduceScatterCmd>(
      thunk.config(), thunk.reduction_kind(), thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const AllToAllThunk& thunk) {
  return std::make_unique<AllToAllCmd>(
      thunk.config(), thunk.has_split_dimension(), thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const AllGatherThunk& thunk) {
  return std::make_unique<AllGatherCmd>(thunk.config(), thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const CollectiveBroadcastThunk& thunk) {
  return std::make_unique<CollectiveBroadcastCmd>(thunk.config(),
                                                  thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const CollectivePermuteThunk& thunk) {
  return std::make_unique<CollectivePermuteCmd>(
      thunk.config(), thunk.p2p_config(), thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const RaggedAllToAllThunk& thunk) {
  return std::make_unique<RaggedAllToAllCmd>(thunk.ragged_all_to_all_config(),
                                             thunk.buffers());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const RecvThunk& thunk) {
  return std::make_unique<RecvCmd>(thunk.config(), thunk.p2p_config(),
                                   thunk.buffer());
}

static absl::StatusOr<std::unique_ptr<Command>> Convert(
    const SendThunk& thunk) {
  return std::make_unique<SendCmd>(thunk.config(), thunk.p2p_config(),
                                   thunk.buffer());
}

//===----------------------------------------------------------------------===//
static absl::StatusOr<std::unique_ptr<Command>> CopyMetadata(
    absl::StatusOr<std::unique_ptr<Command>> cmd, const Thunk& thunk) {
  if (cmd.ok()) {
    (*cmd)->set_profile_annotation(thunk.profile_annotation());
    return cmd;
  }
  return cmd;
}

// Takes Thunk& (non-const) rather than const Thunk& so that thunks which
// also implement Command can be appended as borrowed Command* without
// const_cast (which is banned). The thunks in ThunkSequence are non-const
// (unique_ptr<Thunk>), so callers always have a non-const reference available.
template <typename ThunkType, typename... Args>
static absl::StatusOr<std::unique_ptr<Command>> Convert(Thunk& thunk,
                                                        Args&&... args) {
  return CopyMetadata(
      Convert(static_cast<ThunkType&>(thunk), std::forward<Args>(args)...),
      thunk);
}

static absl::Status AppendCommands(ConversionContext& ctx,
                                   CommandSequence& cmd_sequence, Thunk& thunk,
                                   const ConvertToCommandsOptions& options) {
  auto append =
      [&](absl::StatusOr<std::unique_ptr<Command>> command) -> absl::Status {
    if (!command.ok()) {
      return command.status();
    }

    cmd_sequence.Append(std::move(*command));
    return absl::OkStatus();
  };

  switch (thunk.kind()) {
    case Thunk::Kind::kConditional:
      return append(Convert<ConditionalThunk>(thunk, options));
    case Thunk::Kind::kCopy:
      cmd_sequence.Append(static_cast<DeviceToDeviceCopyThunk*>(&thunk));
      return absl::OkStatus();
    case Thunk::Kind::kCustomCall:
      // CustomCallThunk implements TracedCommand directly; append as borrowed
      // pointer — the thunk outlives the command sequence.
      if (auto* ffi_thunk = dynamic_cast<CustomCallThunk*>(&thunk)) {
        cmd_sequence.Append(ffi_thunk);
        return absl::OkStatus();
      }
      // LegacyCustomCallThunk implements TracedCommand directly; append as
      // borrowed pointer — the thunk outlives the command sequence. Note: in
      // production, command_buffer_conversion_pass excludes
      // LegacyCustomCallThunk from automatic conversion, so this arm is only
      // reached when the emitter is invoked directly (e.g. tests).
      if (auto* legacy_thunk = dynamic_cast<LegacyCustomCallThunk*>(&thunk)) {
        cmd_sequence.Append(legacy_thunk);
        return absl::OkStatus();
      }
      return absl::InternalError("Unknown custom call thunk type");
    // CustomKernelThunk implements Command directly; append borrowed pointer.
    case Thunk::Kind::kCustomKernel:
      cmd_sequence.Append(static_cast<CustomKernelThunk*>(&thunk));
      return absl::OkStatus();
    // KernelThunk implements Command directly; append borrowed pointer.
    case Thunk::Kind::kKernel:
      cmd_sequence.Append(static_cast<KernelThunk*>(&thunk));
      return absl::OkStatus();
    // GemmThunk implements TracedCommand directly; append as borrowed
    // pointer — the thunk outlives the command sequence.
    case Thunk::Kind::kGemm:
      cmd_sequence.Append(static_cast<GemmThunk*>(&thunk));
      return absl::OkStatus();
    // CublasLtMatmulThunk implements TracedCommand directly; append as
    // borrowed pointer — the thunk outlives the command sequence.
    case Thunk::Kind::kCublasLtMatmul:
      cmd_sequence.Append(static_cast<CublasLtMatmulThunk*>(&thunk));
      return absl::OkStatus();
    case Thunk::Kind::kMemzero:
      cmd_sequence.Append(static_cast<MemzeroThunk*>(&thunk));
      return absl::OkStatus();
    case Thunk::Kind::kAllGather:
      return append(Convert<AllGatherThunk>(thunk));
    case Thunk::Kind::kAllReduce:
      return append(Convert<AllReduceThunk>(thunk));
    case Thunk::Kind::kReduceScatter:
      return append(Convert<ReduceScatterThunk>(thunk));
    case Thunk::Kind::kAllToAll:
      return append(Convert<AllToAllThunk>(thunk));
    case Thunk::Kind::kCollectiveBroadcast:
      return append(Convert<CollectiveBroadcastThunk>(thunk));
    case Thunk::Kind::kCollectivePermute:
      return append(Convert<CollectivePermuteThunk>(thunk));
    case Thunk::Kind::kRaggedAllToAll:
      return append(Convert<RaggedAllToAllThunk>(thunk));
    case Thunk::Kind::kRecv:
      return append(Convert<RecvThunk>(thunk));
    case Thunk::Kind::kSend:
      return append(Convert<SendThunk>(thunk));
    // These thunks implement Command directly; append borrowed pointers.
    // Note: kCopy also borrows DeviceToDeviceCopyThunk (see case above).
    case Thunk::Kind::kMemset32BitValue:
      cmd_sequence.Append(static_cast<Memset32BitValueThunk*>(&thunk));
      return absl::OkStatus();
    case Thunk::Kind::kPartitionId:
      cmd_sequence.Append(static_cast<PartitionIdThunk*>(&thunk));
      return absl::OkStatus();
    case Thunk::Kind::kReplicaId:
      cmd_sequence.Append(static_cast<ReplicaIdThunk*>(&thunk));
      return absl::OkStatus();
    case Thunk::Kind::kAsyncDone:
      // Async done thunks are no-ops in command buffers.
      return absl::OkStatus();
    case Thunk::Kind::kWhile:
      return append(Convert<WhileThunk>(thunk, options));
    // CuDnnThunk implements Command (via TracedCommand) directly; append
    // borrowed pointer.
    case Thunk::Kind::kCuDnn:
      cmd_sequence.Append(static_cast<CuDnnThunk*>(&thunk));
      return absl::OkStatus();
    // Sequential thunk does not have any special semantics and we simply inline
    // all nested thunks into command buffer.
    case Thunk::Kind::kSequential:
      return AppendCommands(ctx, cmd_sequence,
                            static_cast<const SequentialThunk&>(thunk).thunks(),
                            options);

    // Async start thunks inline their nested thunk sequence into the command
    // buffer. Command buffers rely on DAG structure for dependencies.
    case Thunk::Kind::kAsyncStart: {
      auto& start = static_cast<const AsyncStartThunk&>(thunk);
      return AppendCommands(ctx, cmd_sequence, start.thunks(), options);
    }

    case Thunk::Kind::kCommandBuffer:
      return Internal(
          "Error trying to emit command for a CommandBufferThunk. Input HLO "
          "must already contain command buffers and XLA should not run command "
          "buffer scheduling pass the second time. If it happens in the test, "
          "try explicitly disabling command buffers in tested HLO module.");

    default:
      return Internal("Unsupported thunk kind: %s",
                      Thunk::KindToString(thunk.kind()));
  }
}

namespace {

// Simple list scheduler for thunks in a concurrent region. Generates a schedule
// with two concurrent lanes (kMaxConcurrentOps). That is, at most two thunks
// are executed concurrently. The schedule may execute thunks out of order to
// achieve better parallelism.
class ConcurrentRegionScheduler {
 public:
  explicit ConcurrentRegionScheduler(absl::Span<Thunk*> thunks) {
    InitializeSuccessors(thunks);
    InitializeInDegrees(thunks);
    Schedule(thunks);
  }

  // Returns the scheduled thunks in topological order.
  std::vector<Thunk*> scheduled_thunks() { return scheduled_thunks_; }
  // Returns the source nodes of the current region, i.e. the nodes that will
  // execute first.
  std::vector<Thunk*> scheduled_source_thunks() {
    return scheduled_source_thunks_;
  }
  // Returns the sink nodes of the current region, i.e. the nodes that will
  // execute last.
  std::vector<Thunk*> scheduled_sink_thunks() { return scheduled_sink_thunks_; }
  // Returns artificial dependencies inserted by the scheduler.
  absl::flat_hash_map<const Thunk*, std::vector<Thunk*>>
  scheduled_successors() {
    return scheduled_successors_;
  }

 private:
  // Build map of successors from buffer conflicts among thunks.
  void InitializeSuccessors(absl::Span<Thunk*> thunks) {
    // Pre-compute buffer read-write sets for all thunks in this sequence.
    std::vector<BufferUse::ReadWriteSet> rw_sets(thunks.size());
    for (size_t i = 0; i < thunks.size(); ++i) {
      thunks[i]->Walk(
          [&](auto* thunk) { rw_sets[i].AddAll(thunk->buffer_uses()); });
    }
    for (size_t i = 0; i < thunks.size(); ++i) {
      for (size_t j = i + 1; j < rw_sets.size(); ++j) {
        if (rw_sets[j].HasConflicts(rw_sets[i])) {
          thunks_[thunks[i]].successors.push_back(thunks[j]);
        }
      }
    }
    // TODO(tjoerg): Thunks support resource dependencies. Though they are
    // currently not used for XLA GPU, support should be added in the future.
  }

  // Initialize in-degrees of all thunks, based on the successors map.
  void InitializeInDegrees(absl::Span<Thunk*> thunks) {
    for (size_t i = 0; i < thunks.size(); ++i) {
      for (Thunk* successor : thunks_[thunks[i]].successors) {
        thunks_[successor].in_degree++;
      }
    }
    if (VLOG_IS_ON(5)) {
      for (Thunk* thunk : thunks) {
        VLOG(5) << "In-degree of " << thunk->thunk_info().profile_annotation
                << ": " << thunks_[thunk].in_degree;
      }
    }
  }

  // Insert thunks with in-degree 0 into the ready queue. The queue is divided
  // into kMaxConcurrentOps lanes for cache locality. For initialization, simply
  // assign to lanes in a round robin fashion.
  void InitializeReady(absl::Span<Thunk*> thunks) {
    for (size_t i = 0; i < thunks.size(); ++i) {
      if (thunks_[thunks[i]].in_degree == 0) {
        ready_[i % kMaxConcurrentOps].push_back(thunks[i]);
        ready_size_++;
      }
    }
  }

  void EnqueueReady(Thunk* thunk, size_t lane_id) {
    // Insert in the front for cache locality.
    ready_[lane_id].push_front(thunk);
    ready_size_++;
  }

  // Move ready thunks to the active list. Prefer scheduling thunks in the same
  // lane as their data predecessors.
  void MoveReadyToActive(std::vector<Thunk*>& active) {
    size_t num_ops_to_move = std::min(kMaxConcurrentOps, ready_size_);
    for (size_t i = 0; i < num_ops_to_move; ++i) {
      // Prefer thunks in the same lane, but use thunks from other lanes if
      // necessary.
      if (!ready_[i].empty()) {
        active.push_back(ready_[i].front());
        ready_[i].pop_front();
        ready_size_--;
      } else {
        for (size_t j = 0; j < kMaxConcurrentOps; ++j) {
          if (!ready_[j].empty()) {
            active.push_back(ready_[j].front());
            ready_[j].pop_front();
            ready_size_--;
            break;
          }
        }
      }
    }
  }

  // Returns the number of thunks in the ready queue.
  size_t ready_size() const { return ready_size_; }

  // Generate a schedule: While there are thunks in the ready queue, move ready
  // thunks to the active list, update in-degrees of successors, update the
  // ready queue, and start over.
  void Schedule(absl::Span<Thunk*> thunks) {
    InitializeReady(thunks);
    std::vector<Thunk*> active;
    std::vector<Thunk*> previously_active;
    scheduled_sink_thunks_.resize(kMaxConcurrentOps);

    while (ready_size() > 0) {
      MoveReadyToActive(active);
      VLOG(5) << "Active list: "
              << absl::StrJoin(
                     active, ", ", [](std::string* out, const Thunk* thunk) {
                       absl::StrAppend(out,
                                       thunk->thunk_info().profile_annotation);
                     });
      // Record the first active thunks of this region.
      if (scheduled_source_thunks_.empty()) {
        scheduled_source_thunks_ = active;
      }
      // Update in-degrees of successors and enqueue ready thunks.
      for (size_t lane_id = 0; lane_id < active.size(); ++lane_id) {
        Thunk* thunk = active.at(lane_id);
        for (Thunk* successor : thunks_[thunk].successors) {
          thunks_[successor].in_degree--;
          if (thunks_[successor].in_degree == 0) {
            EnqueueReady(successor, lane_id);
          }
        }
        scheduled_thunks_.push_back(thunk);

        // Add dependencies to previously active ops.
        if (lane_id < previously_active.size()) {
          Thunk* previously_active_same_lane = previously_active.at(lane_id);
          scheduled_successors_[previously_active_same_lane].push_back(thunk);
        }
      }
      previously_active = active;
      // Record the active thunks as sinks for now. Note that last while loop
      // iteration may not have all lanes filled, hence do it on each iteration.
      for (size_t lane_id = 0; lane_id < active.size(); ++lane_id) {
        scheduled_sink_thunks_[lane_id] = active[lane_id];
      }
      active.clear();
    }
  }

  // Number of concurrent lanes in the schedule.
  // Using 2 lanes is a conservative default. Higher degrees of concurrency are
  // prone to jitter, i.e. variations in the execution time. When multiple GPUs
  // run in lockstep, the worst case performance determines the overall speed.
  static constexpr size_t kMaxConcurrentOps = 2;
  // Ready queue with kMaxConcurrentOps lanes.
  std::deque<Thunk*> ready_[kMaxConcurrentOps];

  // See comments on the methods above.
  size_t ready_size_ = 0;
  struct ThunkState {
    std::vector<Thunk*> successors;
    int64_t in_degree = 0;
  };
  absl::flat_hash_map<const Thunk*, ThunkState> thunks_;
  std::vector<Thunk*> scheduled_thunks_;
  absl::flat_hash_map<const Thunk*, std::vector<Thunk*>> scheduled_successors_;
  std::vector<Thunk*> scheduled_source_thunks_;
  std::vector<Thunk*> scheduled_sink_thunks_;
};

absl::Status AppendCommandsInConcurrentRegions(
    ConversionContext& ctx, CommandSequence& cmd_sequence,
    const ThunkSequence& sequence, const ConvertToCommandsOptions& options) {
  absl::flat_hash_map<int64_t, std::vector<Thunk*>>
      concurrent_region_id_to_thunks;
  std::vector<int64_t> concurrent_region_ids;
  for (const std::unique_ptr<Thunk>& thunk : sequence) {
    TF_RET_CHECK(thunk->concurrent_region_id().has_value())
        << "Concurrent region id must be set in scheduling mode "
           "kConcurrentRegions.";
    int64_t concurrent_region_id = thunk->concurrent_region_id().value();
    concurrent_region_id_to_thunks[concurrent_region_id].push_back(thunk.get());
    if (concurrent_region_ids.empty() ||
        concurrent_region_id > concurrent_region_ids.back()) {
      concurrent_region_ids.push_back(concurrent_region_id);
    }
    CHECK_GE(concurrent_region_id, concurrent_region_ids.back())
        << "Concurrent region ids are not monotonic.";
    VLOG(3) << "Thunk " << thunk->thunk_info().profile_annotation
            << " is in region " << concurrent_region_id;
  }
  std::vector<ConcurrentRegionScheduler> concurrent_region_schedules;
  absl::flat_hash_map<const Thunk*, int64_t> thunk_to_index;
  for (int64_t concurrent_region_id : concurrent_region_ids) {
    std::vector<Thunk*> thunks =
        concurrent_region_id_to_thunks[concurrent_region_id];
    ConcurrentRegionScheduler& scheduler =
        concurrent_region_schedules.emplace_back(absl::MakeSpan(thunks));
    for (Thunk* thunk : scheduler.scheduled_thunks()) {
      TF_RETURN_IF_ERROR(AppendCommands(ctx, cmd_sequence, *thunk, options));
      int64_t index = cmd_sequence.size() - 1;
      thunk_to_index[thunk] = index;
    }
  }
  // Ensure extra_resources is sized to cover all commands added so far
  // (including those added by nested AppendCommands calls).
  ctx.extra_resources.resize(cmd_sequence.size());

  // Add dependencies within concurrent regions.
  for (auto& scheduler : concurrent_region_schedules) {
    for (auto& [thunk, successors] : scheduler.scheduled_successors()) {
      for (Thunk* successor : successors) {
        ctx.extra_resources[thunk_to_index[thunk]].push_back(ResourceUse::Read(
            cmd_sequence[thunk_to_index[successor]]->token()));
      }
    }
  }
  // Add dependencies between concurrent regions. The source nodes of subsequent
  // regions depend on all sink nodes of prior regions.
  for (size_t i = 1; i < concurrent_region_schedules.size(); ++i) {
    for (Thunk* sink :
         concurrent_region_schedules[i - 1].scheduled_sink_thunks()) {
      if (sink == nullptr) {
        continue;
      }
      for (Thunk* source :
           concurrent_region_schedules[i].scheduled_source_thunks()) {
        // The source nodes of region `i` depend on the sink nodes of region
        // `i-1` to ensure serialization between regions.
        ctx.extra_resources[thunk_to_index[source]].push_back(
            ResourceUse::Read(cmd_sequence[thunk_to_index[sink]]->token()));
      };
    }
  }
  return absl::OkStatus();
}

}  // namespace

static absl::Status AppendCommands(ConversionContext& ctx,
                                   CommandSequence& cmd_sequence,
                                   const ThunkSequence& sequence,
                                   const ConvertToCommandsOptions& options) {
  if (options.synchronization_mode ==
      CommandExecutor::SynchronizationMode::kConcurrentRegions) {
    return AppendCommandsInConcurrentRegions(ctx, cmd_sequence, sequence,
                                             options);
  }
  absl::flat_hash_map<const Thunk*, int64_t> thunk_to_index;
  for (const std::unique_ptr<Thunk>& thunk : sequence) {
    RETURN_IF_ERROR(AppendCommands(ctx, cmd_sequence, *thunk, options));
    int64_t index = cmd_sequence.size() - 1;
    thunk_to_index[thunk.get()] = index;
  }

  return absl::OkStatus();
}

absl::StatusOr<CommandExecutor> ConvertToCommands(
    const ThunkSequence& sequence, const ConvertToCommandsOptions& options) {
  VLOG(3) << absl::StreamFormat(
      "Convert thunk sequence to command executor: synchronization_mode=%v",
      options.synchronization_mode);
  ConversionContext ctx;
  CommandSequence cmd_sequence;
  RETURN_IF_ERROR(AppendCommands(ctx, cmd_sequence, sequence, options));
  return CommandExecutor::Create(std::move(cmd_sequence),
                                 options.synchronization_mode,
                                 std::move(ctx.extra_resources));
}

}  // namespace xla::gpu
