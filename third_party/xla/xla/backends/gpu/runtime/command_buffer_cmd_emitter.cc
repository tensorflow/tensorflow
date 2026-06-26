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
#include <array>
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
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/dynamic_slice_fusion_v2_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/status_macros.h"
#include "xla/util.h"

namespace xla::gpu {

namespace {
// A context for tracking thunks to commands conversion details.
struct ConversionContext {
  std::vector<Command::ResourceUses> extra_resources;
};

// A thunk with its concurrent region id in the case where inherited
// region id is used.
struct ThunkWithRegionId {
  Thunk* thunk;
  std::optional<uint64_t> region_id;
};

void FlattenThunksImpl(const ThunkSequence& sequence,
                       std::optional<uint64_t> inherited_region_id,
                       std::vector<ThunkWithRegionId>& thunks) {
  for (const auto& thunk : sequence) {
    std::optional<uint64_t> region_id =
        thunk->concurrent_region_id().has_value()
            ? thunk->concurrent_region_id()
            : inherited_region_id;
    // Additional thunks with nesting must be added here to support
    // concurrent regions.
    if (thunk->kind() == Thunk::Kind::kSequential) {
      FlattenThunksImpl(
          static_cast<const SequentialThunk*>(thunk.get())->thunks(), region_id,
          thunks);
    } else if (thunk->kind() == Thunk::Kind::kAsyncStart) {
      FlattenThunksImpl(
          static_cast<const AsyncStartThunk*>(thunk.get())->thunks(), region_id,
          thunks);
    } else if (thunk->kind() == Thunk::Kind::kAsyncDone) {
      // AsyncDone is a no-op in command buffers; filter it out.
      continue;
    } else {
      thunks.push_back({thunk.get(), region_id});
    }
  }
}

// Returns a flattened list of thunks in the given sequence with their
// concurrent region ids.
// Eg:
// Input:  [A(R1), B(R1), SequentialThunk(R2){C, D}, E(R3)]
// Output: [{A, 1}, {B, 1}, {C, 2}, {D, 2}, {E, 3}]
std::vector<ThunkWithRegionId> FlattenThunks(
    const ThunkSequence& sequence,
    std::optional<uint64_t> inherited_region_id) {
  std::vector<ThunkWithRegionId> thunks;
  FlattenThunksImpl(sequence, inherited_region_id, thunks);
  return thunks;
}
}  // namespace

// Appends command(s) converted from `sequence` to `cmd_sequence`.
static absl::Status AppendCommands(ConversionContext& ctx,
                                   CommandSequence& cmd_sequence,
                                   const ThunkSequence& sequence,
                                   const ConvertToCommandsOptions& options);

//===----------------------------------------------------------------------===//
// Conversions from Thunk to Command
//===----------------------------------------------------------------------===//

static absl::Status SetOrUpdateCommandBufferExecutors(
    WhileThunk& thunk, const ConvertToCommandsOptions& options) {
  VLOG(1) << "WhileThunk: " << thunk.profile_annotation();
  ASSIGN_OR_RETURN(
      CommandExecutor cond_cmds,
      ConvertToCommands(thunk.condition_executor().thunks(), options));
  ASSIGN_OR_RETURN(CommandExecutor body_cmds,
                   ConvertToCommands(thunk.body_executor().thunks(), options));

  return thunk.SetOrUpdateCommandBufferExecutors(
      std::move(cond_cmds), std::move(body_cmds), options.enable_loop_unroll);
}

static absl::Status SetOrUpdateCommandBufferBranchExecutors(
    ConditionalThunk& thunk, const ConvertToCommandsOptions& options) {
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
    for (const ThunkExecutor& branch_thunk : thunk.branch_executors()) {
      ASSIGN_OR_RETURN(CommandExecutor cmds,
                       ConvertToCommands(branch_thunk.thunks(), options));
      branch_cmds.emplace_back(std::move(cmds));
    }
  }
  return thunk.SetOrUpdateCommandBufferBranchExecutors(std::move(branch_cmds));
}

static absl::Status AppendCommands(ConversionContext& ctx,
                                   CommandSequence& cmd_sequence, Thunk& thunk,
                                   const ConvertToCommandsOptions& options) {
  switch (thunk.kind()) {
    case Thunk::Kind::kConditional: {
      auto& conditional_thunk = static_cast<ConditionalThunk&>(thunk);
      RETURN_IF_ERROR(
          SetOrUpdateCommandBufferBranchExecutors(conditional_thunk, options));
      cmd_sequence.Append(&conditional_thunk);
      return absl::OkStatus();
    }
    case Thunk::Kind::kAsyncDone:
      // Async done thunks are no-ops in command buffers.
      return absl::OkStatus();
    case Thunk::Kind::kWhile: {
      auto& while_thunk = static_cast<WhileThunk&>(thunk);
      RETURN_IF_ERROR(SetOrUpdateCommandBufferExecutors(while_thunk, options));
      cmd_sequence.Append(&while_thunk);
      return absl::OkStatus();
    }
    case Thunk::Kind::kDynamicSliceFusion: {
      auto& dynamic_slice_fusion_thunk =
          static_cast<DynamicSliceFusionV2Thunk&>(thunk);
      ASSIGN_OR_RETURN(
          CommandExecutor cmds,
          ConvertToCommands(dynamic_slice_fusion_thunk.thunks(), options));
      RETURN_IF_ERROR(
          dynamic_slice_fusion_thunk.SetOrUpdateCommandBufferExecutor(
              std::move(cmds)));
      cmd_sequence.Append(&dynamic_slice_fusion_thunk);
      return absl::OkStatus();
    }
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
      break;
  }

  if (auto* command = dynamic_cast<Command*>(&thunk)) {
    // Command/thunk hybrids are owned by the input ThunkSequence and outlive
    // the returned CommandSequence, so command sequences borrow them directly.
    cmd_sequence.Append(command);
    return absl::OkStatus();
  }

  return Internal("Unsupported thunk kind: %s",
                  Thunk::KindToString(thunk.kind()));
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
  const std::vector<Thunk*>& scheduled_thunks() const {
    return scheduled_thunks_;
  }
  // Returns the source nodes of the current region, i.e. the nodes that will
  // execute first.
  const std::vector<Thunk*>& scheduled_source_thunks() const {
    return scheduled_source_thunks_;
  }
  // Returns the sink nodes of the current region, i.e. the nodes that will
  // execute last.
  absl::Span<Thunk* const> scheduled_sink_thunks() const {
    return absl::MakeConstSpan(scheduled_sink_thunks_);
  }
  // Returns artificial dependencies inserted by the scheduler.
  const absl::flat_hash_map<const Thunk*, std::vector<Thunk*>>&
  scheduled_successors() const {
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
  std::array<std::deque<Thunk*>, kMaxConcurrentOps> ready_;

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
  std::array<Thunk*, kMaxConcurrentOps> scheduled_sink_thunks_ = {};
};

struct ConcurrentRegion {
  uint64_t id;
  std::vector<Thunk*> thunks;
};

absl::StatusOr<std::vector<ConcurrentRegion>> CollectConcurrentRegions(
    const ThunkSequence& sequence,
    std::optional<uint64_t> inherited_region_id) {
  std::vector<ThunkWithRegionId> flat_thunks =
      FlattenThunks(sequence, inherited_region_id);
  std::vector<ConcurrentRegion> concurrent_regions;

  for (const auto& [thunk, region_id] : flat_thunks) {
    TF_RET_CHECK(region_id.has_value())
        << "Concurrent region id must be set in scheduling mode "
           "kConcurrentRegions. Failed on thunk: "
        << Thunk::KindToString(thunk->kind()) << " ("
        << thunk->profile_annotation() << ")";

    const uint64_t concurrent_region_id = region_id.value();
    if (concurrent_regions.empty() ||
        concurrent_region_id != concurrent_regions.back().id) {
      TF_RET_CHECK(concurrent_regions.empty() ||
                   concurrent_region_id > concurrent_regions.back().id)
          << "Concurrent region ids are not monotonic. Failed on thunk: "
          << Thunk::KindToString(thunk->kind()) << " ("
          << thunk->profile_annotation() << ")";
      concurrent_regions.push_back({concurrent_region_id});
    }

    concurrent_regions.back().thunks.push_back(thunk);
    VLOG(3) << "Thunk " << thunk->thunk_info().profile_annotation
            << " is in region " << concurrent_region_id;
  }

  return concurrent_regions;
}

int64_t CommandIndex(
    const absl::flat_hash_map<const Thunk*, int64_t>& thunk_to_index,
    const Thunk* thunk) {
  auto it = thunk_to_index.find(thunk);
  CHECK(it != thunk_to_index.end())
      << "Missing command for thunk " << thunk->profile_annotation();
  return it->second;
}

void AddCommandDependency(
    ConversionContext& ctx, const CommandSequence& cmd_sequence,
    const absl::flat_hash_map<const Thunk*, int64_t>& thunk_to_index,
    const Thunk* predecessor, const Thunk* successor) {
  const int64_t predecessor_index = CommandIndex(thunk_to_index, predecessor);
  const int64_t successor_index = CommandIndex(thunk_to_index, successor);
  ctx.extra_resources[successor_index].push_back(
      ResourceUse::Read(cmd_sequence[predecessor_index]->token()));
}

absl::Status AppendScheduledConcurrentRegionCommands(
    ConversionContext& ctx, CommandSequence& cmd_sequence,
    std::vector<ConcurrentRegion>& concurrent_regions,
    const ConvertToCommandsOptions& options,
    std::vector<ConcurrentRegionScheduler>& concurrent_region_schedules,
    absl::flat_hash_map<const Thunk*, int64_t>& thunk_to_index) {
  concurrent_region_schedules.reserve(concurrent_regions.size());

  for (ConcurrentRegion& region : concurrent_regions) {
    ConcurrentRegionScheduler& scheduler =
        concurrent_region_schedules.emplace_back(absl::MakeSpan(region.thunks));
    for (Thunk* thunk : scheduler.scheduled_thunks()) {
      const int64_t command_index = cmd_sequence.size();
      RETURN_IF_ERROR(AppendCommands(ctx, cmd_sequence, *thunk, options));
      TF_RET_CHECK(cmd_sequence.size() == command_index + 1)
          << "Concurrent region thunk must append exactly one command: "
          << Thunk::KindToString(thunk->kind()) << " ("
          << thunk->profile_annotation() << ")";
      thunk_to_index[thunk] = command_index;
    }
  }

  return absl::OkStatus();
}

void AddDependenciesWithinConcurrentRegions(
    ConversionContext& ctx, const CommandSequence& cmd_sequence,
    absl::Span<const ConcurrentRegionScheduler> concurrent_region_schedules,
    const absl::flat_hash_map<const Thunk*, int64_t>& thunk_to_index) {
  for (const auto& scheduler : concurrent_region_schedules) {
    // Suppress custom-deterministic-iteration-order.
    // Non-determinism is fine in this case since ctx.extra_resources is
    // ordered and accessed by index. NOLINTNEXTLINE
    for (const auto& [thunk, successors] : scheduler.scheduled_successors()) {
      for (Thunk* successor : successors) {
        AddCommandDependency(ctx, cmd_sequence, thunk_to_index, thunk,
                             successor);
      }
    }
  }
}

void AddDependenciesBetweenConcurrentRegions(
    ConversionContext& ctx, const CommandSequence& cmd_sequence,
    absl::Span<const ConcurrentRegionScheduler> concurrent_region_schedules,
    const absl::flat_hash_map<const Thunk*, int64_t>& thunk_to_index) {
  // The source nodes of each region depend on all sink nodes of the previous
  // region to enforce serialization between regions.
  for (size_t i = 1; i < concurrent_region_schedules.size(); ++i) {
    for (Thunk* sink :
         concurrent_region_schedules[i - 1].scheduled_sink_thunks()) {
      if (sink == nullptr) {
        continue;
      }
      for (Thunk* source :
           concurrent_region_schedules[i].scheduled_source_thunks()) {
        AddCommandDependency(ctx, cmd_sequence, thunk_to_index, sink, source);
      }
    }
  }
}

absl::Status AppendCommandsInConcurrentRegions(
    ConversionContext& ctx, CommandSequence& cmd_sequence,
    const ThunkSequence& sequence, const ConvertToCommandsOptions& options,
    std::optional<uint64_t> inherited_region_id) {
  ASSIGN_OR_RETURN(std::vector<ConcurrentRegion> concurrent_regions,
                   CollectConcurrentRegions(sequence, inherited_region_id));

  std::vector<ConcurrentRegionScheduler> concurrent_region_schedules;
  absl::flat_hash_map<const Thunk*, int64_t> thunk_to_index;
  RETURN_IF_ERROR(AppendScheduledConcurrentRegionCommands(
      ctx, cmd_sequence, concurrent_regions, options,
      concurrent_region_schedules, thunk_to_index));

  // Ensure extra_resources is sized to cover all commands added so far
  // (including those added by nested AppendCommands calls).
  ctx.extra_resources.resize(cmd_sequence.size());

  AddDependenciesWithinConcurrentRegions(
      ctx, cmd_sequence, concurrent_region_schedules, thunk_to_index);
  AddDependenciesBetweenConcurrentRegions(
      ctx, cmd_sequence, concurrent_region_schedules, thunk_to_index);
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
                                             options, std::nullopt);
  }
  for (const std::unique_ptr<Thunk>& thunk : sequence) {
    RETURN_IF_ERROR(AppendCommands(ctx, cmd_sequence, *thunk, options));
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
