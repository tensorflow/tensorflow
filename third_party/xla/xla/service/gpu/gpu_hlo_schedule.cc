/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_hlo_schedule.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/buffer_value.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_schedule_postprocessing.h"
#include "xla/service/gpu/model/analytical_latency_estimator.h"
#include "xla/service/hlo_memory_scheduler.h"
#include "xla/service/hlo_pass_pipeline.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/service/p2p_schedule_preparation.h"
#include "xla/service/profile_guided_latency_estimator.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

bool IsNopInstruction(const HloInstruction& hlo) {
  HloOpcode op = hlo.opcode();
  return op == HloOpcode::kGetTupleElement || op == HloOpcode::kBitcast ||
         op == HloOpcode::kConstant || op == HloOpcode::kParameter ||
         hlo.IsEffectiveBitcast();
}

bool ShouldScheduleAsEarlyAsPossible(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kCollectivePermuteStart:
      return !IsSyncCollective(&instr);
    case HloOpcode::kCustomCall:
      return static_cast<const HloCustomCallInstruction&>(instr)
                 .custom_call_schedule() ==
             CustomCallSchedule::SCHEDULE_EARLIEST;
    default:
      return false;
  }
}

bool ShouldScheduleSuccessor(const HloInstruction& sussessor,
                             const HloPredicate& is_scheduled) {
  return ShouldScheduleAsEarlyAsPossible(sussessor) &&
         absl::c_all_of(sussessor.operands(), is_scheduled) &&
         absl::c_all_of(sussessor.control_predecessors(), is_scheduled);
}

bool ShouldScheduleAsLateAsPossible(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kCollectivePermuteDone:
      return ShouldScheduleAsEarlyAsPossible(*instr.operand(0));
    case HloOpcode::kCustomCall:
      return static_cast<const HloCustomCallInstruction&>(instr)
                 .custom_call_schedule() == CustomCallSchedule::SCHEDULE_LATEST;
    default:
      return false;
  }
}

bool ShouldSchedulePredecessor(const HloInstruction& predecessor,
                               const HloPredicate& is_scheduled) {
  return ShouldScheduleAsLateAsPossible(predecessor) &&
         absl::c_all_of(predecessor.users(), is_scheduled) &&
         absl::c_all_of(predecessor.control_successors(), is_scheduled);
}

// Schedules certain ops as early or late as possible. This supports a
// custom-call use case, where a logical operation is lowered into two HLOs
// (e.g., PerformX and PerformXDone). We utilize this mechanism to either hide
// host latencies between the pair of the custom-calls or more accurately
// identify the def-use relationship of the two calls (typically PerformX is
// scheduled right after all of its producers have been scheduled and
// PerformXDone is scheduled right before its first consumer.)
HloInstructionSequence PostprocessorToScheduleAsEarlyOrLateAsPossible(
    const HloInstructionSequence& input) {
  std::vector<HloInstruction*> earliest_scheduled;
  {
    absl::flat_hash_set<HloInstruction*> scheduled;
    auto is_scheduled = [&](const HloInstruction* instr) -> bool {
      return scheduled.contains(instr);
    };
    auto add_to_schedule = [&](HloInstruction* instr) {
      earliest_scheduled.push_back(instr);
      scheduled.insert(instr);
    };

    for (HloInstruction* instr : input.instructions()) {
      if (is_scheduled(instr)) continue;

      add_to_schedule(instr);

      // Schedule any successor that should be scheduled as early as possible if
      // all of its producers and control_predecessors have been scheduled.
      for (HloInstruction* user : instr->users()) {
        if (is_scheduled(user)) continue;

        if (ShouldScheduleSuccessor(*user, is_scheduled)) {
          add_to_schedule(user);
        }
      }
      for (HloInstruction* successor : instr->control_successors()) {
        if (is_scheduled(successor)) continue;

        if (ShouldScheduleSuccessor(*successor, is_scheduled)) {
          add_to_schedule(successor);
        }
      }
    }
  }

  std::deque<HloInstruction*> latest_scheduled;
  {
    absl::flat_hash_set<HloInstruction*> scheduled;
    auto is_scheduled = [&](const HloInstruction* instr) -> bool {
      return scheduled.contains(instr);
    };
    auto add_to_schedule = [&](HloInstruction* instr) {
      latest_scheduled.push_front(instr);
      scheduled.insert(instr);
    };
    for (auto it = earliest_scheduled.rbegin(); it != earliest_scheduled.rend();
         it++) {
      if (is_scheduled(*it)) continue;

      add_to_schedule(*it);

      // Schedule any predecessor that should be scheduled as late as possible
      // if all of its users and control_successors have been scheduled.
      for (HloInstruction* operand : (*it)->operands()) {
        if (is_scheduled(operand)) continue;

        if (ShouldSchedulePredecessor(*operand, is_scheduled)) {
          add_to_schedule(operand);
        }
      }
      for (HloInstruction* predecessor : (*it)->control_predecessors()) {
        if (is_scheduled(predecessor)) continue;

        if (ShouldSchedulePredecessor(*predecessor, is_scheduled)) {
          add_to_schedule(predecessor);
        }
      }
    }
  }

  HloInstructionSequence result;
  absl::c_for_each(latest_scheduled,
                   [&](HloInstruction* i) { result.push_back(i); });

  // Schedule post-processing can't introduce new instructions.
  CHECK(input.instructions().size() == result.size())
      << "schedule as early or late post-processing changed schedule size from "
      << input.instructions().size() << " to " << result.size();

  return result;
}

// Post process to move start/done for synchronous collectives next to each
// other.
HloInstructionSequence PostprocessorToScheduleSyncCollectives(
    const HloInstructionSequence& input) {
  HloInstructionSequence result;

  // Returns true if `inst` is a synchronous version of async collective start
  // operation (marked with `is_sync` attribute).
  auto is_sync_start = [](const HloInstruction* instr) {
    return hlo_query::IsAsyncCollectiveStartOp(instr,
                                               /*include_send_recv=*/true) &&
           IsSyncCollective(instr);
  };

  for (HloInstruction* instr : input.instructions()) {
    // Skip synchronous start instruction as it will be scheduled later when
    // we'll process corresponding done instruction.
    if (is_sync_start(instr)) continue;

    // Find a start instruction corresponding to done and schedule it right
    // before a done if it's a synchronous version.
    if (hlo_query::IsAsyncCollectiveDoneOp(instr, true)) {
      HloInstruction* start = instr->mutable_operand(0);
      if (is_sync_start(start)) result.push_back(start);
    }

    result.push_back(instr);
  }

  // Schedule post-processing can't introduce new instructions.
  CHECK(input.instructions().size() == result.size())
      << "sync collectives post-processing changed schedule size from "
      << input.instructions().size() << " to " << result.size();

  return result;
}

absl::StatusOr<HloSchedule> ScheduleGpuModuleWithMemoryScheduler(
    const HloModule* module, int64_t pointer_size) {
  return ScheduleModule(
      module,
      [pointer_size](const BufferValue& buffer) {
        return ShapeUtil::ByteSizeOf(buffer.shape(), pointer_size);
      },
      ComputationSchedulerToModuleScheduler(DefaultMemoryScheduler,
                                            PostProcessSchedule));
}

// Latency hiding scheduler support.

CanonicalAsyncOp GpuGetCanonicalAsyncOp(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kSend:
      return {HloOpcode::kAsyncStart, HloOpcode::kSend};
    case HloOpcode::kSendDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kSend};
    case HloOpcode::kRecv:
      return {HloOpcode::kAsyncStart, HloOpcode::kRecv};
    case HloOpcode::kRecvDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kRecv};
    default:
      return DefaultGetCanonicalAsyncOp(hlo);
  }
}

SchedulerConfig GetSchedulerConfig(int64_t memory_limit) {
  SchedulerConfig config;
  config.all_reduce_overlap_limit = 1;
  config.collective_broadcast_overlap_limit = 1;
  config.collective_permute_overlap_limit = 1;
  config.use_real_cost_model = false;
  config.aggressive_scheduling_policies = true;
  config.schedule_send_recvs = true;
  config.memory_limit = memory_limit;
  return config;
}

// GPU specific resources for latency hiding scheduler.
//
// We use two different set of resources to model the scheduling of asynchronous
// collective operations and P2P Send and Recv operations. This corresponds to
// the fact that the runtime use a stream to run asynchronous collective
// operations and two other streams to run P2P Send and Recv operations.
enum class GpuResourceType {
  kGpuAsyncStreamSend0 = 0,        // A resource for P2P Send operation.
  kGpuAsyncStreamSend1 = 1,        // Another resource for P2P Send operation.
  kGpuAsyncStreamRecv0 = 2,        // A resource for P2P Recv operation.
  kGpuAsyncStreamRecv1 = 3,        // Another resource for P2P Recv operation.
  kGpuAsyncStreamCollectives = 4,  // The resource for collective operations.
  kGpuAsyncStreamComputes = 5,     // The resource for async compute operations.
  kNumTargetResources = 6,
};

// Returns the pipeline stream for a P2P instruction recorded in a frontend
// attribute.
int64_t GetPipelineStream(const HloInstruction& start) {
  auto it = start.frontend_attributes().map().find(kSendRecvPipelineAttr);
  if (it != start.frontend_attributes().map().end() && it->second == "1") {
    return 1;
  }
  return 0;
}

// Returns the resource type and resource usage for a P2P instruction.
std::pair<GpuResourceType, ResourceUsageType> GetP2PResourceAndUsage(
    const HloInstruction& instr, const CanonicalAsyncOp& op) {
  ResourceUsageType usage = op.outer == HloOpcode::kAsyncStart
                                ? ResourceUsageType::kResourceRelease
                                : ResourceUsageType::kResourceOccupy;
  int64_t pipeline = GetPipelineStream(instr);
  HloOpcode opcode = op.inner;
  GpuResourceType resource;
  if (pipeline == 0) {
    resource = opcode == HloOpcode::kSend
                   ? GpuResourceType::kGpuAsyncStreamSend0
                   : GpuResourceType::kGpuAsyncStreamRecv0;
  } else {
    resource = opcode == HloOpcode::kSend
                   ? GpuResourceType::kGpuAsyncStreamSend1
                   : GpuResourceType::kGpuAsyncStreamRecv1;
  }
  return {resource, usage};
}

// Base GPU async tracker that enables async tracking only for async collectives
// that are marked for async execution.
class GpuAsyncTrackerBase : public AsyncTracker {
 public:
  using AsyncTracker::AsyncTracker;

  explicit GpuAsyncTrackerBase(
      const SchedulerConfig& config,
      GetCanonicalAsyncOpFunc func = GpuGetCanonicalAsyncOp)
      : AsyncTracker(config, func) {}

  bool IsAsyncComputeOp(const HloInstruction& hlo) const {
    return (hlo.opcode() == HloOpcode::kAsyncStart ||
            hlo.opcode() == HloOpcode::kAsyncDone) &&
           !hlo_query::IsCollectiveCommunicationOp(
               hlo.async_wrapped_opcode()) &&
           hlo.async_execution_thread() != hlo.parent()->execution_thread();
  }

  bool IsSupportedAsyncDone(const HloInstruction& hlo) const override {
    return (hlo_query::IsAsyncCollectiveDoneOp(&hlo,
                                               /*include_send_recv=*/true) &&
            !IsSyncCollective(hlo.operand(0))) ||
           IsAsyncComputeOp(hlo);
  }

  // Returns if this is an Async op start that the scheduler supports.
  bool IsSupportedAsyncStart(const HloInstruction& hlo) const override {
    return (hlo_query::IsAsyncCollectiveStartOp(&hlo,
                                                /*include_send_recv=*/true) &&
            !IsSyncCollective(&hlo)) ||
           IsAsyncComputeOp(hlo);
  }

  void PostProcessScheduleGraph(
      HloScheduleGraph* schedule_graph,
      const LatencyEstimator* latency_estimator) const override {
    for (auto inst : schedule_graph->GetOriginalInstrList()) {
      // Force pipelined Recv to be closed to Recvdone so that copies inserted
      // for RecvDone can be eliminated.
      if (inst->opcode() == HloOpcode::kRecv) {
        if (inst->frontend_attributes().map().count(kSendRecvPipelineAttr) >
            0) {
          HloGraphNode& node = schedule_graph->GetNode(inst);
          node.SetForceEarly(true);
          VLOG(5) << "Setting force early for instruction: "
                  << inst->ToString();
        }
      }
      if (inst->has_backend_config()) {
        auto gpu_config = inst->backend_config<GpuBackendConfig>();
        if (gpu_config.ok()) {
          HloGraphNode& node = schedule_graph->GetNode(inst);
          node.SetForceDelay(gpu_config->force_earliest_schedule());
          VLOG(5) << "Setting force delay for instruction: "
                  << inst->ToString();
        }
      }
    }
  }
};

// GPU async tracker maps all collectives onto an async stream resource.
class GpuAsyncTracker : public GpuAsyncTrackerBase {
 public:
  explicit GpuAsyncTracker(const SchedulerConfig& config)
      : GpuAsyncTrackerBase(config) {}

  ResourcesVector GetResourcesFromInstruction(
      const HloInstruction& instr) const override {
    CanonicalAsyncOp op = GetCanonicalAsyncOp(instr);
    if (op.outer == HloOpcode::kAsyncStart ||
        op.outer == HloOpcode::kAsyncDone) {
      ResourceUsageType usage;
      GpuResourceType resource;
      if (op.inner == HloOpcode::kSend || op.inner == HloOpcode::kRecv) {
        std::tie(resource, usage) = GetP2PResourceAndUsage(instr, op);
      } else {
        usage = op.outer == HloOpcode::kAsyncStart
                    ? ResourceUsageType::kResourceRelease
                    : ResourceUsageType::kResourceOccupy;
        resource = hlo_query::IsCollectiveCommunicationOp(op.inner)
                       ? GpuResourceType::kGpuAsyncStreamCollectives
                       : GpuResourceType::kGpuAsyncStreamComputes;
      }
      return {std::make_pair(
          GetFirstTargetDefinedResource() + static_cast<int64_t>(resource),
          usage)};
    }
    return GpuAsyncTrackerBase::GetResourcesFromInstruction(instr);
  }

  int64_t GetNumTargetDefinedResources() const override {
    return static_cast<int64_t>(GpuResourceType::kNumTargetResources);
  };

  // Returns how many instructions using the given resource_type we can overlap
  int64_t GetNumAvailableResources(int64_t resource_type) const override {
    const int64_t first_target_resource = GetFirstTargetDefinedResource();
    if (resource_type < first_target_resource) {
      return GpuAsyncTrackerBase::GetNumAvailableResources(resource_type);
    }
    CHECK_LT(resource_type,
             first_target_resource +
                 static_cast<int64_t>(GpuResourceType::kNumTargetResources));

    // We will allow upto 1 outstanding collective on the async stream. This
    // controls the number of collectives in flight in the schedule (a
    // collective is in flight if the start is issued but not done). As an
    // example, with 1, LHS will generate the schedule: s0,e0,s1,e1, i.e., s1
    // is not scheduled until e0 is scheduled. With 2, the scheduler can
    // schedule s0,s1,e0,e1, because it assumes that the 2 instances of the
    // resources do not interfere with each other. If we do want to support > 1
    // async stream, we can increase this number and then do a post-pass on the
    // scheduled code to assign async stream-id to collectives (and actually
    // support > 1 async stream in the runtime).
    // The only case we'd allow 2 for now is when the current resource is
    // for an async computation operation which will be allocated with
    // a dedicated compute stream. It can run concurrently with
    // another collective.
    if ((resource_type - first_target_resource) ==
        static_cast<int64_t>(GpuResourceType::kGpuAsyncStreamComputes)) {
      return 2;
    }

    return 1;
  }

  absl::string_view GetResourceName(int64_t resource_type) const override {
    const int64_t first_target_resource = GetFirstTargetDefinedResource();
    if (resource_type < first_target_resource) {
      return GpuAsyncTrackerBase::GetResourceName(resource_type);
    }
    CHECK_LE(resource_type,
             first_target_resource + GetNumTargetDefinedResources());
    switch (
        static_cast<GpuResourceType>(resource_type - first_target_resource)) {
      case GpuResourceType::kGpuAsyncStreamSend0:
        return "kGpuAsyncStreamSend0";
      case GpuResourceType::kGpuAsyncStreamSend1:
        return "kGpuAsyncStreamSend1";
      case GpuResourceType::kGpuAsyncStreamRecv0:
        return "kGpuAsyncStreamRecv0";
      case GpuResourceType::kGpuAsyncStreamRecv1:
        return "kGpuAsyncStreamRecv1";
      case GpuResourceType::kGpuAsyncStreamCollectives:
        return "kGpuAsyncStreamCollectives";
      case GpuResourceType::kGpuAsyncStreamComputes:
        return "kGpuAsyncStreamComputes";
      default:
        return "kUnsupportedResource";
    }
  }

  ResourceHazardType GetResourceHazardType(
      int64_t resource_type) const override {
    const int64_t first_target_resource = GetFirstTargetDefinedResource();
    if (resource_type < first_target_resource) {
      return GpuAsyncTrackerBase::GetResourceHazardType(resource_type);
    }
    CHECK_LE(resource_type,
             first_target_resource + GetNumTargetDefinedResources());
    return ResourceHazardType::kUnshareable;
  }

  int64_t GetNumResourcesPerInstruction(
      int64_t resource_type, const HloInstruction& instr) const override {
    int64_t num_resources = GpuAsyncTrackerBase::GetNumResourcesPerInstruction(
        resource_type, instr);
    if (num_resources <= 0 || instr.opcode() != HloOpcode::kWhile) {
      return num_resources;
    }
    // For while-loop with pipelined Send/Recv, the while-body first releases
    // the Send/Recv resource and then uses the resource. Therefore, subtract 1
    // from num_resources for the relevant resource type.
    int64_t first_p2p_resource =
        GetFirstTargetDefinedResource() +
        static_cast<int64_t>(GpuResourceType::kGpuAsyncStreamSend0);
    if (resource_type < first_p2p_resource ||
        resource_type > first_p2p_resource + 4) {
      return num_resources;
    }
    auto find_instruction_for_pipeline = [&](HloOpcode opcode,
                                             int64_t pipeline) {
      for (auto user1 : instr.users()) {
        if (user1->opcode() == HloOpcode::kGetTupleElement) {
          for (auto user2 : user1->users()) {
            if (user2->opcode() == opcode) {
              if (GetPipelineStream(*user2) == pipeline) {
                return true;
              }
            }
          }
        }
      }
      return false;
    };
    bool found;
    // Look into the users of the while-result to find pipelined Send-done or
    // Recv-done.
    if (resource_type == first_p2p_resource) {
      found = find_instruction_for_pipeline(HloOpcode::kSendDone, 0);
    } else if (resource_type == first_p2p_resource + 1) {
      found = find_instruction_for_pipeline(HloOpcode::kSendDone, 1);
    } else if (resource_type == first_p2p_resource + 2) {
      found = find_instruction_for_pipeline(HloOpcode::kRecvDone, 0);
    } else {
      found = find_instruction_for_pipeline(HloOpcode::kRecvDone, 1);
    }
    return num_resources - (found ? 1 : 0);
  }
};

class GpuLatencyEstimator : public ApproximateLatencyEstimator {
 public:
  explicit GpuLatencyEstimator(
      GetCanonicalAsyncOpFunc func = GpuGetCanonicalAsyncOp)
      : ApproximateLatencyEstimator(func) {}
  TimeCost NodeCost(const HloInstruction* instr) const override {
    if (IsNopInstruction(*instr)) {
      return 0.0;
    }
    // Consider cublas/cuddn/softmax custom calls as medium cost. Since the
    // latency between async-start and async-done is 5000 and cost of each
    // custom call is 1000, the LHS will try to schedule approximately 5 of
    // these in between each start/end pair.
    if (instr->opcode() == HloOpcode::kCustomCall) {
      if (IsCublasGemm(*instr) || IsCustomCallToDnnConvolution(*instr)) {
        return ApproximateLatencyEstimator::kMediumCost;
      }
      // consider other custom calls as medium cost for now. Keeping the case
      // explicitly separate for further tuning.
      return ApproximateLatencyEstimator::kMediumCost;
    }
    return ApproximateLatencyEstimator::NodeCost(instr);
  }

  LatencyEstimator::TimeCost GetLatencyBetween(
      const HloGraphNode& from, const HloGraphNode& target) const override {
    if (IsAsyncPair(from, target)) {
      if (from.GetInstr().opcode() == HloOpcode::kRecv) {
        // Recv -> RecvDone has a low latency.
        return ApproximateLatencyEstimator::kLowLatency;
      } else if (from.GetInstr().opcode() == HloOpcode::kSend) {
        // Send -> SendDone has a very high latency.
        return ApproximateLatencyEstimator::kHighLatency * 10;
      }

      return ApproximateLatencyEstimator::kHighLatency;
    }
    // Every other instruction we consider synchronous, which means the
    // latency between each of them is always one unit.
    return ApproximateLatencyEstimator::kLowLatency;
  }
};

tensorflow::profiler::ProfiledInstructionsProto GetProfileForFingerprint(
    tensorflow::profiler::ProfiledInstructionsProto& profile,
    const std::string& fingerprint) {
  tensorflow::profiler::ProfiledInstructionsProto result;
  bool merge_remat_clones = false;
  for (const auto& cost : profile.costs()) {
    absl::string_view cost_name = cost.name();
    std::string new_cost_name = cost.name();
    absl::string_view cost_sep = "::";
    if (absl::StrContains(cost_name, cost_sep)) {
      std::vector<std::string> split_names =
          absl::StrSplit(cost_name, cost_sep);
      if (split_names.size() != 2 || split_names[0] != fingerprint) {
        continue;
      }
      new_cost_name = split_names[1];
    }

    // Check if we see instructions that have ".rematX" suffix. These are clones
    // of original instructions created by HLO rematerialization pass. We will
    // average the costs of the remat clones and the original instruction and
    // use that as the new cost of the original one.
    merge_remat_clones |= absl::StrContains(new_cost_name, ".remat");
    auto* new_cost = result.add_costs();
    new_cost->set_cost_us(cost.cost_us());
    new_cost->set_name(new_cost_name);
  }

  if (!merge_remat_clones) {
    return result;
  }

  auto strip_remat_suffix = [](absl::string_view name) -> absl::string_view {
    absl::string_view suffix = ".remat";
    size_t index = name.rfind(suffix);
    if (index == std::string::npos) {
      return name;
    }
    auto after_suffix = name.substr(index + suffix.size());
    // Everything after ".remat" should be a digit or empty. If yes, strip the
    // .rematN suffix.
    int64_t numeric_suffix;
    if (after_suffix.empty() ||
        absl::SimpleAtoi(after_suffix, &numeric_suffix)) {
      return name.substr(0, index);
    }
    return name;
  };

  // Map from stripped name -> pair<accumulated cost, count>
  absl::flat_hash_map<absl::string_view, std::pair<double, int64_t>> costs;
  for (const auto& cost : result.costs()) {
    std::pair<double, int64_t>& data = costs[strip_remat_suffix(cost.name())];
    data.first += cost.cost_us();
    data.second++;
  }

  tensorflow::profiler::ProfiledInstructionsProto merged_result;
  for (const auto& cost : costs) {
    auto* new_cost = merged_result.add_costs();
    double average = cost.second.first / cost.second.second;
    new_cost->set_cost_us(average);
    new_cost->set_name(std::string(cost.first));
  }

  return merged_result;
}

std::optional<tensorflow::profiler::ProfiledInstructionsProto> ReadPGLEProfile(
    const HloModule* module, const std::string& fingerprint) {
  tensorflow::profiler::ProfiledInstructionsProto profile;

  absl::string_view fdo_profile = module->config().fdo_profile();
  // First attempt to read the profile from `fdo_profile` in ModuleConfig
  if (!fdo_profile.empty()) {
    // Attempt to parse it as a binary proto.
    if (tsl::ParseProtoUnlimited(&profile, fdo_profile.data(),
                                 fdo_profile.size())) {
      LOG(INFO) << "Using PGLE profile for module from fdo_profile (binary)";
      return GetProfileForFingerprint(profile, fingerprint);
    }
    // If not a binary proto, attempt to parse it as a text proto.
    profile.Clear();
    if (tsl::protobuf::TextFormat::ParseFromString(std::string(fdo_profile),
                                                   &profile)) {
      LOG(INFO) << "Using PGLE profile for module from fdo_profile (text)";
      return GetProfileForFingerprint(profile, fingerprint);
    }
    LOG(ERROR) << "Unable to prase FDO profile: not a valid text or binary "
                  "ProfiledInstructionsProto";
  }

  const std::string& pgle_profile_file_or_dir_path =
      module->config()
          .debug_options()
          .xla_gpu_pgle_profile_file_or_directory_path();
  if (pgle_profile_file_or_dir_path.empty()) {
    return std::nullopt;
  }
  tsl::Env* env = tsl::Env::Default();
  auto read_text_or_binary_profile = [&profile, env, &fingerprint](
                                         const std::string& text_path,
                                         const std::string& binary_path)
      -> std::optional<tensorflow::profiler::ProfiledInstructionsProto> {
    if (env->FileExists(text_path).ok()) {
      absl::Status s = tsl::ReadTextProto(env, text_path, &profile);
      if (s.ok()) {
        LOG(INFO) << "Using PGLE profile from " << text_path;
        return GetProfileForFingerprint(profile, fingerprint);
      } else {
        LOG(ERROR) << "Unable to read PGLE text proto from " << text_path
                   << ": " << s.message();
      }
      profile.Clear();
    }
    if (env->FileExists(binary_path).ok()) {
      absl::Status s = tsl::ReadBinaryProto(env, binary_path, &profile);
      if (s.ok()) {
        LOG(INFO) << "Using PGLE profile from " << binary_path;
        return GetProfileForFingerprint(profile, fingerprint);
      } else {
        LOG(ERROR) << "Unable to read PGLE binary proto from " << binary_path
                   << ": " << s.message();
      }
      profile.Clear();
    }
    return std::nullopt;
  };

  // If its a directory, use fingerprint to look for the profile for this
  // specific module.
  if (env->IsDirectory(pgle_profile_file_or_dir_path).ok()) {
    std::string pgle_profile_path_prefix =
        pgle_profile_file_or_dir_path + "/" + fingerprint;
    return read_text_or_binary_profile(pgle_profile_path_prefix + ".pbtxt",
                                       pgle_profile_path_prefix + ".pb");
  }

  // The pgle_profile_file_or_dir is a file. Attempt to read the profile as text
  // proto or binary proto. Attempt to infer the file type based on the
  // extension.
  auto extension = tsl::io::Extension(pgle_profile_file_or_dir_path);
  if (extension == "pbtxt") {
    return read_text_or_binary_profile(pgle_profile_file_or_dir_path, "");
  } else if (extension == "pb") {
    return read_text_or_binary_profile("", pgle_profile_file_or_dir_path);
  } else {
    return read_text_or_binary_profile(pgle_profile_file_or_dir_path,
                                       pgle_profile_file_or_dir_path);
  }
}

// Return true if the profile is applicable to the module. That is true if every
// instruction in the profile is present in the module.
absl::Status IsProfileApplicable(
    const HloModule* module,
    const tensorflow::profiler::ProfiledInstructionsProto& profile) {
  absl::flat_hash_set<absl::string_view> instruction_names;
  for (HloComputation* comp : module->MakeNonfusionComputations()) {
    for (HloInstruction* instr : comp->instructions()) {
      instruction_names.insert(instr->name());
    }
  }

  for (const auto& cost : profile.costs()) {
    if (!instruction_names.contains(cost.name())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "cost name %s not in module %s", cost.name(), module->name()));
    }
  }
  for (const auto& latency : profile.latencies()) {
    if (!instruction_names.contains(latency.source())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "cost name %s not in module %s", latency.source(), module->name()));
    }

    if (!instruction_names.contains(latency.target())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "cost name %s not in module %s", latency.target(), module->name()));
    }
  }
  return absl::OkStatus();
}

}  // end namespace

int64_t GetSizeOfShape(const Shape& shape, int pointer_size) {
  int64_t size = ShapeUtil::ByteSizeOf(shape, pointer_size);
  if (shape.IsTuple() || shape.is_static()) {
    return size;
  }
  // Each dynamic dimension size is represented as a S32.
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions_size();
  return size + metadata_size;
}

static int64_t GetSchedulerMemoryLimit(
    const HloModule* module, const se::DeviceDescription& gpu_device_info,
    int pointer_size);

absl::StatusOr<ScheduleMetadata> ScheduleGpuModule(
    HloModule* module, int64_t pointer_size,
    const se::DeviceDescription& gpu_device_info) {
  int64_t memory_limit =
      GetSchedulerMemoryLimit(module, gpu_device_info, pointer_size);
  if (module->has_schedule()) {
    return ScheduleMetadata{memory_limit};
  }

  HloPassPipeline prepare_pipeline("p2p-schedule-preparation");
  prepare_pipeline.AddPass<P2PSchedulePreparation>();
  TF_RETURN_IF_ERROR(prepare_pipeline.Run(module).status());

  TF_ASSIGN_OR_RETURN(
      HloSchedule schedule,
      ScheduleGpuModuleWithMemoryScheduler(module, pointer_size));
  TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));

  // Tag the module with its 128 bit fingerprint. The fingerprint should include
  // instruction name with ids.
  std::string fingerprint = module->GetFingerprint128(
      HloPrintOptions::Canonical().set_print_backend_config(true));
  FrontendAttributes attributes;
  (*attributes.mutable_map())[std::string(kFingerprintBeforeLHS)] = fingerprint;
  module->add_frontend_attributes(attributes);
  VLOG(1) << "Fingerprint before LHS for module " << module->name() << "("
          << module->unique_id() << ") = " << fingerprint;

  const bool enable_latency_hiding_scheduler =
      module->config()
          .debug_options()
          .xla_gpu_enable_latency_hiding_scheduler();

  if (!enable_latency_hiding_scheduler) {
    return ScheduleMetadata{memory_limit};
  }

  SchedulerConfig config = GetSchedulerConfig(memory_limit);
  auto gpu_latency_estimator = std::make_unique<GpuLatencyEstimator>();

  std::unique_ptr<LatencyEstimator> latency_estimator;
  std::optional<tensorflow::profiler::ProfiledInstructionsProto> profile =
      ReadPGLEProfile(module, fingerprint);

  const bool enable_analytical_latency_estimator =
      module->config()
          .debug_options()
          .xla_gpu_enable_analytical_latency_estimator();
  if (profile.has_value()) {
    latency_estimator = std::make_unique<ProfileGuidedLatencyEstimator>(
        config, std::move(gpu_latency_estimator), profile.value());
    LOG(INFO) << "Found profile, using profile guided latency estimator";
    absl::Status s = IsProfileApplicable(module, profile.value());
    if (!s.ok()) {
      LOG(INFO) << "PGLE profile may not applicable to the module, but will "
                   "still be used : "
                << s.message();
    }
  } else if (enable_analytical_latency_estimator) {
    latency_estimator = std::make_unique<AnalyticalLatencyEstimator>(
        config, std::move(gpu_latency_estimator), gpu_device_info,
        [input_pointer_size = pointer_size](const Shape& shape) {
          return GetSizeOfShape(shape, input_pointer_size);
        },
        module->entry_computation());
    LOG(INFO) << "Using analytical latency estimator";
  } else {
    latency_estimator = std::move(gpu_latency_estimator);
  }

  auto async_tracker = [&]() -> std::unique_ptr<AsyncTracker> {
    return module->config()
                   .debug_options()
                   .xla_gpu_lhs_enable_gpu_async_tracker()
               ? std::make_unique<GpuAsyncTracker>(config)
               : std::make_unique<GpuAsyncTrackerBase>(config);
  }();

  auto shape_size_in_bytes = [pointer_size](const Shape& shape) {
    return GetSizeOfShape(shape, pointer_size);
  };
  HloPassPipeline pipeline("latency-hiding-scheduler");
  auto scheduler_core = std::make_unique<DefaultSchedulerCore>(
      shape_size_in_bytes, async_tracker.get(), latency_estimator.get(),
      config);
  pipeline.AddPass<LatencyHidingScheduler>(
      std::move(latency_estimator), std::move(async_tracker),
      std::move(scheduler_core), shape_size_in_bytes);

  TF_RETURN_IF_ERROR(pipeline.Run(module).status());

  HloPassPipeline postprocessing_pipeline("gpu-schedule-postprocessing");
  postprocessing_pipeline.AddPass<GpuSchedulePostprocessing>();
  TF_RETURN_IF_ERROR(postprocessing_pipeline.Run(module).status());

  return ScheduleMetadata{memory_limit};
}

HloInstructionSequence PostProcessSchedule(
    const HloInstructionSequence& input) {
  HloInstructionSequence result = PostprocessorToScheduleSyncCollectives(input);
  return PostprocessorToScheduleAsEarlyOrLateAsPossible(result);
}

// Compute the device memory limit to be used by passes like scheduler and
// HLO rematerialization.
static int64_t GetSchedulerMemoryLimit(
    const HloModule* module, const se::DeviceDescription& gpu_device_info,
    int pointer_size) {
  // There is a "base" value which is either specified in HloModuleConfig (this
  // value should take into account the fact that we need to leave some memory
  // free for allocations that happen outside of XLA's allocator) or
  // obtained from GPU device info (we scale down this value to leave some space
  // for these outside XLA's allocator allocation).
  //
  // From that base value, subtract any input and output sizes (assuming they
  // are live throughout the execution) and then apply a slop factor.
  const int64_t base_limit =
      module->config().device_memory_size() != 0
          ? module->config().device_memory_size()
          : gpu_device_info.device_memory_size() * 80 / 100;

  // Find the total size of inputs and outputs.
  int64_t total_io_size = 0;
  for (HloInstruction* param :
       module->entry_computation()->parameter_instructions()) {
    ShapeUtil::ForEachSubshape(
        param->shape(),
        [&](const Shape& subshape, const ShapeIndex& /*index*/) {
          total_io_size += GetSizeOfShape(subshape, pointer_size);
        });
  }
  ShapeUtil::ForEachSubshape(
      module->result_shape(),
      [&](const Shape& subshape, const ShapeIndex& /*index*/) {
        total_io_size += GetSizeOfShape(subshape, pointer_size);
      });

  // If any inputs and outputs are aliased, do not double count them.
  module->input_output_alias_config().ForEachAlias(
      [&](const ShapeIndex& output_index,
          const HloInputOutputAliasConfig::Alias&) {
        const Shape& subshape =
            ShapeUtil::GetSubshape(module->result_shape(), output_index);
        total_io_size -= GetSizeOfShape(subshape, pointer_size);
      });

  int64_t limit =
      (base_limit - total_io_size) *
      module->config().debug_options().xla_gpu_memory_limit_slop_factor() / 100;
  return limit;
}

}  // namespace gpu
}  // namespace xla
