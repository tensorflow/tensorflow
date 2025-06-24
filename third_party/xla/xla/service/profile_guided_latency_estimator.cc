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

#include "xla/service/profile_guided_latency_estimator.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {

namespace {

// Small wrapper ensuring aggregator is provided and if it is, then it performs
// forwarding the instruction to an appropriate handler.
void HandleMissingInstructionCost(ProfileStatisticsAggregator* aggregator,
                                  const HloInstruction* instruction) {
  if (aggregator != nullptr) {
    aggregator->HandleMissingInstructionCost(*instruction);
  }
}

// Small wrapper ensuring aggregator is provided and if it is, then it performs
// forwarding the instruction to an appropriate handler.
void HandleFoundInstructionCost(ProfileStatisticsAggregator* aggregator,
                                const HloInstruction* instruction) {
  if (aggregator != nullptr) {
    aggregator->HandleFoundInstructionCost(*instruction);
  }
}

// Small wrapper ensuring aggregator is provided and if it is, then it performs
// forwarding the from/to instruction pair to an appropriate handler.
void HandleMissingInstructionLatency(ProfileStatisticsAggregator* aggregator,
                                     const HloGraphNode& from,
                                     const HloGraphNode& to) {
  if (aggregator != nullptr) {
    aggregator->HandleMissingInstructionLatency(from.GetInstr(), to.GetInstr());
  }
}

// Small wrapper ensuring aggregator is provided and if it is, then it performs
// forwarding the from/to instruction pair to an appropriate handler.
void HandleFoundInstructionLatency(ProfileStatisticsAggregator* aggregator,
                                   const HloGraphNode& from,
                                   const HloGraphNode& to) {
  if (aggregator != nullptr) {
    aggregator->HandleFoundInstructionLatency(from.GetInstr(), to.GetInstr());
  }
}

}  // namespace

LatencyEstimator::TimeCost ProfileGuidedLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  static constexpr HloGraphNode::TimeCost kLowLatency = 1.0;
  const HloOpcode from_op = from.GetInstr().opcode();
  if (!config_.schedule_send_recvs &&
      (from_op == HloOpcode::kSend || from_op == HloOpcode::kRecv)) {
    return kLowLatency;
  }

  auto it = instr_map_.find(from.GetInstr().name());
  if (it == instr_map_.end() &&
      (from.GetInstr().opcode() == HloOpcode::kAsyncStart ||
       from.GetInstr().opcode() == HloOpcode::kAsyncDone)) {
    absl::string_view wrapped_inst_name =
        from.GetInstr().async_wrapped_instruction()->name();
    VLOG(2) << "PGLE found async wrapped instruction: " << wrapped_inst_name
            << " in " << from.GetInstr().name();
    it = instr_map_.find(wrapped_inst_name);
  }

  if (it == instr_map_.end()) {
    VLOG(1)
        << "PGLE did NOT find wrapped instruction name or async start. From: "
        << from.GetInstr().name();
    HandleMissingInstructionLatency(aggregator_.get(), from, target);
    return latency_estimator_->GetLatencyBetween(from, target);
  }

  auto it2 = it->second.latencies.find(target.GetInstr().name());
  if (it2 == it->second.latencies.end() &&
      (target.GetInstr().opcode() == HloOpcode::kAsyncStart ||
       target.GetInstr().opcode() == HloOpcode::kAsyncDone)) {
    it2 = it->second.latencies.find(
        target.GetInstr().async_wrapped_instruction()->name());
  }
  if (it2 != it->second.latencies.end()) {
    VLOG(2) << "PGLE found latency between " << from.GetInstr().name()
            << " and " << target.GetInstr().name() << " in latency info";
    HandleFoundInstructionLatency(aggregator_.get(), from, target);
    return it2->second * CyclesPerMicrosecond();
  }

  // For async-start/done instructions, if there is no entry in latencies, fall
  // back to using instruction cost as the latency.
  if (it->second.cost.has_value() &&
      (IsAsyncPair(from, target) || IsP2pPair(from, target))) {
    VLOG(2) << "PGLE found latency for async op " << from.GetInstr().name()
            << " and (assumed)" << target.GetInstr().name()
            << " in instruction costs";
    HandleFoundInstructionLatency(aggregator_.get(), from, target);
    return *it->second.cost * CyclesPerMicrosecond();
  }

  VLOG(1) << "PGLE did not find relevant profiling info for '"
          << from.GetInstr().name() << "', and '" << target.GetInstr().name()
          << "'.";
  HandleMissingInstructionLatency(aggregator_.get(), from, target);
  return latency_estimator_->GetLatencyBetween(from, target);
}

LatencyEstimator::TimeCost ProfileGuidedLatencyEstimator::NodeCost(
    const HloInstruction* instr) const {
  if (hlo_query::IsAsyncCollectiveStartOp(instr, /*include_send_recv=*/true) ||
      hlo_query::IsAsyncCollectiveDoneOp(instr, /*include_send_recv=*/true)) {
    static constexpr TimeCost kLowCost = 1.0;
    return kLowCost;
  }
  if (auto it = instr_map_.find(instr->name());
      it != instr_map_.end() && it->second.cost.has_value()) {
    VLOG(2) << "PGLE found cost for: " << instr->name();
    HandleFoundInstructionCost(aggregator_.get(), instr);
    return *it->second.cost;
  }
  VLOG(1) << "PGLE missed cost for: " << instr->name();
  HandleMissingInstructionCost(aggregator_.get(), instr);
  return latency_estimator_->NodeCost(instr);
}

ProfileStatisticsAggregator::Statistics
ProfileStatisticsAggregator::GetStats() {
  return {
      /*found_instructions_count=*/found_instructions_count_,
      /*missing_instructions=*/missing_instructions_,
  };
}

absl::Status ProfileGuidedLatencyEstimator::CheckAccuracy(
    const HloModule& module) {
  if (aggregator_ == nullptr) {
    return absl::FailedPreconditionError(
        "Failing because `aggregator_` was not provided when constructing "
        "PGLE.");
  }

  for (const auto& comp : module.computations()) {
    // We only check profile application for while bodies and entry computation
    // to avoid fine-grained exclusion of fusion computations, wrapped async
    // computations, trivial to_apply computations (present in e.g. reductions)
    // etc.
    if (!comp->IsEntryComputation() &&
        !comp->GetUniqueCaller(HloOpcode::kWhile)) {
      continue;
    }
    for (const HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      NodeCost(instr);
      HloGraphNode from(instr, /*original_position=*/-1);
      for (const HloInstruction* user : instr->users()) {
        HloGraphNode to(user, /*original_position=*/-1);
        GetLatencyBetween(from, to);
      }
    }
  }
  ProfileStatisticsAggregator::Statistics stats = aggregator_->GetStats();
  size_t missing_instructions_count = stats.missing_instructions.size();
  if (missing_instructions_count > 0) {
    LOG(WARNING) << "Found " << stats.found_instructions_count
                 << " instructions from the profile.";
    LOG(WARNING) << "Missing " << missing_instructions_count
                 << " instructions from the profile.";
    for (const HloInstruction* instr : stats.missing_instructions) {
      LOG(WARNING) << "  " << instr->name();
    }
    if (module.config().debug_options().xla_gpu_pgle_accuracy_checker() ==
        DebugOptions::PGLE_STRICTNESS_LEVEL_ERROR) {
      return absl::InvalidArgumentError(
          absl::StrCat("Found ", missing_instructions_count,
                       " missing instructions. Discarding the profile."));
    }
  }
  return absl::OkStatus();
}

ProfileGuidedLatencyEstimator::ProfileGuidedLatencyEstimator(
    const SchedulerConfig& config,
    std::unique_ptr<LatencyEstimator> latency_estimator,
    const tensorflow::profiler::ProfiledInstructionsProto& proto,
    std::unique_ptr<ProfileStatisticsAggregator> aggregator)
    : config_(config),
      latency_estimator_(std::move(latency_estimator)),
      aggregator_(std::move(aggregator)) {
  const int cycles_per_microsecond = latency_estimator_->CyclesPerMicrosecond();
  for (const auto& instr_cost : proto.costs()) {
    instr_map_[instr_cost.name()] =
        ProfileInfo{instr_cost.cost_us() * cycles_per_microsecond};
  }
  for (const auto& latency : proto.latencies()) {
    auto it = instr_map_.insert(std::make_pair(latency.source(), ProfileInfo{}))
                  .first;
    it->second.latencies[latency.target()] =
        latency.latency_us() * cycles_per_microsecond;
  }
}

}  // namespace xla
