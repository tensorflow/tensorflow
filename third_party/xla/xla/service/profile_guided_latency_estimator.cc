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

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "tsl/profiler/protobuf/profiled_instructions.pb.h"

namespace xla {

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
    VLOG(10) << "PGLE found async wrapped instruction: " << wrapped_inst_name
             << " in " << from.GetInstr().name();
    it = instr_map_.find(wrapped_inst_name);
  }

  if (it == instr_map_.end()) {
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
    VLOG(10) << "PGLE found latency between " << from.GetInstr().name()
             << " and " << target.GetInstr().name() << " in latency info";
    return it2->second * CyclesPerMicrosecond();
  }

  // For async-start/done instructions, if there is no entry in latencies, fall
  // back to using instruction cost as the latency.
  if (it->second.cost.has_value() && IsAsyncPair(from, target)) {
    VLOG(10) << "PGLE found latency for async op " << from.GetInstr().name()
             << " and (assumed)" << target.GetInstr().name()
             << " in instruction costs";
    return *it->second.cost * CyclesPerMicrosecond();
  }

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
    VLOG(10) << "PGLE found cost for: " << instr->name();
    return *it->second.cost;
  }
  VLOG(10) << "PGLE missed cost for: " << instr->name();
  return latency_estimator_->NodeCost(instr);
}

ProfileGuidedLatencyEstimator::ProfileGuidedLatencyEstimator(
    const SchedulerConfig& config,
    std::unique_ptr<LatencyEstimator> latency_estimator,
    const tensorflow::profiler::ProfiledInstructionsProto& proto)
    : config_(config), latency_estimator_(std::move(latency_estimator)) {
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
