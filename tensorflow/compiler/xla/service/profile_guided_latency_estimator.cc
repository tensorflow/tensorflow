/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/profile_guided_latency_estimator.h"

#include <memory>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/latency_hiding_scheduler.h"
#include "tensorflow/compiler/xla/xla.pb.h"

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
  if (it == instr_map_.end()) {
    return latency_estimator_->GetLatencyBetween(from, target);
  }
  auto it2 = it->second.latencies.find(target.GetInstr().name());
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
  const HloOpcode opcode = instr->opcode();
  if (hlo_query::IsAsyncCollectiveStartOp(opcode) ||
      hlo_query::IsAsyncCollectiveDoneOp(opcode) ||
      opcode == HloOpcode::kSend || opcode == HloOpcode::kRecv ||
      opcode == HloOpcode::kSendDone || opcode == HloOpcode::kRecvDone) {
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
    const ProfiledInstructionsProto& proto)
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
