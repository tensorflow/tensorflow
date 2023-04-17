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
#include "tensorflow/compiler/xla/service/latency_hiding_scheduler.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {

LatencyEstimator::TimeCost ProfileGuidedLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  static constexpr HloGraphNode::TimeCost kLowLatency = 1.0;
  auto get_latency = [this, &from, &target]() {
    auto from_it = instr_map_.find(from.GetInstr().name());
    auto target_it = instr_map_.find(target.GetInstr().name());
    if (from_it != instr_map_.end() && target_it != instr_map_.end()) {
      const TimeCost from_ts = from_it->second.first;
      const TimeCost from_dur = from_it->second.second;
      const TimeCost target_ts = target_it->second.first;
      const TimeCost target_dur = target_it->second.second;
      CHECK_LE(from_ts, target_ts);
      CHECK_GE(from_dur, 0.0);
      CHECK_GE(target_dur, 0.0);
      return target_ts + target_dur - from_ts - from_dur;
    }
    LOG(FATAL) << "PGLE failed to get latency between "
               << from.GetInstr().name() << " and " << target.GetInstr().name();
  };
  switch (from.GetInstr().opcode()) {
    case HloOpcode::kCollectivePermuteStart:
      if (target.GetInstr().opcode() == HloOpcode::kCollectivePermuteDone) {
        return get_latency();
      }
      break;
    case HloOpcode::kAllGatherStart:
      if (target.GetInstr().opcode() == HloOpcode::kAllGatherDone) {
        return get_latency();
      }
      break;
    case HloOpcode::kSend:
      if (!config_.schedule_send_recvs) {
        return kLowLatency;
      }
      if (target.GetInstr().opcode() == HloOpcode::kSendDone) {
        return get_latency();
      }
      // Cross-slice communication case.
      if (target.GetInstr().opcode() == HloOpcode::kRecvDone) {
        return get_latency();
      }
      break;
    case HloOpcode::kRecv:
      if (!config_.schedule_send_recvs) {
        return kLowLatency;
      }
      if (target.GetInstr().opcode() == HloOpcode::kRecvDone) {
        return get_latency();
      }
      break;
    default:
      break;
  }
  return kLowLatency;
}

LatencyEstimator::TimeCost ProfileGuidedLatencyEstimator::NodeCost(
    const HloInstruction* instr) const {
  static constexpr HloGraphNode::TimeCost kLowCost = 1.0;
  if (instr->IsOutputFusion() || instr->IsLoopFusion() ||
      instr->opcode() == HloOpcode::kConvolution ||
      instr->opcode() == HloOpcode::kWhile) {
    auto it = instr_map_.find(instr->name());
    if (it != instr_map_.end()) {
      VLOG(10) << "PGLE found cost for: " << instr->name();
      return it->second.second;
    }
    VLOG(10) << "PGLE missed cost for: " << instr->name();
    return latency_estimator_->NodeCost(instr);
  }
  return kLowCost;
}

int ProfileGuidedLatencyEstimator::CyclesPerMicrosecond() const {
  return latency_estimator_->CyclesPerMicrosecond();
}

ProfileGuidedLatencyEstimator::ProfileGuidedLatencyEstimator(
    const SchedulerConfig& config,
    std::unique_ptr<LatencyEstimator> latency_estimator,
    const ProfiledInstructionsProto& proto)
    : config_(config), latency_estimator_(std::move(latency_estimator)) {
  const int cycles_per_microsecond = latency_estimator_->CyclesPerMicrosecond();
  for (const auto& instr : proto.instructions()) {
    instr_map_[instr.name()] =
        std::make_pair(instr.timestamp_us() * cycles_per_microsecond,
                       instr.duration_us() * cycles_per_microsecond);
  }
}

}  // namespace xla
