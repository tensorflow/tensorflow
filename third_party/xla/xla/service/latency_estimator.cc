/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/latency_estimator.h"

#include <cstdint>
#include <optional>

#include "xla/frontend_attributes.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {

CanonicalAsyncOp DefaultGetCanonicalAsyncOp(const HloInstruction& hlo) {
  switch (hlo.opcode()) {
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncDone:
      return {hlo.opcode(), hlo.async_wrapped_opcode()};
    case HloOpcode::kAllReduceStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kAllReduce};
    case HloOpcode::kAllGatherStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kAllGather};
    case HloOpcode::kCollectivePermuteStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kCollectivePermute};
    case HloOpcode::kCopyStart:
      return {HloOpcode::kAsyncStart, HloOpcode::kCopy};
    case HloOpcode::kCopyDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kCopy};
    case HloOpcode::kAllReduceDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kAllReduce};
    case HloOpcode::kAllGatherDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kAllGather};
    case HloOpcode::kCollectivePermuteDone:
      return {HloOpcode::kAsyncDone, HloOpcode::kCollectivePermute};
    default:
      return {hlo.opcode(), hlo.opcode()};
  }
}

bool LatencyEstimator::IsAsyncPair(const HloInstruction& from,
                                   const HloInstruction& target) const {
  CanonicalAsyncOp from_op = GetCanonicalAsyncOp(from);
  CanonicalAsyncOp target_op = GetCanonicalAsyncOp(target);
  return from_op.outer == HloOpcode::kAsyncStart &&
         target_op.outer == HloOpcode::kAsyncDone &&
         from_op.inner == target_op.inner;
}

bool LatencyEstimator::IsP2pPair(const HloInstruction& from,
                                 const HloInstruction& target) const {
  return (from.opcode() == HloOpcode::kSend &&
          target.opcode() == HloOpcode::kSendDone) ||
         (from.opcode() == HloOpcode::kRecv &&
          target.opcode() == HloOpcode::kRecvDone);
}

std::optional<LatencyEstimator::TimeCost>
LatencyEstimator::GetLatencyFromMetadata(
    const HloInstruction& instruction) const {
  const std::optional<int64_t> latency_ns =
      frontend::GetLatencyMetadataNs(instruction);
  if (!latency_ns.has_value()) {
    return std::nullopt;
  }
  return static_cast<TimeCost>(*latency_ns) * CyclesPerMicrosecond() / 1000.0;
}

}  // namespace xla
