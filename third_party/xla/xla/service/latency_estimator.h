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

#ifndef XLA_SERVICE_LATENCY_ESTIMATOR_H_
#define XLA_SERVICE_LATENCY_ESTIMATOR_H_

#include <functional>
#include <optional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {

struct CanonicalAsyncOp {
  HloOpcode outer;  // kAsyncStart or kAsyncDone
  HloOpcode inner;  // kAllReduce, kAllGather, kAllToAll, kCollectiveBroadcast,
                    // kCollectivePermute, or kReduceScatter
};

CanonicalAsyncOp DefaultGetCanonicalAsyncOp(const HloInstruction& hlo);

using GetCanonicalAsyncOpFunc =
    std::function<CanonicalAsyncOp(const HloInstruction& hlo)>;

// Defined in latency_hiding_scheduler.h.
class HloGraphNode;

// Class used to estimate latency between instructions and cost of HLOs.
class LatencyEstimator {
 public:
  using TimeCost = double;
  // Uses the approximate or cost model function for GetLatencyBetween based on
  // a flag.
  virtual TimeCost GetLatencyBetween(const HloGraphNode& from,
                                     const HloGraphNode& target) const = 0;
  // Uses the approximate or cost model function for NodeCost based on a flag.
  virtual TimeCost NodeCost(const HloInstruction* node) const = 0;
  // Returns the core frequency used in latency estimation.
  virtual int CyclesPerMicrosecond() const = 0;

  // Returns the latency cycles of the instruction if it is specified in the
  // frontend attributes. Otherwise, returns std::nullopt.
  std::optional<TimeCost> GetLatencyFromMetadata(
      const HloInstruction& instruction) const;

  virtual ~LatencyEstimator() = default;

  CanonicalAsyncOp GetCanonicalAsyncOp(const HloInstruction& hlo) const {
    return get_canonical_async_op_(hlo);
  }
  // Returns true if `from` is an async start and `target` is its matching
  // async done kind (same canonical inner opcode).
  bool IsAsyncPair(const HloInstruction& from,
                   const HloInstruction& target) const;
  // Returns true if `from` -> `target` is a Send -> SendDone or
  // Recv -> RecvDone pair.
  bool IsP2pPair(const HloInstruction& from,
                 const HloInstruction& target) const;
  // HloGraphNode overloads of the above. Defined inline in
  // latency_hiding_scheduler.h, which owns the HloGraphNode definition.
  bool IsAsyncPair(const HloGraphNode& from, const HloGraphNode& target) const;
  bool IsP2pPair(const HloGraphNode& from, const HloGraphNode& target) const;
  explicit LatencyEstimator(
      GetCanonicalAsyncOpFunc func = DefaultGetCanonicalAsyncOp)
      : get_canonical_async_op_(func) {}

 private:
  GetCanonicalAsyncOpFunc get_canonical_async_op_;
};

}  // namespace xla

#endif  // XLA_SERVICE_LATENCY_ESTIMATOR_H_
