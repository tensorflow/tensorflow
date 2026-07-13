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

#ifndef XLA_SERVICE_GPU_GPU_HLO_ORDERING_H_
#define XLA_SERVICE_GPU_GPU_HLO_ORDERING_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"

namespace xla {
namespace gpu {

// ConcurrentRegionsHloOrdering generalizes SequentialHloOrdering. It separates
// the sequential order into sequential regions. Within each regions, ops can be
// executed concurrently unless data dependencies exist.
class ConcurrentRegionsHloOrdering : public ::xla::SequentialHloOrdering {
 public:
  explicit ConcurrentRegionsHloOrdering(const HloSchedule& schedule);
  explicit ConcurrentRegionsHloOrdering(HloSchedule&& schedule);
  ~ConcurrentRegionsHloOrdering() override = default;

  // Returns nullptr indicating the computation does not have a strictly
  // sequential ordering.
  const HloInstructionSequence* SequentialOrder(
      const HloComputation& computation) const override {
    return nullptr;
  }
  std::string ToString() const override;

  // Returns the id of the region this hlo op belongs to. Regions can contain
  // one or multiple hlo ops.
  std::optional<uint64_t> GetConcurrentRegionId(
      const HloInstruction* hlo) const {
    return concurrent_region_id_.contains(hlo)
               ? std::make_optional(concurrent_region_id_.at(hlo))
               : std::nullopt;
  }

 protected:
  void Initialize();

  bool ExecutesBeforeInSameComputation(const HloInstruction* a,
                                       const HloInstruction* b) const override;

 private:
  absl::flat_hash_map<const HloInstruction*, uint64_t> concurrent_region_id_;
  // The sequence ids of the schedule in sorted order. This allows to iterate
  // over the sequences in deterministic order.
  std::vector<int64_t> sorted_schedule_sequence_ids_;
  absl::flat_hash_map<const HloComputation*,
                      std::unique_ptr<HloReachabilityMap>>
      predecessors_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_HLO_ORDERING_H_
