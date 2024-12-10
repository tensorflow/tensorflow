/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/fusion_deduplication_cache.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/hash/hash.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

namespace {

// HloInstructionPtrHash and HloInstructionPtrEq are a pair of helper classes
// that can be used in absl::flat_hash_map to deduplicate HloInstruction
// pointers that point to identical instructions.
//
// HloInstructionPtrHash uses default AbslHashValue for HloInstruction.
// HloInstructionPtrEq calls into HloInstruction::Identical() and compares
// operand by shape.
//
// It is guaranteed that HloInstruction AbslHashValue returns the same hash for
// equivalent HLO instructions, with respect to HloInstruction::Identical()
// method.
class HloInstructionPtrHash {
 public:
  size_t operator()(const HloInstruction* instr) const {
    return absl::HashOf(*instr);
  }
};

// See comment to HloInstructionPtrHash.
class HloInstructionPtrEq {
 public:
  size_t operator()(const HloInstruction* instr1,
                    const HloInstruction* instr2) const {
    auto operands_eq = [](const HloInstruction* a, const HloInstruction* b) {
      if (a == b) return true;
      return ShapeUtil::Equal(a->shape(), b->shape());
    };

    auto eq_computations = [](const HloComputation* a,
                              const HloComputation* b) { return *a == *b; };

    return instr1->Identical(*instr2, operands_eq, eq_computations);
  }
};

}  // namespace

/*static*/ FusionDeduplicationCache FusionDeduplicationCache::Create(
    const HloModule& module,
    absl::FunctionRef<bool(const HloInstruction&)> is_fusible_fn) {
  absl::flat_hash_map<const HloInstruction*, InstructionId,
                      HloInstructionPtrHash, HloInstructionPtrEq>
      deduplicated_id_map;
  absl::flat_hash_map<const HloInstruction*, InstructionId> instruction_id_map;

  int64_t instruction_count = module.instruction_count();
  deduplicated_id_map.reserve(instruction_count);
  instruction_id_map.reserve(instruction_count);

  int64_t next_id = 0;
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      if (!is_fusible_fn(*instruction)) {
        continue;
      }

      auto it = deduplicated_id_map.emplace(instruction, next_id);
      if (it.second) {
        ++next_id;
      }

      instruction_id_map[instruction] = it.first->second;
    }
  }

  return FusionDeduplicationCache(next_id, std::move(instruction_id_map));
}

FusionDeduplicationCache::InstructionId
FusionDeduplicationCache::GetInstructionId(const HloInstruction& instruction) {
  return instruction_id_map_.at(&instruction);
}

FusionDeduplicationCache::FusionId FusionDeduplicationCache::GetFusionId(
    const HloInstruction& producer, const HloInstruction& consumer,
    int64_t consumer_operand_index) {
  FusionDeduplicationCache::FusionId fusion_id{GetInstructionId(producer),
                                               GetInstructionId(consumer),
                                               consumer_operand_index};

  if (fusion_id_map_.emplace(fusion_id, next_id_).second) {
    ++next_id_;
  }

  return fusion_id;
}

FusionDeduplicationCache::FusionId FusionDeduplicationCache::GetFusionId(
    const HloInstruction& producer, const HloInstruction& consumer) {
  return GetFusionId(producer, consumer, consumer.operand_index(&producer));
}

void FusionDeduplicationCache::UpdateFusedInstructionId(
    const HloInstruction& fusion_instruction,
    const HloInstruction& original_producer,
    const HloInstruction& original_consumer, int64_t consumer_operand_index) {
  instruction_id_map_[&fusion_instruction] = fusion_id_map_.at(GetFusionId(
      original_producer, original_consumer, consumer_operand_index));
}

}  // namespace gpu
}  // namespace xla
