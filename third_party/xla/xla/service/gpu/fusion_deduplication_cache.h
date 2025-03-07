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

#ifndef XLA_SERVICE_GPU_FUSION_DEDUPLICATION_CACHE_H_
#define XLA_SERVICE_GPU_FUSION_DEDUPLICATION_CACHE_H_

#include <cstdint>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {

// A cache that helps to track identical HLO instructions and their fusions. The
// cache assigns an InstructionId to each instruction. Instructions that are the
// same in terms of `HloInstruction::Identical` have the same id. The cache
// operates with HloInstruction pointers and does not dereference them when
// retrieving/updating a FusionId, except one method which has a corresponding
// comment.
//
// The id depends on the fusion order. If we have the following chain of HLO
// instructions:
//
//   a = instr1()
//   b = instr2(a)
//   c = instr3(b)
//
// It depends if we fuse `a` and `b` first or `b` and `c` first.
// `id(fuse(fuse(a, b), c))` != `id(fuse(a, fuse(b, c)))`
//
// This is usually not a problem in practice and allows us to catch most of the
// cases, because similar HLO instructions are usually fused together in the
// same order.
class FusionDeduplicationCache {
 public:
  // An id for an HLO instruction.
  using InstructionId = int64_t;

  // An id for a fusion of `producer` and `consumer`. Values in the tuple should
  // not be interpreted by API users and are subject to change. The id should be
  // used as an opaque key to compare different fusions.
  using FusionId =
      std::tuple</*producer=*/InstructionId, /*consumer=*/InstructionId,
                 /*operand_index=*/int64_t, /*allow_multi_output=*/bool>;

  // Initializes the cache for all fusible instructions in the given module.
  // `is_fusible_fn` callback returns true if the instruction is fusible.
  // Identical HLO instructions (in terms of `HloInstruction::Identical`) will
  // be assigned the same id.
  static FusionDeduplicationCache Create(
      const HloModule& module,
      absl::FunctionRef<bool(const HloInstruction&)> is_fusible_fn);

  // Returns the id for the given instruction. The instruction should have an id
  // already assigned, either during the initialization process in `Create` or
  // manually after the fusion by `SetFusedInstructionId`.
  InstructionId GetInstructionId(const HloInstruction* instruction);

  // Returns the id for the fusion of `producer` and `consumer`.
  // `allow_multi_output` should be set to true if we allow to create a
  // multi-output fusion to avoid having to duplicate `producer`. `consumer`
  // should not be deleted yet.
  FusionId GetFusionId(const HloInstruction* producer,
                       const HloInstruction* consumer,
                       bool allow_multi_output = false);

  // Sets the new id for the `fusion_instruction`.
  //
  // The `fusion_instruction` should be the result of fusing `original_producer`
  // and `original_consumer`. It can happen that `fusion_instruction` is equal
  // to `original_consumer`. That means that `producer` was fused into
  // `consumer` fusion and `fusion_instruction` gets a new id.
  //
  // `consumer_operand_index` is the operand index of `original_producer` in
  // `original_consumer`.
  //
  // The operand index needs to be obtained before the fusion happened and
  // provided explicitly, because at this point `original_producer` and
  // `original_consumer` have been modified and became disconnected.
  void UpdateFusedInstructionId(const HloInstruction* fusion_instruction,
                                const HloInstruction* original_producer,
                                const HloInstruction* original_consumer,
                                int64_t consumer_operand_index,
                                bool allow_multi_output = false);

 private:
  FusionDeduplicationCache(
      int64_t next_id, absl::flat_hash_map<const HloInstruction*, InstructionId>
                           instruction_id_map)
      : next_id_(next_id), instruction_id_map_(std::move(instruction_id_map)) {}

  FusionId GetFusionId(const HloInstruction* producer,
                       const HloInstruction* consumer,
                       int64_t consumer_operand_index, bool allow_multi_output);

  int64_t next_id_ = 0;

  // A map from an HLO instruction pointers to ids.
  absl::flat_hash_map<const HloInstruction*, InstructionId> instruction_id_map_;

  // A map from producer-consumer fusions to ids. After `producer` and
  // `consumer` are fused, the id of the resulting fusion instruction will be
  // equal to the id from this map.
  absl::flat_hash_map<FusionId, InstructionId> fusion_id_map_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSION_DEDUPLICATION_CACHE_H_
