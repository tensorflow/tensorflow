/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_FUSION_NODE_INDEXING_EVALUATION_H_
#define XLA_SERVICE_FUSION_NODE_INDEXING_EVALUATION_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/types.h"

namespace xla {
class FusionNodeIndexingEvaluation {
 public:
  explicit FusionNodeIndexingEvaluation(const HloInstruction* fusion,
                                        int64_t root_usage_count = 1);

  // Evaluate the number of times 'producer' would be emitted if it is fused
  // into 'fusion_'. If the duplication is "too high" (some arbitrary chosen
  // constant), returns true.
  bool CodeDuplicationTooHigh(const HloInstruction* producer) const;

  // Evaluate the maximum code duplication inside the fusion node. If the
  // maximum code duplication is "too high" (some arbitrary chosen constant),
  // returns true.
  bool MaxCodeDuplicationTooHigh() const;

  // Evaluate the number of times 'producer' would be emitted if it is fused
  // into 'fusion_'.
  int64_t EvaluateEmittedInstructions(const HloInstruction* producer) const;

  // Update the evaluation cache after having fused 'producer' into 'fusion_'.
  // 'producer' is the cloned instruction which is now part of the fusion
  // computation. 'indexing_users_of_producer' are the direct or indirect users
  // of 'producer' which pass index values created by them.
  void UpdateEvaluationCache(
      const HloInstruction* producer,
      absl::flat_hash_set<const HloInstruction*> indexing_users_of_producer);

  // Prior to fusing, we need to erase the indexing_users_ entry of the
  // producer to be fused, because the HloInstruction pointer will be
  // invalidated. We return the set of direct or indirect users which pass index
  // values created by them to the fusion parameter corresponding to this
  // producer. This will be needed for updating the evaluation cache (see
  // UpdateEvaluationCache).
  absl::flat_hash_set<const HloInstruction*> RemoveFusionOperand(
      HloInstruction* fusion_operand);

 private:
  // We don't want to have too much code duplication, because it slows down the
  // compilation time. There is a tradeoff between compilation time and runtime.
  // This constant defines the maximum amount of times that we allow to emit the
  // same op (indexed with different index values).
  static const int64_t kAllowedCodeDuplication;

  // Computes the 'indexing_users_' and 'index_usage_count_' maps based on the
  // current instructions inside the fusion node. Also updates
  // 'total_emitted_instructions_' accordingly.
  void RecomputeCache();

  // Computes the 'index_usage_count_' entry for 'instruction'.
  void UpdateIndexUsageCount(const HloInstruction* instruction);

  // Updates the 'indexing_users_' entry of the operands of 'instruction'.
  void UpdateIndexingUsersOfOperands(const HloInstruction* instruction);

  // Collects for each instruction in a fusion node from which direct or
  // indirect users newly created index values are passed. Roughly speaking, we
  // reuse index values if the shapes are equal when ignoring the element type
  // (we may reuse also if the shape change is a bitcast, but we don't consider
  // that here). By ignoring potential reuses our estimate of which instruction
  // generates a new index value is a bit more conservative than necessary.
  absl::flat_hash_map<const HloInstruction*,
                      absl::flat_hash_set<const HloInstruction*>>
      indexing_users_;

  // Stores the number of different index accesses for each instruction in a
  // fusion node. The fusion emitter caches access with the same index, so this
  // value indicates how many times a specific instruction will be emitted.
  absl::flat_hash_map<const HloInstruction*, int64_t> index_usage_count_;

  // The fusion instruction.
  const HloInstruction* fusion_;
};
}  // namespace xla

#endif  // XLA_SERVICE_FUSION_NODE_INDEXING_EVALUATION_H_
