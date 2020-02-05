/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_FUSION_NODE_INDEXING_EVALUATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_FUSION_NODE_INDEXING_EVALUATION_H_

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
class FusionNodeIndexingEvaluation {
 public:
  explicit FusionNodeIndexingEvaluation(const HloInstruction* fusion);

  // Evaluate the average number of times an instruction is emitted inside the
  // fusion node, if 'producer' is fused into 'fusion_'.
  int64 EvaluateAverageDuplication(const HloInstruction* producer) const;

  // Evaluate the total number of times an instruction is emitted inside the
  // fusion node, if 'producer' is fused into 'fusion_'. An instruction may be
  // emitted several times, once for each different index value with which it is
  // indexed.
  int64 EvaluateTotalEmittedInstructions(const HloInstruction* producer) const;

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
  absl::flat_hash_map<const HloInstruction*, int64> index_usage_count_;

  // The fusion instruction.
  const HloInstruction* fusion_;

  // The total number of emitted instructions.
  int64 total_emitted_instructions_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_FUSION_NODE_INDEXING_EVALUATION_H_
