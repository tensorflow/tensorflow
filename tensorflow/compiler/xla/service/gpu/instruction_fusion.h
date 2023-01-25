/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_INSTRUCTION_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_INSTRUCTION_FUSION_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"

namespace xla {
namespace gpu {

class GpuInstructionFusion : public InstructionFusion {
 public:
  explicit GpuInstructionFusion(bool may_duplicate, const GpuDeviceInfo& d)
      : InstructionFusion(GpuInstructionFusion::IsExpensive, may_duplicate),
        device_info_(d) {}

  static bool IsExpensive(const HloInstruction& instruction);

  using HloPassInterface::Run;
  StatusOr<bool> Run(HloModule* module,
                     const absl::flat_hash_set<absl::string_view>&
                         execution_threads) override {
    fusion_node_evaluations_.clear();
    return InstructionFusion::Run(module, execution_threads);
  }

 protected:
  FusionDecision ShouldFuse(HloInstruction* consumer,
                            int64_t operand_index) override;

  HloInstruction::FusionKind ChooseKind(
      const HloInstruction* producer, const HloInstruction* consumer) override;

  // Return computations on which to run Fusion. We explicitly filter out
  // softmax custom-call computations.
  std::vector<HloComputation*> GetFusionComputations(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // This method is called by ShouldFuse() to do all the computationally
  // inexpensive checks whether we should fuse the operand into 'consumer'.
  FusionDecision ShouldFuseInexpensiveChecks(HloInstruction* consumer,
                                             int64_t operand_index);

  HloInstruction* FuseInstruction(HloInstruction* fusion_instruction,
                                  HloInstruction* producer) override;

  // Keep track of the number of times each instruction inside a fusion node is
  // indexed with different index vectors.
  absl::flat_hash_map<const HloInstruction*, FusionNodeIndexingEvaluation>
      fusion_node_evaluations_;

  const GpuDeviceInfo device_info_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_INSTRUCTION_FUSION_H_
