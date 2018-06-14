/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MULTI_OUTPUT_FUSION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MULTI_OUTPUT_FUSION_H_

#include "tensorflow/compiler/xla/service/multi_output_fusion.h"

namespace xla {
namespace gpu {

// Multi-output fusion of sibling and producer-consumer instructions for the
// Jellyfish backend.
class GpuMultiOutputFusion : public MultiOutputFusion {
 public:
  GpuMultiOutputFusion();

 protected:
  // Test if instr1 and instr2 have the compatible shapes that can be legally
  // fused.
  bool ShapesCompatibleForFusion(HloInstruction* instr1,
                                 HloInstruction* instr2) override;

  // We currently only consider reduce and reduce fusion nodes as candidates.
  bool IsFusible(HloInstruction* instr) override;

  // This function estimates the amount of memory reads saved by merging
  // instr1 and instr2 into one multi-output fusion instruction. For a fusion
  // instruction, all the operands need to be loaded from memory. If we merge
  // instr1 and instr2, common operands will not be loaded twice. The profit is
  // estimated as the size of the common operands b/w instr1 and instr2.
  int64 GetProfit(HloInstruction* instr1, HloInstruction* instr2) override;

  // Test if it's legal to fuse instr1 and instr2 into one fusion instruction.
  bool LegalToFuse(HloInstruction* instr1, HloInstruction* instr2) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MULTI_OUTPUT_FUSION_H_
