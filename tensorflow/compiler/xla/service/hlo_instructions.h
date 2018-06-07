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

// All HloInstruction subclasses are put in this file.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTIONS_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {

class HloBatchNormInstruction : public HloInstruction {
 public:
  // Returns feature_index field associated with the instruction. The index
  // represents the index of the feature dimension.
  int64 feature_index() const { return feature_index_; }

  // Returns a epsilon value associated with the instruction. The is a small
  // number added to the variance to avoid divide-by-zero error.
  float epsilon() const { return epsilon_; }

  // Returns string representation of op-specific attributes.
  std::vector<string> ExtraAttributesToString(
      const HloPrintOptions& options) const override;

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 protected:
  HloBatchNormInstruction(HloOpcode opcode, const Shape& shape,
                          HloInstruction* operand, HloInstruction* scale,
                          float epsilon, int64 feature_index);

 private:
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // A small float number added to the variance to avoid divide-by-zero error.
  float epsilon_ = 0.0f;

  // An integer value representing the index of the feature dimension.
  int64 feature_index_ = -1;
};

class HloBatchNormTrainingInstruction : public HloBatchNormInstruction {
 public:
  HloBatchNormTrainingInstruction(const Shape& shape, HloInstruction* operand,
                                  HloInstruction* scale, HloInstruction* offset,
                                  float epsilon, int64 feature_index);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;
};

class HloBatchNormInferenceInstruction : public HloBatchNormInstruction {
 public:
  HloBatchNormInferenceInstruction(const Shape& shape, HloInstruction* operand,
                                   HloInstruction* scale,
                                   HloInstruction* offset, HloInstruction* mean,
                                   HloInstruction* variance, float epsilon,
                                   int64 feature_index);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;
};

class HloBatchNormGradInstruction : public HloBatchNormInstruction {
 public:
  HloBatchNormGradInstruction(const Shape& shape, HloInstruction* operand,
                              HloInstruction* scale, HloInstruction* mean,
                              HloInstruction* variance,
                              HloInstruction* grad_output, float epsilon,
                              int64 feature_index);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTIONS_H_
