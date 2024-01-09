/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_CONVOLUTION_PRED_EXPANDER_H_
#define XLA_SERVICE_CONVOLUTION_PRED_EXPANDER_H_

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/op_expander_pass.h"
#include "xla/statusor.h"

namespace xla {

// A pass that rewrites boolean convolutions to floating point and converts the
// result back to boolean. This is necessary, as the convolutions on GPUs are
// implemented using custom call to cuDNN, which only supports FP and S8 inputs.
class ConvolutionPredExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override {
    return "convolution-pred-expander";
  }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_CONVOLUTION_PRED_EXPANDER_H_
