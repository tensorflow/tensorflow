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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_TRANSPOSE_FOLDING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_TRANSPOSE_FOLDING_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// HLO pass that folds transpose operators into Dot operators, where the Dot
// operator is implemented by a GEMM kernel that can transpose its inputs.
class TransposeFolding : public HloModulePass {
 public:
  using OperandIndices = std::vector<int64>;

  // Returns the set of foldable operands for a given HLO and some candidate
  // operands.
  using FoldableOperands = std::function<OperandIndices(const HloInstruction&,
                                                        const OperandIndices&)>;
  using TransposableGemmOperandsFn = FoldableOperands;
  using TransposableConvOperandsFn = FoldableOperands;

  // Helper function to explicitly not fold transposes.
  static OperandIndices NeverFoldTranspose(const HloInstruction&,
                                           const OperandIndices&) {
    return {};
  }
  // transposable_gemm_operands returns the set of operands it wants to fold if
  // the instruction argument is implemented as a GEMM kernel that supports
  // transposing its arguments.
  //
  // transposable_conv_operands returns the set of operands it wants to fold if
  // the instruction argument is implemented as a convolution that supports
  // transposing its arguments.
  explicit TransposeFolding(
      TransposableGemmOperandsFn transposable_gemm_operands,
      TransposableConvOperandsFn transposable_conv_operands);
  absl::string_view name() const override { return "transpose-folding"; }

  StatusOr<bool> Run(HloModule* module) override;

 private:
  TransposableGemmOperandsFn transposable_gemm_operands_;
  TransposableConvOperandsFn transposable_conv_operands_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TRANSPOSE_FOLDING_H_
