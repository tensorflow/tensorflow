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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_SUPPORT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_SUPPORT_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"

namespace xla {

class BFloat16Support {
 public:
  BFloat16Support() {}
  virtual ~BFloat16Support() {}

  // Returns whether the backend supports BF16 operand for the HLO instruction
  // at the given index.
  virtual bool SupportsBF16Operand(const HloInstruction& hlo,
                                   int64_t operand_index) const;

  // Returns whether the backend supports BF16 output for the HLO instruction.
  virtual bool SupportsBF16Output(const HloInstruction& hlo) const;

  // Returns whether the backend support mixed precision: the operands, output,
  // and parameters/output of the called computations can have different
  // precisions (BF16 and F32).
  virtual bool SupportsMixedPrecisions(const HloInstruction& hlo) const;

  // Returns whether the given HLO preserves its BF16 operand precision at the
  // given index, so even if the output is F32, elements in the output that
  // depend on the BF16 operand will still have BF16 effective precision even if
  // they have F32 format. Similarly, this also means if the output is BF16 then
  // increasing the operand precision from BF16 to F32 will not change the
  // output. This typically includes HLOs that pass elements from the operand to
  // the output without arithmetic operations.
  static bool EffectiveOperandPrecisionIsOutputPrecision(
      const HloInstruction& hlo, int64_t operand_index);

  // Returns if the backend only uses BF16 precision for the operand at the
  // specified index, even if the operand is F32.
  virtual bool EffectiveOperandPrecisionIsBF16(const HloInstruction& hlo,
                                               int64_t operand_index) const;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_SUPPORT_H_
