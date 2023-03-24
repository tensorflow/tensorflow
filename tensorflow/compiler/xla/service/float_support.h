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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_FLOAT_SUPPORT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_FLOAT_SUPPORT_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// This class has methods to query if a certain low-precision floating-point
// type, such as bfloat16, is supported in certain instructions on a given
// backend.
class FloatSupport {
 public:
  explicit FloatSupport(PrimitiveType low_precision_type)
      : low_precision_type_(low_precision_type) {}
  virtual ~FloatSupport() = default;

  // The low-precision type. Callers can use this class to query whether the
  // backend supports this type.
  PrimitiveType LowPrecisionType() const { return low_precision_type_; }

  // A high-precision type that should be used in place of the low-precision
  // type if the backend does not support the low-precision type for a certain
  // instruction.
  PrimitiveType HighPrecisionType() const {
    if (low_precision_type_ == F8E5M2 || low_precision_type_ == F8E4M3FN) {
      return F16;
    }
    DCHECK_EQ(low_precision_type_, BF16);
    return F32;
  }

  // Returns whether the backend supports a low-precision operand for the HLO
  // instruction at the given index.
  virtual bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                           int64_t operand_index) const;

  // Returns whether the backend supports a low-precision output for the HLO
  // instruction.
  virtual bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const;

  // Returns whether the backend support mixed precision: the operands, output,
  // and parameters/output of the called computations can have different
  // precisions (both the low-precision and the high-precision types).
  virtual bool SupportsMixedPrecisions(const HloInstruction& hlo) const;

  // Returns whether the given HLO preserves its low-precision operand precision
  // at the given index, so even if the output is the high-precision type,
  // elements in the output that depend on the low-precision operand will still
  // effectively have low precision even if they are in the high-precision
  // format. Similarly, this also means if the output is low-precision then
  // increasing the operand precision from the low-precision type to the
  // high-precision type will not change the output. This typically includes
  // HLOs that pass elements from the operand to the output without arithmetic
  // operations.
  static bool EffectiveOperandPrecisionIsOutputPrecision(
      const HloInstruction& hlo, int64_t operand_index);

  // Returns if the backend only uses low precision for the operand at the
  // specified index, even if the operand is in the high-precision type.
  virtual bool EffectiveOperandPrecisionIsLowPrecision(
      const HloInstruction& hlo, int64_t operand_index) const;

 private:
  PrimitiveType low_precision_type_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_FLOAT_SUPPORT_H_
