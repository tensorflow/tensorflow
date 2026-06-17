/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_FLOAT_SUPPORT_H_
#define XLA_SERVICE_FLOAT_SUPPORT_H_

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/xla_data.pb.h"

namespace xla {

// This class has methods to query if a certain low-precision types, such as
// bfloat16, is supported in certain instructions on a given backend.
// TODO(reedwm): Rename this to NumberSupport, as it supports int4 in additional
// to float types
class FloatSupport {
 public:
  explicit FloatSupport(PrimitiveType low_precision_type,
                        PrimitiveType high_precision_type = F32)
      : low_precision_type_(low_precision_type),
        high_precision_type_(high_precision_type) {}
  virtual ~FloatSupport() = default;

  // The low-precision type. Callers can use this class to query whether the
  // backend supports this type.
  PrimitiveType LowPrecisionType() const { return low_precision_type_; }

  // A high-precision type that should be used in place of the low-precision
  // type if the backend does not support the low-precision type for a certain
  // instruction.
  PrimitiveType HighPrecisionType() const { return high_precision_type_; }

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

  // Returns whether FloatNormalization should skip analyzing the instruction.
  virtual bool ShouldSkipInstruction(const HloInstruction& hlo) const {
    return false;
  }

  // Returns whether FloatNormalization should skip custom fusion computations.
  virtual bool ShouldSkipComputationsOf(const HloInstruction& hlo) const {
    return false;
  }

 private:
  PrimitiveType low_precision_type_;
  PrimitiveType high_precision_type_;
};

}  // namespace xla

#endif  // XLA_SERVICE_FLOAT_SUPPORT_H_
