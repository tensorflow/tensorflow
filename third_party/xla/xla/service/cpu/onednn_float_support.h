/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_ONEDNN_FLOAT_SUPPORT_H_
#define XLA_SERVICE_CPU_ONEDNN_FLOAT_SUPPORT_H_

#if defined(INTEL_MKL)

#include "xla/service/float_support.h"

namespace xla {
namespace cpu {

class OneDnnFloatSupport : public FloatSupport {
 public:
  explicit OneDnnFloatSupport(PrimitiveType low_precision_type)
      : FloatSupport(low_precision_type) {}

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    return FloatSupport::SupportsLowPrecisionOperand(hlo, operand_index) ||
           IsSupported(hlo);
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    return FloatSupport::SupportsLowPrecisionOutput(hlo) || IsSupported(hlo);
  }

 private:
  bool IsSupported(const HloInstruction& hlo) const;
  // Performs early check for things that cannot be delayed becuase some later
  // passes may change the shape of dot inputs.
  bool DotSupported(const HloInstruction& hlo) const;
};

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL
#endif  // XLA_SERVICE_CPU_ONEDNN_FLOAT_SUPPORT_H_
