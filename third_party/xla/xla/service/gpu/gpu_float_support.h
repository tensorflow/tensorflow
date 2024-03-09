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

#ifndef XLA_SERVICE_GPU_GPU_FLOAT_SUPPORT_H_
#define XLA_SERVICE_GPU_GPU_FLOAT_SUPPORT_H_

#include <cstdint>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/float_support.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

class GpuFloatSupport : public FloatSupport {
 public:
  explicit GpuFloatSupport(PrimitiveType low_precision_type,
                           PrimitiveType high_precision_type = F32)
      : FloatSupport(low_precision_type, high_precision_type) {}

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    return FloatSupport::SupportsLowPrecisionOperand(hlo, operand_index) ||
           IsSupported(hlo);
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    return FloatSupport::SupportsLowPrecisionOutput(hlo) || IsSupported(hlo);
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override;

 private:
  bool IsSupported(const HloInstruction& hlo) const;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_FLOAT_SUPPORT_H_
