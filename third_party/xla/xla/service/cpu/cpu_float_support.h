/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_CPU_FLOAT_SUPPORT_H_
#define XLA_SERVICE_CPU_CPU_FLOAT_SUPPORT_H_

#include <functional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/float_support.h"

namespace xla {
namespace cpu {

class CpuFloatSupport : public FloatSupport {
 public:
  using DotStrategyChecker = std::function<bool(const HloInstruction& hlo)>;

  explicit CpuFloatSupport(PrimitiveType low_precision_type,
                           DotStrategyChecker call_library_for_dot)
      : FloatSupport(low_precision_type),
        call_library_for_dot_(call_library_for_dot) {}

  // A hatch to skip FloatNormalization for certain instructions.
  bool ShouldSkipInstruction(const HloInstruction& hlo) const override {
    return hlo.opcode() == HloOpcode::kDot && call_library_for_dot_(hlo);
  }

 private:
  DotStrategyChecker call_library_for_dot_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_CPU_FLOAT_SUPPORT_H_
