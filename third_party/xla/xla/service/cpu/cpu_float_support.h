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

#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/xnn_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/float_support.h"

namespace xla {
namespace cpu {

class CpuFloatSupport : public FloatSupport {
 public:
  using DotStrategyChecker = std::function<bool(const HloInstruction& hlo)>;

  explicit CpuFloatSupport(PrimitiveType low_precision_type,
                           DotStrategyChecker call_library_for_dot,
                           TargetMachineFeatures* cpu_features)
      : FloatSupport(low_precision_type),
        call_library_for_dot_(call_library_for_dot),
        cpu_features_(cpu_features) {}

  // Skip trying to upcast the dot if XNNPACK is enabled and the dot is
  // supported by XNNPACK.
  bool ShouldSkipInstruction(const HloInstruction& hlo) const override {
    return hlo.opcode() == HloOpcode::kDot && call_library_for_dot_(hlo) &&
           IsDotSupportedByXnn(hlo.dot_dimension_numbers(),
                               hlo.operand(0)->shape(), hlo.operand(1)->shape(),
                               hlo.shape(), cpu_features_)
               .value_or(false);
  }

  // Makes FloatNormalization skip custom fusion computations for CPU backend.
  bool ShouldSkipComputationsOf(const HloInstruction& hlo) const override {
    return hlo.opcode() == HloOpcode::kFusion &&
           Cast<HloFusionInstruction>(&hlo)->fusion_kind() ==
               HloInstruction::FusionKind::kCustom;
  }

 private:
  DotStrategyChecker call_library_for_dot_;
  TargetMachineFeatures* cpu_features_;
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_CPU_FLOAT_SUPPORT_H_
