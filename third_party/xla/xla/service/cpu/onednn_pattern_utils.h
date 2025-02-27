/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_ONEDNN_PATTERN_UTILS_H_
#define XLA_SERVICE_CPU_ONEDNN_PATTERN_UTILS_H_
#if defined(INTEL_MKL)

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/pattern_matcher.h"

namespace xla {
namespace cpu {

namespace onednn_pattern_utils_internal {
namespace m = match;

template <typename Pattern>
auto OptionalConvert(Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Convert(pattern), std::move(pattern));
}

template <typename Pattern>
auto OptionalBroadcast(Pattern pattern) {
  return m::AnyOf<HloInstruction>(m::Broadcast(pattern), std::move(pattern));
}

// Type conversion from and to any of BF16 and FP32.
// TODO(intel-tf): Support more types when enabled.
template <typename Pattern>
inline auto SupportedConvert(Pattern pattern) {
  auto supported_convert = [](const HloInstruction* instr) -> bool {
    return IsSupportedType(instr->shape().element_type()) &&
           IsSupportedType(instr->operand(0)->shape().element_type());
  };
  return m::Convert(pattern).WithPredicate(supported_convert);
}

template <typename Pattern>
inline auto SupportedConvert(HloInstruction** convert, Pattern pattern) {
  auto supported_convert = [](const HloInstruction* instr) -> bool {
    return IsSupportedType(instr->shape().element_type()) &&
           IsSupportedType(instr->operand(0)->shape().element_type());
  };
  return m::Convert(convert, pattern).WithPredicate(supported_convert);
}
}  // namespace onednn_pattern_utils_internal
}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL
#endif  // XLA_SERVICE_CPU_ONEDNN_PATTERN_UTILS_H_
