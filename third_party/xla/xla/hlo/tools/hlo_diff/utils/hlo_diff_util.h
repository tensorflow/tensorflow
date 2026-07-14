/*
 * Copyright 2025 The OpenXLA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_HLO_TOOLS_HLO_DIFF_UTILS_HLO_DIFF_UTIL_H_
#define XLA_HLO_TOOLS_HLO_DIFF_UTILS_HLO_DIFF_UTIL_H_

#include <cstdint>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "tsl/platform/fingerprint.h"

namespace xla::hlo_diff {

inline uint64_t GetHloInstructionFingerprint(
    const HloInstruction* instruction,
    const HloPrintOptions& hlo_print_options) {
  return tsl::Fingerprint64(instruction->ToString(hlo_print_options));
}

inline uint64_t GetHloInstructionFingerprint(
    const HloInstruction* instruction) {
  return GetHloInstructionFingerprint(instruction,
                                      HloPrintOptions::Fingerprint());
}

}  // namespace xla::hlo_diff

#endif  // XLA_HLO_TOOLS_HLO_DIFF_UTILS_HLO_DIFF_UTIL_H_
