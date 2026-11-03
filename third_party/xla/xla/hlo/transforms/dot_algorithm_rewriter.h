/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_DOT_ALGORITHM_REWRITER_H_
#define XLA_HLO_TRANSFORMS_DOT_ALGORITHM_REWRITER_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Rewrites a BF16 or F32 dot with ALG_DOT_BF16_BF16_FP8X3 or
// ALG_DOT_BF16_BF16_FP8X4 into per channel power of two scaling and 3 or 4
// F8E4M3FN dots accumulated in F32. `f8_dot_precision` is the operand precision
// of the emitted F8E4M3FN dots. Returns false and leaves `dot` unchanged for
// other algorithms, operand types, or dots without contracting dimensions.
absl::StatusOr<bool> RewriteFp8xNDot(
    HloInstruction* dot, PrecisionConfig::Precision f8_dot_precision);

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_DOT_ALGORITHM_REWRITER_H_
