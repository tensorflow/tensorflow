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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DOT_ALGORITHM_REWRITER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DOT_ALGORITHM_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// DotAlgorithmRewriter is an HLO pass that rewrites Dot operations marked
// with specific algorithm precision config enums into a sequence of Dot
// operations that emulate higher precision accumulation using lower precision
// operations. For example, it can rewrite an F32 dot marked as
// BF16_BF16_F32_X3 into three BF16xBF16->F32 dots to achieve higher precision.
//
// This is useful for hardware that has much higher throughput for
// lower-precision types like BF16 or TF32 than for F32. By decomposing a
// single F32 operation into multiple lower-precision ones, we can achieve
// higher precision than a single truncated lower-precision operation, and
// potentially higher performance than a native F32 operation, thus allowing a
// tradeoff between performance and precision.
//
// This pass rewrites HloOpcode::kDot instructions based on their
// PrecisionConfig algorithm enum (e.g. ALG_DOT_BF16_BF16_F32_X3).
//
// This class also provides static methods to emulate elementwise multiplication
// with higher precision (e.g. MakeMultiplyForBF16BF16F32X3). These are used
// to emit equivalent logic for HloOpcode::kMultiply, for example in fusion
// emitters like Triton.
class DotAlgorithmRewriter : public HloModulePass {
 public:
  DotAlgorithmRewriter() = default;
  absl::string_view name() const override { return "dot-algorithm-rewriter"; }

  static absl::StatusOr<HloInstruction*> MakeMultiplyForBF16BF16F32(
      HloInstruction* lhs, HloInstruction* rhs);
  static absl::StatusOr<HloInstruction*> MakeMultiplyForBF16BF16F32X3(
      HloInstruction* lhs, HloInstruction* rhs);
  static absl::StatusOr<HloInstruction*> MakeMultiplyForBF16BF16F32X6(
      HloInstruction* lhs, HloInstruction* rhs);
  static absl::StatusOr<HloInstruction*> MakeMultiplyForBF16BF16F32X9(
      HloInstruction* lhs, HloInstruction* rhs);
  static absl::StatusOr<HloInstruction*> MakeMultiplyForTF32TF32F32(
      HloInstruction* lhs, HloInstruction* rhs);
  static absl::StatusOr<HloInstruction*> MakeMultiplyForTF32TF32F32X3(
      HloInstruction* lhs, HloInstruction* rhs);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DOT_ALGORITHM_REWRITER_H_
