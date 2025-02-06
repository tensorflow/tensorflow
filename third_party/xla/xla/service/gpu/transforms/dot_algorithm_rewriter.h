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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_DOT_ALGORITHM_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_DOT_ALGORITHM_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

class DotAlgorithmRewriter : public HloModulePass {
 public:
  DotAlgorithmRewriter() = default;
  absl::string_view name() const override { return "dot-algorithm-rewriter"; }
  using HloPassInterface::Run;

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  static absl::StatusOr<HloInstruction*> MakeMultiplyForBF16BF16F32(
      HloInstruction* lhs, HloInstruction* rhs);
  static absl::StatusOr<HloInstruction*> MakeMultiplyForBF16BF16F32X3(
      HloInstruction* lhs, HloInstruction* rhs);
  static absl::StatusOr<HloInstruction*> MakeMultiplyForBF16BF16F32X6(
      HloInstruction* lhs, HloInstruction* rhs);
  static absl::StatusOr<HloInstruction*> MakeMultiplyForTF32TF32F32(
      HloInstruction* lhs, HloInstruction* rhs);
  static absl::StatusOr<HloInstruction*> MakeMultiplyForTF32TF32F32X3(
      HloInstruction* lhs, HloInstruction* rhs);
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_DOT_ALGORITHM_REWRITER_H_
