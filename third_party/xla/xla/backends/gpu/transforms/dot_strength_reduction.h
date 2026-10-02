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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"
#include "xla/stream_executor/device_description.h"

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DOT_STRENGTH_REDUCTION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DOT_STRENGTH_REDUCTION_H_

namespace xla {
namespace gpu {

// This pass strength-reduces dot operations into simpler operations.
// This transformation is applied to:
// - One-hot dots into gather and select.
// - The following dots into broadcast, multiply, and reduce:
//   - All vector * vector dots.
//   - Some vector * matrix dots (depending on size).
//   - s32 * s32 -> s32 dots with RHS contracting dimension 1 (unsupported by
//     Triton and slow in the loop emitter).
class DotStrengthReduction : public OpExpanderPass {
 public:
  explicit DotStrengthReduction(se::GpuComputeCapability compute_capability,
                                bool enable_one_hot_dot_to_gather = false)
      : compute_capability_(compute_capability),
        enable_one_hot_dot_to_gather_(enable_one_hot_dot_to_gather) {}
  absl::string_view name() const override { return "dot-strength-reduction"; }

 private:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;
  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;

  se::GpuComputeCapability compute_capability_;
  bool enable_one_hot_dot_to_gather_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DOT_STRENGTH_REDUCTION_H_
