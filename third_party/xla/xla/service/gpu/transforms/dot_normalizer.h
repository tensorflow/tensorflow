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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_DOT_NORMALIZER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_DOT_NORMALIZER_H_

#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"
#include "xla/util.h"

namespace xla::gpu {

// Ensures that a dot has at least 1 contracting dimension. If there are no
// contracting dimensions, a trivial 1-sized contracting dimension is added.
// This pass is expected to be run after layout assignment.
class DotNormalizer : public OpExpanderPass {
 public:
  explicit DotNormalizer(HloPredicate extra_filter = nullptr)
      : OpExpanderPass(std::move(extra_filter)) {}

  absl::string_view name() const override { return "dot_normalizer"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_DOT_NORMALIZER_H_
