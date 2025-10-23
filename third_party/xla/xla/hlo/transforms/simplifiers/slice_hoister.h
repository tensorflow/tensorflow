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

<<<<<<< HEAD:third_party/xla/xla/service/gpu/transforms/scaled_dot_rewriter.h
#ifndef XLA_SERVICE_GPU_TRANSFORMS_SCALED_DOT_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_SCALED_DOT_REWRITER_H_
=======
#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_SLICE_HOISTER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_SLICE_HOISTER_H_
>>>>>>> upstream/master:third_party/xla/xla/hlo/transforms/simplifiers/slice_hoister.h

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

<<<<<<< HEAD:third_party/xla/xla/service/gpu/transforms/scaled_dot_rewriter.h
// This pass rewrites ScaledDot instructions into a sequence of other HLO
// instructions, including Convert, Broadcast, Reshape, Multiply, and Dot.
class ScaledDotRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "scaled-dot-rewriter"; }
=======
// An HLO pass that hoists slice operations through add operations.
class SliceHoister : public HloModulePass {
 public:
  SliceHoister() = default;
>>>>>>> upstream/master:third_party/xla/xla/hlo/transforms/simplifiers/slice_hoister.h

  absl::string_view name() const override { return "slice-hoister"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

<<<<<<< HEAD:third_party/xla/xla/service/gpu/transforms/scaled_dot_rewriter.h
#endif  // XLA_SERVICE_GPU_TRANSFORMS_SCALED_DOT_REWRITER_H_
=======
#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_SLICE_HOISTER_H_
>>>>>>> upstream/master:third_party/xla/xla/hlo/transforms/simplifiers/slice_hoister.h
