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

#ifndef XLA_SERVICE_REDUCE_WINDOW_REWRITER_H_
#define XLA_SERVICE_REDUCE_WINDOW_REWRITER_H_

#include <cstdint>
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/status.h"
#include "xla/statusor.h"

namespace xla {

// Rewrite ReduceWindow to be more performant in cases it is written in a
// quadratic way:
//
// 1) Work around unimplemented cases in the implementation of ReduceWindow.
//
// This rewrites all R1 ReduceWindow nodes. We reshape the operand to an
// R2, perform the operation, and reshape back to R1. The reshapes correspond to
// a bitcast if the tensor length is less than or equal to a passed parameter.
// The motivation for this is to avoid use of overly large reductions and the
// complexities and restrictions therein.
//
// 2) Rewrite ReduceWindow ops that represent a CumSum/CumProd into a
// tree-reduction (see details in the implementation).
// Note that this may itself generate R1 ReduceWindow ops, which means this pass
// needs to be run to a fixed point.
class ReduceWindowRewriter : public HloModulePass {
 public:
  // `base_length` is a size of a reduce-window we are comfortable with
  // executing.
  explicit ReduceWindowRewriter(int64_t base_length)
      : base_length_(base_length) {}

  absl::string_view name() const override { return "reduce-window-rewriter"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::Status ReplaceReduceWindowWithReshape(
      HloReduceWindowInstruction* reduce_window);

  absl::StatusOr<bool> TryOptimizeCumSumOrProd(
      HloReduceWindowInstruction* reduce_window);

  int64_t base_length_;
};

}  // namespace xla

#endif  // XLA_SERVICE_REDUCE_WINDOW_REWRITER_H_
