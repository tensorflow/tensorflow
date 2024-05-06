/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRIANGULAR_SOLVE_REWRITER_H_
#define XLA_SERVICE_GPU_TRIANGULAR_SOLVE_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites HLO TriangularSolve ops into a custom-call.
//
// The motivation for this is that we need to add temp memory to batched
// triangular-solve ops in order to call cublas trsmBatched.  We rewrite batch 1
// ops as well so that we have fewer codepaths to worry about in the backend.
//
// cublas trsmBatched takes arrays in GPU memory of pointers to the inputs and
// outputs, `a` and `b`.  In XLA the inputs/outputs are always contiguous, but
// we still have to materialize out these arrays.
//
// We use the same trick as for cudnn convolutions: This custom-call returns a
// tuple (actual-result, temp-memory).  In this our case the temp buffer always
// has size 2 * sizeof(void*) * batch_size, because we need two arrays of
// pointers.
//
// The custom-call has a backend-config equal to the TriangularSolveOptions
// object.
class TriangularSolveRewriter : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "triangular-solve-rewriter";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRIANGULAR_SOLVE_REWRITER_H_
