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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_ONEHOT_REWRITER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_ONEHOT_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites a One-Hot encoded Dot operation into a Gather operation.
//
// Matches:
//   Dot(OneHot(Indices), Weights)
//
// Converts to:
//   Gather(Weights, Indices)
//
// The rewrite preserves the semantics of the original One-Hot Dot even for
// out-of-bounds indices.
//    - If an index is out of bounds, the One-Hot vector is all zeros, and the
//      Dot result is zero.
//    - A raw Gather with out-of-bounds indices is implementation-defined
//    (clamping on GPU), which does not match the original semantics.
//    - This pass emits a safe sequence:
//        a. Clamp indices to [0, depth-1].
//        b. Perform Gather with clamped indices.
//        c. Select 0 where the original index was out of bounds.
class OneHotGatherRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "one-hot-rewriter"; }

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_ONEHOT_REWRITER_H_
