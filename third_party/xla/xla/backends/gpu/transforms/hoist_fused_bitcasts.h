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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_HOIST_FUSED_BITCASTS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_HOIST_FUSED_BITCASTS_H_

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Hoist bitcasts and reshapes in the computation out of "__triton_gemm" fusions
// with a dot instruction.
class HoistFusedBitcasts : public HloModulePass {
 public:
  absl::string_view name() const override { return "hoist-fused-bitcasts"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> RunOnModule(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads);
};

namespace detail {

// Returns the start indices of consecutive non-overlapping subsequences of `a`
// and `b` with the same product (see `CommonFactors` from `util.h`) grouping
// ranges having product of 1 with neighbors.
//
// For example, if a=[2, 5, 1, 3] and b=[1, 10, 3, 1], the result will be
// {{0, 0}, {2, 2}, {4, 4}}, grouping [2,5] with [1,10] and [1,3] with [3,1].
absl::InlinedVector<std::pair<int64_t, int64_t>, 8>
CommonFactorsMergingTrivialRanges(absl::Span<const int64_t> a,
                                  absl::Span<const int64_t> b);

}  // namespace detail

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_HOIST_FUSED_BITCASTS_H_
