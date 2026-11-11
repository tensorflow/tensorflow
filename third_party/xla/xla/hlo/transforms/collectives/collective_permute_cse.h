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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_PERMUTE_CSE_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_PERMUTE_CSE_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Removes redundant collective permutes
//
// An optimization that removes duplicate CollectivePermute instructions based
// on identical source-target pairs and operands.
//
// For example, if we have two CollectivePermutes with the same source-target
// pairs:
//
//   %cp0 = collective-permute(%input)
//   %cp1 = collective-permute(%input)
//
// This pass will replace %cp1 with %cp0.
//
// It also CSEs CollectivePermute through slice instructions. For example:
//
//   %slice = slice(%input)
//   %cp0 = collective-permute(%input)
//   %cp1 = collective-permute(%slice)
//
// This pass will replace %cp1 with slice(%cp0).
//
// Operands that read windows of one array through chains of unit-stride
// slices are compared by their windows on that array: a window inside another
// reuses the larger permute as above, and two windows that differ in a single
// dimension with touching or overlapping intervals are sent by one permute of
// their union, which moves no element that neither of them moved:
//
//   %s0 = slice(%input), slice={[0:3]}
//   %s1 = slice(%input), slice={[1:4]}
//   %cp0 = collective-permute(%s0)
//   %cp1 = collective-permute(%s1)
//
// becomes one collective-permute of %input, with %cp0 and %cp1 slices of it.
//
// When xla_enable_enzyme_comms_opt is enabled, two collective-permutes with the
// same source-target pairs that read identical slice windows on paired (hi, lo)
// operands (from 64-bit decomposition such as X64SplitLow/X64SplitHigh, TwoSum
// outputs, or adjacent tuple elements) are fused into one collective-permute of
// their concatenation along dimension 0, with each original collective-permute
// replaced by a slice of the fused result.
//
class CollectivePermuteCSE : public HloModulePass {
 public:
  absl::string_view name() const override { return "collective-permute-cse"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_PERMUTE_CSE_H_
