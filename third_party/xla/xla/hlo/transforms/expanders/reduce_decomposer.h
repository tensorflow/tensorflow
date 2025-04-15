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

#ifndef XLA_HLO_TRANSFORMS_EXPANDERS_REDUCE_DECOMPOSER_H_
#define XLA_HLO_TRANSFORMS_EXPANDERS_REDUCE_DECOMPOSER_H_

#include <functional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// For each reduction R(I), ensures the postcondition:
//
//    !custom_layout_allowed(R)
//      =>
//    layout(R) == layout(I) # modulo removed dimensions
//
// To achieve that, decomposes layout-mutating reductions which do not satisfy
// `custom_layout_allowed` into a reduction and a copy.
//
// For a singular reduction:
//
//   -> reduce ->
//
// Gets turned into:
//
//    -> reduce -> copy ->
//
// For a variadic recuction, the layout assignment guarantees that the layout
// is the same for all outputs. This pass will transpose the variadic reduction
// inputs which have different physical layout to the first operand.
//
//   A{L} \
//   B{L} -> reduce{L'} ->
//   C{L} /
//
// Get turned into:
//
//   A{L} \                 / GTE(1) -> copy{L'} \
//   B{L} -> reduce{E(L)} --- GTE(2) -> copy{L'} - Tuple{L'}
//   C{L} /                 \ GTE(3) -> copy{L'} /
//
//   Where E(L) is expected layout of a reduction (original layout with reduce
//   dimensions dropped).
//
// PRECONDITION:
//  In variadic reduction, all outputs have the same layout
//  (enforced by layout assignment).
class ReduceDecomposer : public HloModulePass {
 public:
  explicit ReduceDecomposer(HloPredicate custom_layout_allowed = nullptr)
      : custom_layout_allowed_(custom_layout_allowed) {}

  absl::string_view name() const override { return "reduce-decomposer"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  HloPredicate custom_layout_allowed_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_EXPANDERS_REDUCE_DECOMPOSER_H_
