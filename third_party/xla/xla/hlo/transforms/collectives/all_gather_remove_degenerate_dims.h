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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_ALL_GATHER_REMOVE_DEGENERATE_DIMS_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_ALL_GATHER_REMOVE_DEGENERATE_DIMS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// A pass that removes degenerate dimensions from all-gathers.
//
// For example:
//
//   %all-gather = f32[1,64,8192]{2,1,0} all-gather(%in), (...), dimensions={1}
//
// becomes:
//
//   %reshape = f32[64,8192]{1,0} reshape(%in)
//   %all-gather.reshaped = f32[64,8192]{1,0} all-gather(%reshape), (...),
//       dimensions={0}
//   %all-gather = f32[1,64,8192]{2,1,0} reshape(%all-gather.reshaped)
//
// This helps layout assignment assign better layouts for the all-gather, since
// the GPU requires the all-gather dimension to be major. The reshapes will turn
// into bitcasts.
class AllGatherRemoveDegenerateDims : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "all-gather-remove-degenerate-dims";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_ALL_GATHER_REMOVE_DEGENERATE_DIMS_H_
