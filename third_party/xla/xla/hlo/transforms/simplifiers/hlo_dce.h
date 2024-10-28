/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_DCE_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_DCE_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// HLO pass which removes dead instructions from each computation in the module
// and removes dead computations from the module.
//
// An instruction is dead if it is not reachable from the root. A computation is
// dead if it is not the entry computation of the module and it is not reachable
// from the entry computation.
//
// This pass does not remove dead parameter instructions, as parameter
// instructions cannot be deleted.
class HloDCE : public HloModulePass {
 public:
  HloDCE() : remove_cross_partition_collective_ops_(false) {}
  explicit HloDCE(bool remove_cross_partition_collective_ops)
      : remove_cross_partition_collective_ops_(
            remove_cross_partition_collective_ops) {}
  ~HloDCE() override {}
  absl::string_view name() const override { return "dce"; }

  // Run DCE on a computation.
  static absl::StatusOr<bool> RunOnComputation(
      HloComputation* computation, bool remove_cross_partition_collective_ops);

  // Run the pass on the given module. Returns whether the module was changed
  // (instructions were removed).
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  bool remove_cross_partition_collective_ops_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_DCE_H_
