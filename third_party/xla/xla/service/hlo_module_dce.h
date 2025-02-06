/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_MODULE_DCE_H_
#define XLA_SERVICE_HLO_MODULE_DCE_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// HLO pass which removes dead code from computations in the module using
// HloModule-scoped analysis (HloLivenessAnalysis).
//
// Sweeps through live instructions which cross computation boundaries (kWhile),
// and removes code at dead shape indices.
//
class HloModuleDCE : public HloModulePass {
 public:
  ~HloModuleDCE() override {}
  absl::string_view name() const override { return "hlo-module-dce"; }

  // Run the pass on the given module. Returns whether the module was changed
  // (instructions were removed).
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_MODULE_DCE_H_
