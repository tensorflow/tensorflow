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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_IDENTITY_COMPUTATION_REMOVER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_IDENTITY_COMPUTATION_REMOVER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

class HloIdentityComputationRemover : public HloModulePass {
 public:
  explicit HloIdentityComputationRemover(bool run_cleanup = false)
      : run_cleanup_(run_cleanup) {}
  ~HloIdentityComputationRemover() override = default;
  absl::string_view name() const override {
    return "identity_computation_remover";
  }

  static bool IsIdentityComputation(HloComputation* computation);

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  bool run_cleanup_;
  absl::Status CleanUp(HloModule* module);
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_IDENTITY_COMPUTATION_REMOVER_H_
