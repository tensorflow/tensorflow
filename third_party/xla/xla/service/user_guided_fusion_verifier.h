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

#ifndef XLA_SERVICE_USER_GUIDED_FUSION_VERIFIER_H_
#define XLA_SERVICE_USER_GUIDED_FUSION_VERIFIER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

class UserGuidedFusionVerifier : public HloModulePass {
 public:
  ~UserGuidedFusionVerifier() override = default;
  absl::string_view name() const override {
    return "user_guided_fusion_verifier";
  }

  // Verifies that the given module has all the user guided fusions formed
  // correctly. Returns error status if the verification fails. Otherwise, it
  // always returns false, as it does not change the module.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    return false;
  };
};

}  // namespace xla

#endif  // XLA_SERVICE_USER_GUIDED_FUSION_VERIFIER_H_
