/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVES_SCHEDULE_LINEARIZER_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVES_SCHEDULE_LINEARIZER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/util.h"

namespace xla {

// Enforces a total order on all collectives present in the module, based on the
// order given to the instructions.
//
// Does not insert inter-computation dependencies, only linearizes the order
// within each computation.
class CollectivesScheduleLinearizer : public HloModulePass {
 public:
  explicit CollectivesScheduleLinearizer(HloModulePredicate is_enabled = {})
      : is_enabled_(is_enabled) {}

  absl::string_view name() const override {
    return "collectives-schedule-linearizer";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  HloModulePredicate is_enabled_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVES_SCHEDULE_LINEARIZER_H_
