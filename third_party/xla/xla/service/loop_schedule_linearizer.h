/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_LOOP_SCHEDULE_LINEARIZER_H_
#define XLA_SERVICE_LOOP_SCHEDULE_LINEARIZER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Adds control dependency edges from instructions which "write" values inside
// the loop, to instructions which "read" those same values, in order to avoid
// extraneous copies. This is not always possible with our buffer layout
// constraints (that is, assuming that every element of the tuple the while loop
// operates upon gets the same buffer) as it may create cycles (an easiest
// example of a dependency cycle is a loop doing `(a, b) = (b, a)`). Thus we
// take a best-effort approach instead: add dependency edges only if we can show
// they don't create a cycle.
class LoopScheduleLinearizer : public HloModulePass {
 public:
  absl::string_view name() const override { return "loop-schedule-linearizer"; }

  explicit LoopScheduleLinearizer(const AliasInfo* alias_info)
      : alias_info_(alias_info) {}

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Backend specific information about whether an instruction can share buffer
  // with its operand.
  const AliasInfo* alias_info_;
};

}  // namespace xla

#endif  // XLA_SERVICE_LOOP_SCHEDULE_LINEARIZER_H_
