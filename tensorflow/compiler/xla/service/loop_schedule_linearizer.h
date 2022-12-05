/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LOOP_SCHEDULE_LINEARIZER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LOOP_SCHEDULE_LINEARIZER_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_alias_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

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

  explicit LoopScheduleLinearizer(
      const HloDataflowAnalysis::CanShareBuffer& can_share_buffer = nullptr)
      : can_share_buffer_(can_share_buffer) {}

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Backend specific function that decides whether an instruction can share
  // buffer with its operand.
  HloDataflowAnalysis::CanShareBuffer can_share_buffer_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LOOP_SCHEDULE_LINEARIZER_H_
