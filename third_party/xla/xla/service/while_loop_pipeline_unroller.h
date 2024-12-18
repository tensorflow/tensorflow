/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_WHILE_LOOP_PIPELINE_UNROLLER_H_
#define XLA_SERVICE_WHILE_LOOP_PIPELINE_UNROLLER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
// Pipelined loops have inherent aliasing interference in them, due to loop
// inputs shifting positions across iterations. This results in copy insertion
// adding copies for each pipelined input. In some cases extra copies on top of
// this are needed to properly sequence all the mandatory aliasing copies.
//
// It is not necessary to insert copies to resolve interference in this case.
// The loop inputs, despite directly carried out as loop outputs, still have
// finite lifetimes across a certain amount of loop iterations. If the loop was
// unrolled just enough times to have the lifetimes of its inputs end before the
// outputs would be materialized, this would implicitly remove any sort of
// interference. The drawback of this approach is that it can in some cases
// drastically increase compile times due to linearly increasing graph size.
class WhileLoopPipelineUnroller : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "while_loop_pipeline_unroller";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // The pipeline depth of a while loop is the number of loop iterations that
  // pipelined loop inputs live throughout. This is used to determine how many
  // times to unroll the loop in order to remove aliasing interference.
  static int64_t ComputeWhileLoopPipelineDepth(
      const HloInstruction& while_instruction);
};
}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_PIPELINE_UNROLLER_H_
