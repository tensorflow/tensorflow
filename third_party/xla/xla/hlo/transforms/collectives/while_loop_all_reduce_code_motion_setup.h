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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_WHILE_LOOP_ALL_REDUCE_CODE_MOTION_SETUP_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_WHILE_LOOP_ALL_REDUCE_CODE_MOTION_SETUP_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/transforms/expanders/op_expander_pass.h"

namespace xla {

// Reorder the sequence of reduce-scatter, convert, transpose, and add
// operations. This transformation changes the pattern from:
//   add(transpose(convert(reduce-scatter(operand))), get-tuple(parameter(0))
//   add(transpose(reduce-scatter(operand)), get-tuple(parameter(0))
// to:
//   add(reduce-scatter(transpose(convert(operand))), get-tuple(parameter(0))
//   add(reduce-scatter(transpose(operand)), get-tuple(parameter(0))
class ReorderReduceTranspose : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "reorder-reduce-transpose"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;
  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

// Reorder the reduce-scatter/all-reduce and convert operations followed
// by an add. This transformation changes the pattern from:
//   add(convert(reduce-scatter(operand)), get-tuple(parameter(0)))
//   add(convert(all-reduce(operand)), get-tuple(parameter(0)))
// to:
//   add(reduce-scatter(convert(operand)), get-tuple(parameter(0)))
//   add(all-reduce(convert(operand)), get-tuple(parameter(0)))
class ReorderConvertReduceAdd : public OpExpanderPass {
 public:
  absl::string_view name() const override {
    return "reorder-convert-reduce-add";
  }

  // Constructor with optional enable_reduce_scatter parameter
  explicit ReorderConvertReduceAdd(bool enable_reduce_scatter = false)
      : enable_reduce_scatter_(enable_reduce_scatter) {}

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;
  absl::StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
  // Enable transformation of reduce-scatter op.
  bool enable_reduce_scatter_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_WHILE_LOOP_ALL_REDUCE_CODE_MOTION_SETUP_H_
