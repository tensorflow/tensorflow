/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/all_reduce_promotion.h"

#include <memory>
#include <string>
#include <utility>

namespace xla {

namespace {
bool IsAllReduce(const HloInstruction* inst) {
  return inst->opcode() == HloOpcode::kAllReduce ||
         inst->opcode() == HloOpcode::kReduceScatter;
}

std::unique_ptr<HloInstruction> CloneAllReduce(
    const HloInstruction* inst, const Shape& shape,
    absl::Span<HloInstruction* const> operands) {
  // clone an all-reduce or reduce-scatter and also clone the attached
  // computation to match the type.
  std::unique_ptr<HloInstruction> new_inst =
      inst->CloneWithNewOperands(shape, operands);
  HloComputation* to_apply = new_inst->to_apply();
  HloComputation* to_apply_promoted = [&]() {
    PrimitiveType type = shape.element_type();
    std::string name = to_apply->name() + "_promoted";
    HloComputation::Builder promoted(name);
    auto x = promoted.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/0, ShapeUtil::MakeShape(type, {}), "x"));
    auto y = promoted.AddInstruction(HloInstruction::CreateParameter(
        /*parameter_number=*/1, ShapeUtil::MakeShape(type, {}), "y"));
    promoted.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(type, {}), to_apply->root_instruction()->opcode(),
        x, y));
    return inst->GetModule()->AddEmbeddedComputation(promoted.Build());
  }();
  new_inst->set_to_apply(to_apply_promoted);
  return new_inst;
}

}  // namespace

// Promote 16-bit integer all-reduce and reduce-scatter to 32-bit integer types.
// {{U16, U32}, {S16, S32}}
AllReducePromotion::AllReducePromotion(
    absl::Span<std::pair<PrimitiveType, PrimitiveType> const> from_to_types)
    : pass_(from_to_types, IsAllReduce, CloneAllReduce) {}

StatusOr<bool> AllReducePromotion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return pass_.Run(module, execution_threads);
}

}  // namespace xla
