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

#include "xla/service/collective_quantizer.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

namespace m = match;

// Matches a broadcast of a scalar operand.
template <typename... Args>
auto ScalarBroadcast(Args... args) {
  return m::Broadcast(args...).WithPredicate([](const HloInstruction* instr) {
    return ShapeUtil::IsScalar(instr->operand(0)->shape());
  });
}

// Matches a bitcast that preserves the element type of the operand.
auto BitcastPreservesElementType() {
  return m::Bitcast().WithPredicate([](const HloInstruction* instr) {
    return ShapeUtil::SameElementType(instr->shape(),
                                      instr->operand(0)->shape());
  });
}

// Matches a type conversion to a type with a smaller byte size than that of the
// operand.
auto ConvertToNarrowerType() {
  auto converts_to_narrower_type = [](const HloInstruction* instr) -> bool {
    return ShapeUtil::ByteSizeOfPrimitiveType(instr->shape().element_type()) <
           ShapeUtil::ByteSizeOfPrimitiveType(
               instr->operand(0)->shape().element_type());
  };
  return m::Convert().WithPredicate(converts_to_narrower_type);
}

// Returns true iff instr describes a quantization, i.e. a multiplication or
// division by a broadcasted scalar followed by a clamp and a type conversion,
// or a plain type conversion to a narrower type. Unary bitcast, copy, reshape
// or slice ops with one user may precede the quantization or type conversion.
bool IsSupportedQuantization(HloInstruction* instr, HloInstruction** convert,
                             HloInstruction** binary, HloInstruction** clamp,
                             HloInstruction** scale_bcast,
                             std::vector<HloInstruction*>& unary_ops) {
  std::vector<HloInstruction*> ops;
  while (instr->user_count() <= 1) {
    if (Match(instr, m::AnyOf<HloInstruction>(
                         BitcastPreservesElementType(), m::Copy(), m::Reshape(),
                         m::Slice(), m::Multiply(), m::Divide(), m::Clamp()))) {
      if (instr->user_count() > 0) {
        ops.emplace_back(instr);
        instr = instr->users()[0];
        continue;
      }
      break;
    }

    if (Match(instr, ConvertToNarrowerType())) {
      ops.emplace_back(instr);
      break;
    }
    VLOG(5) << "Unsupported instruction.";
    return false;
  }

  // In the quantization case, the type conversion is preceded by a
  // multiplication or division by a broadcasted scalar and a clamp instruction.
  if (ops.size() > 2 &&
      (Match(ops.back(),
             m::Convert(convert, m::Clamp(clamp, ScalarBroadcast(),
                                          m::MultiplyAnyOrder(
                                              binary, m::Op(),
                                              ScalarBroadcast(scale_bcast)),
                                          ScalarBroadcast()))) ||
       Match(
           ops.back(),
           m::Convert(convert, m::Clamp(clamp, ScalarBroadcast(),
                                        m::Divide(binary, m::Op(),
                                                  ScalarBroadcast(scale_bcast)),
                                        ScalarBroadcast()))))) {
    unary_ops = {ops.begin(), ops.end() - 3};
  } else if (ops.size() > 0 && Match(ops.back(), m::Convert(convert))) {
    unary_ops = {ops.begin(), ops.end() - 1};
  } else {
    VLOG(5) << "Did not find type conversion or quantization pattern.";
    return false;
  }

  // The collected unary ops between collective and quantization/type conversion
  // may only include bitcast, copy, reshape and slice instructions.
  for (HloInstruction* unary_op : unary_ops) {
    if (!Match(unary_op, m::AnyOf<HloInstruction>(m::Bitcast(), m::Copy(),
                                                  m::Reshape(), m::Slice()))) {
      VLOG(5) << "Unexpected instruction in unary ops.";
      return false;
    }
  }
  return true;
}

bool IsSupportedCollective(HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kAllGather ||
         instr->opcode() == HloOpcode::kAllToAll ||
         instr->opcode() == HloOpcode::kCollectiveBroadcast ||
         instr->opcode() == HloOpcode::kCollectivePermute;
}

}  // namespace

absl::StatusOr<bool> CollectiveQuantizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* comp : module->MakeComputationPostOrder()) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      HloInstruction *binary = nullptr, *clamp, *convert, *scale_bcast;
      std::vector<HloInstruction*> unary_ops;
      if (instr->user_count() == 1 && IsSupportedCollective(instr) &&
          IsSupportedQuantization(instr->users()[0], &convert, &binary, &clamp,
                                  &scale_bcast, unary_ops)) {
        HloInstruction* coll_operand = instr->mutable_operand(0);
        HloInstruction *new_binary, *new_clamp;
        // When there is a quantization, insert the scale and clamp ops.
        if (binary) {
          HloInstruction* new_scale_bcast = comp->AddInstruction(
              scale_bcast->CloneWithNewShape(coll_operand->shape()));
          new_binary = comp->AddInstruction(binary->CloneWithNewOperands(
              coll_operand->shape(), {coll_operand, new_scale_bcast}));
          HloInstruction* new_clamp_lower = comp->AddInstruction(
              clamp->operand(0)->CloneWithNewShape(coll_operand->shape()));
          HloInstruction* new_clamp_upper = comp->AddInstruction(
              clamp->operand(2)->CloneWithNewShape(coll_operand->shape()));
          new_clamp = comp->AddInstruction(clamp->CloneWithNewOperands(
              coll_operand->shape(),
              {new_clamp_lower, new_binary, new_clamp_upper}));
        }

        // Move the collective past the conversion to the narrow type.
        Shape new_convert_shape = ShapeUtil::ChangeElementType(
            instr->operand(0)->shape(), convert->shape().element_type());
        HloInstruction* new_convert =
            comp->AddInstruction(convert->CloneWithNewOperands(
                new_convert_shape, {binary ? new_clamp : coll_operand}));
        Shape new_collective_shape = ShapeUtil::ChangeElementType(
            instr->shape(), convert->shape().element_type());
        HloInstruction* new_collective = comp->AddInstruction(
            instr->CloneWithNewOperands(new_collective_shape, {new_convert}));

        // Sequentially apply the collected unary ops to the output of the
        // quantized collective.
        auto shift_unary_ops = [comp, &unary_ops](HloInstruction** x) -> void {
          for (HloInstruction* unary_op : unary_ops) {
            *x = comp->AddInstruction(unary_op->CloneWithNewOperands(
                ShapeUtil::MakeShapeWithDenseLayout(
                    (*x)->shape().element_type(),
                    unary_op->shape().dimensions(),
                    unary_op->shape().layout().minor_to_major()),
                {*x}));
          }
        };

        shift_unary_ops(&new_collective);
        TF_RETURN_IF_ERROR(convert->ReplaceAllUsesWith(new_collective));

        changed = true;
        VLOG(5) << "Quantized collective " << new_collective->ToShortString();
      }
    }
  }

  return changed;
}

}  // namespace xla
