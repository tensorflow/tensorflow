/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/casts_elimination.h"
#include "tensorflow/compiler/plugin/poplar/driver/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/matcher_predicates.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

#include <set>

namespace xla {
namespace poplarplugin {

/*
 * Note about constructing these patterns.  Due to the behaviour of the fuser
 * there must be no backward references.  All nodes should appear after any
 * other nodes that refer to them.
 *
 */

// clang-format off
static const std::vector<HloMatcherPattern> patterns = {
  // Remove convert to/from F32 before/after reduction, where initial value is
  // a constant
  HloMatcherPattern(
    PatternType("reduction_no_convert"),
    PatternMetaTarget(1),
    PatternInputs({4}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kConvert, NodeOperands({1}), IsF32ToF16Convert},
      {HloOpcode::kReduce, NodeOperands({2, 3}), IsF32},
      {HloOpcode::kConvert, NodeOperands({4}), IsF16ToF32Convert},
      {HloOpcode::kConstant, NodeOperands({}), IsF32},
      {HloOpcode::kParameter, NodeOperands({}), IsF16}
    })
  ),

  // Remove convert to/from F32 before/after reduction, where initial value is
  // a convert from F16
  HloMatcherPattern(
    PatternType("reduction_no_convert"),
    PatternMetaTarget(1),
    PatternInputs({4, 5}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kConvert, NodeOperands({1}), IsF32ToF16Convert},
      {HloOpcode::kReduce, NodeOperands({2, 3}), IsF32},
      {HloOpcode::kConvert, NodeOperands({4}), IsF16ToF32Convert},
      {HloOpcode::kConvert, NodeOperands({5}), IsF16ToF32Convert},
      {HloOpcode::kParameter, NodeOperands({}), IsF16},
      {HloOpcode::kParameter, NodeOperands({}), IsF16}
    })
  ),

  // Remove convert to/from F32 before/after average pool
  HloMatcherPattern(
      PatternType("reduction_no_convert_with_divide"),
      PatternMetaTarget(4),
      PatternInputs({7}),
      PatternOutputs({0}),
      Pattern({
          {HloOpcode::kConvert, NodeOperands({1}), IsF32ToF16Convert},
          {HloOpcode::kDivide, NodeOperands({4, 2}), IsF32},
          {HloOpcode::kBroadcast, NodeOperands({3}), IsF32},
          {HloOpcode::kConstant, NodeOperands({}), IsF32},
          {HloOpcode::kReduce, NodeOperands({5, 6}), IsF32},
          {HloOpcode::kConvert, NodeOperands({7}), IsF16ToF32Convert},
          {HloOpcode::kConstant, NodeOperands({}), IsF32},
          {HloOpcode::kParameter, NodeOperands({}), IsF16}
      })
  ),

  // Remove convert to/from F32 before/after reduction window, where initial
  // value is a constant
  HloMatcherPattern(
    PatternType("reducewindow_no_convert"),
    PatternMetaTarget(1),
    PatternInputs({4}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kConvert, NodeOperands({1}), IsF32ToF16Convert},
      {HloOpcode::kReduceWindow, NodeOperands({2, 3}), IsF32},
      {HloOpcode::kConvert, NodeOperands({4}), IsF16ToF32Convert},
      {HloOpcode::kConstant, NodeOperands({}), IsF32},
      {HloOpcode::kParameter, NodeOperands({}), IsF16}
    })
  ),

  // Convert and then convert back F16 -> F32 -> F16
  HloMatcherPattern(
    PatternType("convert_no_use"),
    PatternMetaTarget(0),
    PatternInputs({2}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kConvert, NodeOperands({1}), IsF32ToF16Convert},
      {HloOpcode::kConvert, NodeOperands({2}), IsF16ToF32Convert},
      {HloOpcode::kParameter, NodeOperands({}), IsF16}
    })
  ),

  // Convert and then convert back F32 -> F16 -> F32
  HloMatcherPattern(
    PatternType("convert_no_use"),
    PatternMetaTarget(0),
    PatternInputs({2}),
    PatternOutputs({0}),
    Pattern({
      {HloOpcode::kConvert, NodeOperands({1}), IsF16ToF32Convert},
      {HloOpcode::kConvert, NodeOperands({2}), IsF32ToF16Convert},
      {HloOpcode::kParameter, NodeOperands({}), IsF32}
    })
  ),
};
// clang-format on

CastsElimination::CastsElimination(struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false) {}

namespace {

HloInstruction* ConvertConstant(HloInstruction* constant,
                                const PrimitiveType& new_type) {
  const auto shape = ShapeUtil::ChangeElementType(constant->shape(), new_type);
  auto literal_new_type = constant->literal().ConvertToShape(shape);

  auto* new_inst = constant->parent()->AddInstruction(
      HloInstruction::CreateConstant(std::move(literal_new_type.ValueOrDie())));

  new_inst->set_raw_backend_config_string(
      constant->raw_backend_config_string());

  new_inst->set_metadata(constant->metadata());
  if (constant->has_sharding()) {
    new_inst->set_sharding(constant->sharding());
  }
  return new_inst;
}

}  // namespace

unsigned CastsElimination::ReplaceNodes() {
  unsigned int replacement_count = 0;
  for (int pattern_idx = 0; pattern_idx < matches_.size(); pattern_idx++) {
    for (HloMatcherMatched& match : matches_[pattern_idx]) {
      if (match.ok) {
        HloInstruction* pattern_root = match.instructions[0];
        HloComputation* computation = pattern_root->parent();
        auto type = pattern_root->shape().element_type();
        OutlinedInfo outlined_info = {pattern_root, {}};
        absl::flat_hash_set<HloInstruction*> matched_instructions(
            match.instructions.begin(), match.instructions.end());

        std::vector<HloInstruction*> new_instructions;

        for (HloInstruction* inst : match.instructions) {
          outlined_info.removed_or_modified_instructions.push_back(inst);

          HloInstruction* new_inst;
          if (inst->opcode() == HloOpcode::kConstant) {
            // For constants - replace it with the new constant.
            new_inst = ConvertConstant(inst, type);
          } else {
            // Otherwise clone and change the desired shape.
            new_inst = computation->AddInstruction(inst->Clone());
            new_inst->mutable_shape()->set_element_type(type);
          }
          // Replace all all the users of inst with new_inst in this pattern.
          for (auto user : inst->users()) {
            // Skip the user if it's not in the pattern.
            if (matched_instructions.count(user) == 0) {
              continue;
            }
            // Replace all the operands where the instruction is used with the
            // new instruction.
            for (int64 operand_num : user->OperandIndices(inst)) {
              TF_CHECK_OK(user->ReplaceOperandWith(operand_num, new_inst));
            }
          }
          // Update the set
          matched_instructions.erase(inst);
          matched_instructions.insert(new_inst);
          // Keep track of new instructions.
          new_instructions.push_back(new_inst);
        }
        TF_CHECK_OK(pattern_root->ReplaceAllUsesWith(new_instructions[0]));
        TF_CHECK_OK(
            computation->RemoveInstructionAndUnusedOperands(pattern_root));
        replacement_count += MarkReplacedInstructions(outlined_info);
      }
    }
  }

  return replacement_count;
}

}  // namespace poplarplugin
}  // namespace xla
