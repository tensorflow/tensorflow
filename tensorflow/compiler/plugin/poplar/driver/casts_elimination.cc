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

unsigned CastsElimination::ReplaceReduction(const HloMatcherMatched& match,
                                            const HloOpcode reduction_type) {
  auto* convert_out = match.instructions[0];
  auto* reduction = convert_out->mutable_operand(0);
  auto* to_reduce_convert = reduction->mutable_operand(0);
  auto* init_val = reduction->mutable_operand(1);
  auto* value_in = to_reduce_convert->mutable_operand(0);

  // Create a new reduce_computation
  // Check the reduction op is elementwise binary and takes in two
  // parameters only. If that's not the case then we can't convert this
  // reduction.
  auto* reduce_computation = reduction->to_apply();
  auto* reduce_op = reduce_computation->root_instruction();
  if (!(reduce_op->IsElementwiseBinary() &&
        reduce_op->operand(0)->opcode() == HloOpcode::kParameter &&
        reduce_op->operand(1)->opcode() == HloOpcode::kParameter)) {
    return 0;
  }
  // Build the new reduce_computation
  auto builder = HloComputation::Builder(reduce_computation->name());
  {
    auto* in0 = builder.AddInstruction(reduce_op->operand(0)->Clone());
    in0->mutable_shape()->set_element_type(F16);
    auto* in1 = builder.AddInstruction(reduce_op->operand(1)->Clone());
    in1->mutable_shape()->set_element_type(F16);
    const auto shape_op_fp16 =
        ShapeUtil::ChangeElementType(reduce_op->shape(), F16);
    builder.AddInstruction(
        reduce_op->CloneWithNewOperands(shape_op_fp16, {in0, in1}));
  }
  auto* new_reduce_computation =
      match.computation->parent()->AddEmbeddedComputation(builder.Build());

  // Get the initial value
  HloInstruction* new_init_val;
  if (init_val->opcode() == HloOpcode::kConstant) {
    // convert a constant from F32 to F16 and add it to the graph
    const auto shape_init_val_fp16 =
        ShapeUtil::ChangeElementType(init_val->shape(), F16);
    auto literal_f16 = init_val->literal().ConvertToShape(shape_init_val_fp16);
    // If we can't convert shape then skip this one
    if (!literal_f16.ok()) {
      return 0;
    }
    new_init_val = match.computation->AddInstruction(
        HloInstruction::CreateConstant(std::move(literal_f16.ValueOrDie())));
  } else if (init_val->opcode() == HloOpcode::kConvert) {
    // init value is an output of a Convert from FP16 to FP32, so use the
    // argument to convert
    new_init_val = init_val->mutable_operand(0);
  } else {
    LOG(FATAL) << "Unsupported Op for Reduction init value";
  }

  // Create the new reduction
  const auto shape_reduction_fp16 =
      ShapeUtil::ChangeElementType(reduction->shape(), F16);
  // Create the new reduction dependent on the type of reduction
  HloInstruction* new_reduction;
  switch (reduction_type) {
    case HloOpcode::kReduce: {
      new_reduction =
          match.computation->AddInstruction(HloInstruction::CreateReduce(
              shape_reduction_fp16, value_in, new_init_val,
              reduction->dimensions(), new_reduce_computation));
      break;
    }
    case HloOpcode::kReduceWindow: {
      new_reduction =
          match.computation->AddInstruction(HloInstruction::CreateReduceWindow(
              shape_reduction_fp16, value_in, new_init_val, reduction->window(),
              new_reduce_computation));
      break;
    }
    default: { LOG(FATAL) << "Unsupported Op for Reduction init value"; }
  }
  new_reduction->set_metadata(reduction->metadata());

  // Replace all uses with the new reduction
  OutlinedInfo outlined_info;
  outlined_info.removed_or_modified_instructions.push_back(convert_out);
  TF_CHECK_OK(convert_out->ReplaceAllUsesWith(new_reduction));
  return MarkReplacedInstructions(outlined_info);
}

unsigned CastsElimination::ReplaceNodes() {
  unsigned int replacement_count = 0;

  // Handle all the reductions with a casts around them - remove all the casts
  const std::vector<unsigned> casts_around_reduction_patterns = {0, 1};
  for (const auto pattern_index : casts_around_reduction_patterns) {
    for (HloMatcherMatched& match : matches_[pattern_index]) {
      if (match.ok) {
        replacement_count += ReplaceReduction(match, HloOpcode::kReduce);
      }
    }
  }

  // Handle all the reductions with a casts around them - remove all the casts
  const std::vector<unsigned> casts_around_reduction_window_patterns = {2};
  for (const auto pattern_index : casts_around_reduction_window_patterns) {
    for (HloMatcherMatched& match : matches_[pattern_index]) {
      if (match.ok) {
        replacement_count += ReplaceReduction(match, HloOpcode::kReduceWindow);
      }
    }
  }

  // Handle all the unused casts
  const std::vector<unsigned> unused_casts_patterns = {3, 4};
  for (const auto pattern_index : unused_casts_patterns) {
    for (HloMatcherMatched& match : matches_[pattern_index]) {
      if (match.ok) {
        auto* convert_out = match.instructions[0];
        auto* convert_in = convert_out->mutable_operand(0);
        auto* val_in = convert_in->mutable_operand(0);

        // Replace all uses with val_in
        OutlinedInfo outlined_info;
        outlined_info.removed_or_modified_instructions.push_back(convert_out);
        TF_CHECK_OK(convert_out->ReplaceAllUsesWith(val_in));
        replacement_count += MarkReplacedInstructions(outlined_info);
      }
    }
  }

  return replacement_count;
}

}  // namespace poplarplugin
}  // namespace xla
