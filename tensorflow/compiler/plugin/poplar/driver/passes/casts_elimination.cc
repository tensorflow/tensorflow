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

#include "tensorflow/compiler/plugin/poplar/driver/passes/casts_elimination.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_matcher.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/matcher_predicates.h"

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
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16}
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
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16},
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16}
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
          {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16}
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
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16}
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
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF16}
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
      {HloMatcherOpcode::kAnyOpcode, NodeOperands({}), IsF32}
    })
  ),
};
// clang-format on

CastsElimination::CastsElimination(struct CompilerAnnotations& annotations)
    : HloMatcher(patterns, annotations, false, false) {}

namespace {

HloInstruction* ConvertInstruction(HloInstruction* inst,
                                   const PrimitiveType& new_type) {
  HloInstruction* new_inst;
  HloComputation* computation = inst->parent();
  if (inst->opcode() == HloOpcode::kConstant) {
    // For constants - replace it with the new constant.
    const auto shape = ShapeUtil::ChangeElementType(inst->shape(), new_type);
    auto literal_new_type = inst->literal().ConvertToShape(shape);
    new_inst = computation->AddInstruction(HloInstruction::CreateConstant(
        std::move(literal_new_type.ValueOrDie())));
  } else {
    // Otherwise clone and change the desired shape.
    new_inst = computation->AddInstruction(inst->Clone());
    new_inst->mutable_shape()->set_element_type(new_type);
  }

  new_inst->set_raw_backend_config_string(inst->raw_backend_config_string());

  new_inst->set_metadata(inst->metadata());
  if (inst->has_sharding()) {
    new_inst->set_sharding(inst->sharding());
  }
  return new_inst;
}

}  // namespace

// For casts elimination we ignore the sharding device because we replace
// instructions with the sharding they had before.
bool CastsElimination::HandleMatch(HloMatcherMatched& match,
                                   const absl::optional<int64>) {
  // A map from original instructions to their new counterparts
  absl::flat_hash_map<NodeId, HloInstruction*> outlined;
  // A set of nodes which we have already outlined.
  absl::flat_hash_set<NodeId> outlined_node_ids;
  // A set of nodes which we can outline because all the operands have been
  // outlined.
  absl::flat_hash_set<NodeId> to_outline;

  auto pattern = patterns_[match.pattern_idx];
  HloComputation* computation = match.computation;
  HloInstruction* old_pattern_root =
      match.instruction_mapping[pattern.GetOutputs()[0]];
  auto new_type = old_pattern_root->shape().element_type();

  // A node can be outlined if all the operands have been outlined and it has
  // not been outlined yet.
  const auto can_outline = [&](NodeId node_id) {
    for (auto operand_id : pattern.GetNodesToOperandsMetaGraph()[node_id]) {
      if (outlined_node_ids.count(operand_id) == 0) {
        return false;
      }
    }
    return outlined_node_ids.count(node_id) == 0;
  };

  // First mark all the parameter inputs as outlined - these are never modified.
  for (auto node_id : pattern.GetInputs()) {
    HloInstruction* old_pattern_input = match.instruction_mapping.at(node_id);
    HloInstruction* new_pattern_input = old_pattern_input;
    outlined[node_id] = new_pattern_input;
    outlined_node_ids.insert(node_id);
    // Check what we can outline.
    absl::c_copy_if(pattern.GetOperandsToNodesMetaGraph()[node_id],
                    std::inserter(to_outline, std::begin(to_outline)),
                    can_outline);
  }
  // Add all the nodes in the pattern which have no dependencies as well.
  for (auto pair : pattern.GetNodesToOperandsMetaGraph()) {
    NodeId node_id = pair.first;
    auto edges = pair.second;
    if (edges.empty() && !outlined_node_ids.contains(node_id)) {
      to_outline.insert(node_id);
    }
  }

  // Now outline all the remaining nodes
  while (!to_outline.empty()) {
    // Get an instruction which is ready to be outlined.
    NodeId node_id = *to_outline.begin();
    to_outline.erase(node_id);

    HloInstruction* old_inst = match.instruction_mapping.at(node_id);
    // Convert it.
    HloInstruction* new_inst = ConvertInstruction(old_inst, new_type);
    outlined[node_id] = new_inst;
    outlined_node_ids.insert(node_id);
    // Replace all the operands.
    for (int64 operand = 0; operand < new_inst->operand_count(); ++operand) {
      NodeId operand_id =
          pattern.GetPatternNodes()[node_id].GetOperands()[operand];
      TF_CHECK_OK(new_inst->ReplaceOperandWith(operand, outlined[operand_id]));
    }
    // Check if we can outline more instructions.
    absl::c_copy_if(pattern.GetOperandsToNodesMetaGraph()[node_id],
                    std::inserter(to_outline, std::begin(to_outline)),
                    can_outline);
  }

  // Sanity check - make sure we have outlined everything.
  if (outlined.size() != pattern.GetPatternNodes().size()) {
    LOG(FATAL) << "Failed to outline a pattern correctly - not all "
                  "instructions have been outlined. "
               << outlined.size() << " " << pattern.GetPatternNodes().size();
  }

  HloInstruction* new_pattern_root = outlined[pattern.GetOutputs()[0]];
  TF_CHECK_OK(old_pattern_root->ReplaceAllUsesWith(new_pattern_root));
  TF_CHECK_OK(
      computation->RemoveInstructionAndUnusedOperands(old_pattern_root));

  return true;
}

}  // namespace poplarplugin
}  // namespace xla
