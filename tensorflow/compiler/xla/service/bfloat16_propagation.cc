/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/bfloat16_propagation.h"

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

BFloat16Propagation::BFloat16Propagation(
    const BFloat16Support* bfloat16_support)
    : bfloat16_support_(bfloat16_support) {}

void BFloat16Propagation::DetermineAndMutateFusionComputationPrecision(
    HloInstruction* fusion) {
  CHECK_EQ(fusion->opcode(), HloOpcode::kFusion);
  if (!bfloat16_support_->SupportsMixedPrecisions(*fusion)) {
    return;
  }

  // We are depending on the fusion node itself having already been analyzed
  // for whether it can output BF16 and this has been adjusted in the output
  // shape, and now we're looking to update the interior of the fusion node to
  // match the new output shape, as well as recursively process the whole fusion
  // node even if the output shape was not modified.
  auto root = fusion->fused_instructions_computation()->root_instruction();

  // Adjust root's element types according to the fusion's output shape.
  ShapeUtil::ForEachMutableSubshape(
      root->mutable_shape(), [&](Shape* subshape, const ShapeIndex& index) {
        if (subshape->element_type() != F32) {
          return;
        }
        if (ShapeUtil::GetSubshape(fusion->shape(), index).element_type() ==
            BF16) {
          subshape->set_element_type(BF16);
          changed_ = true;
          VLOG(2) << "Fused root " << root->ToString() << " at shape index "
                  << index << " changed to BF16 precision for fusion "
                  << fusion->ToString();
        }
      });

  // Propagate BF16 in the fusion computation.
  auto insts =
      fusion->fused_instructions_computation()->MakeInstructionPostOrder();
  for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
    DetermineAndMutateInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_mutation_pass_.insert(
      fusion->fused_instructions_computation());
}

void BFloat16Propagation::DetermineAndMutateWhileComputationsPrecision(
    HloInstruction* while_hlo) {
  CHECK_EQ(while_hlo->opcode(), HloOpcode::kWhile);

  // We are depending on the while node itself having already been analyzed for
  // whether it can output BF16 and this has been adjusted in the output shape,
  // and now we're looking to update the body and condition computations to
  // match the new output shape, as well as recursively process the whole while
  // node even if the output shape was not modified.
  HloComputation* body = while_hlo->while_body();
  auto body_root = body->root_instruction();
  HloComputation* condition = while_hlo->while_condition();

  ShapeUtil::ForEachMutableSubshape(
      body_root->mutable_shape(),
      [this, while_hlo, body_root](Shape* subshape, const ShapeIndex& index) {
        if (subshape->element_type() != F32) {
          return;
        }
        if (ShapeUtil::GetSubshape(while_hlo->shape(), index).element_type() ==
            BF16) {
          subshape->set_element_type(BF16);
          changed_ = true;
          VLOG(2) << "While body root " << body_root->ToString()
                  << " at shape index " << index
                  << " changed to BF16 precision for while "
                  << while_hlo->ToString();
        }
      });

  auto body_insts = body->MakeInstructionPostOrder();
  for (auto inst_it = body_insts.rbegin(); inst_it != body_insts.rend();
       ++inst_it) {
    DetermineAndMutateInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_mutation_pass_.insert(body);

  auto condition_insts = condition->MakeInstructionPostOrder();
  for (auto inst_it = condition_insts.rbegin();
       inst_it != condition_insts.rend(); ++inst_it) {
    DetermineAndMutateInstructionPrecision(*inst_it, /*skip_parameters=*/false);
  }
  computations_visited_in_mutation_pass_.insert(condition);
}

bool BFloat16Propagation::AllUsersConsumeBF16(const HloInstruction& hlo,
                                              const ShapeIndex& index) const {
  auto value_set = dataflow_->GetValueSet(&hlo, index);
  for (const HloValue* value : value_set.values()) {
    if (ContainsKey(values_that_must_be_kept_as_f32_, value)) {
      return false;
    }
    if (value->shape().element_type() == BF16) {
      continue;
    }
    for (const HloUse& use : value->uses()) {
      if (!ContainsKey(instructions_visited_in_mutation_pass_,
                       use.instruction)) {
        // We don't know yet whether use.instruction will consume BF16 since it
        // hasn't been visited. Although we visit instructions in reverse
        // topological order, this is still possible because there may be
        // unvisited instruction that alias the same buffer. In this case, we
        // aggressively skip this use, and if this causes inconsistency (e.g.,
        // one use is in BF16 but another use is in F32), it will be resolved at
        // the end of the BFloat16Propagation pass.
        continue;
      }
      // Any visited user that can accept BF16 has already been updated if
      // necessary, e.g., the output has been changed to BF16 if it propagates
      // precision, or a called computation's parameters have been changed to
      // BF16 for fusions or whiles.
      if (use.instruction->opcode() == HloOpcode::kFusion) {
        const auto* fused_parameter =
            use.instruction->fused_parameter(use.operand_number);
        if (ShapeUtil::GetSubshape(fused_parameter->shape(), use.operand_index)
                .element_type() != BF16) {
          return false;
        }
        continue;
      } else if (use.instruction->opcode() == HloOpcode::kWhile) {
        const auto* cond_parameter =
            use.instruction->while_condition()->parameter_instruction(
                use.operand_number);
        if (ShapeUtil::GetSubshape(cond_parameter->shape(), use.operand_index)
                .element_type() != BF16) {
          return false;
        }
        const auto* body_parameter =
            use.instruction->while_body()->parameter_instruction(
                use.operand_number);
        if (ShapeUtil::GetSubshape(body_parameter->shape(), use.operand_index)
                .element_type() != BF16) {
          return false;
        }
        continue;
      }
      if (bfloat16_support_->EffectiveOperandPrecisionIsBF16(
              *use.instruction, use.operand_number)) {
        continue;
      }
      // If the op propagates precision and it outputs a BF16, then it's OK to
      // supply BF16 also as the input. In the backward mutation pass, the users
      // shapes should have already been processed.
      PrimitiveType user_output_type = PRIMITIVE_TYPE_INVALID;
      if (use.instruction->opcode() == HloOpcode::kTuple ||
          (use.instruction->opcode() == HloOpcode::kCrossReplicaSum &&
           ShapeUtil::IsTuple(use.instruction->shape()))) {
        user_output_type = ShapeUtil::GetSubshape(
                               ShapeUtil::GetSubshape(use.instruction->shape(),
                                                      {use.operand_number}),
                               use.operand_index)
                               .element_type();
      } else {
        user_output_type = use.instruction->shape().element_type();
      }
      if (bfloat16_support_->EffectiveOperandPrecisionIsOutputPrecision(
              *use.instruction, use.operand_number) &&
          user_output_type == BF16) {
        continue;
      }
      return false;
    }
  }
  return true;
}

void BFloat16Propagation::DetermineAndMutateInstructionPrecision(
    HloInstruction* hlo, bool skip_parameters) {
  // We handle any fusion computation or while body/condition after the
  // instruction is handled, because we need to know the output shape of a
  // fusion or while before propagating inside its  computations.
  bool postpone_processing_called_computations = false;
  auto cleaner = tensorflow::gtl::MakeCleanup(
      [this, hlo, &postpone_processing_called_computations] {
        if (!postpone_processing_called_computations) {
          if (hlo->opcode() == HloOpcode::kFusion) {
            DetermineAndMutateFusionComputationPrecision(hlo);
          } else if (hlo->opcode() == HloOpcode::kWhile) {
            DetermineAndMutateWhileComputationsPrecision(hlo);
          }
        }
        instructions_visited_in_mutation_pass_.insert(hlo);
      });

  if (hlo->opcode() == HloOpcode::kWhile &&
      (caller_counts_[hlo->while_condition()] > 1 ||
       caller_counts_[hlo->while_body()] > 1)) {
    postpone_processing_called_computations = true;
    return;
  }

  // Do not change precision for instructions related to entry and exit of a
  // computation, and control flow, because this pass might break the interfaces
  // or assumptions for them.
  if (hlo->opcode() == HloOpcode::kInfeed ||       //
      hlo->opcode() == HloOpcode::kOutfeed ||      //
      hlo->opcode() == HloOpcode::kSend ||         //
      hlo->opcode() == HloOpcode::kSendDone ||     //
      hlo->opcode() == HloOpcode::kRecv ||         //
      hlo->opcode() == HloOpcode::kRecvDone ||     //
      hlo->opcode() == HloOpcode::kCustomCall ||   //
      hlo->opcode() == HloOpcode::kCall ||         //
      hlo->opcode() == HloOpcode::kConditional ||  //
      (hlo->opcode() == HloOpcode::kParameter && skip_parameters)) {
    return;
  }

  // Prevent root instructions from having their output modified by recording
  // all F32 output values as needing to stay as F32.
  CHECK(hlo->parent() != nullptr);
  if (hlo == hlo->parent()->root_instruction()) {
    if (!hlo->parent()->IsFusionComputation()) {
      ShapeUtil::ForEachSubshape(hlo->shape(), [&](const Shape& subshape,
                                                   const ShapeIndex& index) {
        if (subshape.element_type() != F32) {
          return;
        }
        for (const auto* value : dataflow_->GetValueSet(hlo, index).values()) {
          // Since we use HloValues from the dataflow analysis, this can also
          // affect HLO instructions beyond the root, e.g., if the root is a
          // Tuple HLO, then its operands are also affected.
          values_that_must_be_kept_as_f32_.insert(value);
        }
      });
    }
    return;
  }

  if (!ContainsKey(consider_using_bfloat16_, hlo)) {
    return;
  }

  if (!bfloat16_support_->SupportsBF16Output(*hlo)) {
    return;
  }

  ShapeUtil::ForEachMutableSubshape(
      hlo->mutable_shape(),
      [hlo, this](Shape* subshape, const ShapeIndex& index) {
        if (subshape->element_type() == F32 &&
            AllUsersConsumeBF16(*hlo, index)) {
          subshape->set_element_type(BF16);
          changed_ = true;
          VLOG(2) << "HloInstruction output at shape index " << index
                  << " changed to BF16 precision: " << hlo->ToString();
        }
      });
}

bool BFloat16Propagation::InstructionIsCandidateForBF16Output(
    HloInstruction* hlo) {
  if (!bfloat16_support_->SupportsMixedPrecisions(*hlo) &&
      hlo->opcode() != HloOpcode::kTuple &&
      hlo->opcode() != HloOpcode::kGetTupleElement &&
      hlo->shape().element_type() != BF16) {
    for (int64 i = 0; i < hlo->operand_count(); ++i) {
      if (!bfloat16_support_->EffectiveOperandPrecisionIsOutputPrecision(*hlo,
                                                                         i) ||
          !ContainsKey(consider_using_bfloat16_, hlo->operand(i))) {
        return false;
      }
    }
  }
  return true;
}

void BFloat16Propagation::AdjustCalledComputationParameters(
    HloInstruction* hlo) {
  auto adjust_computation =
      [this, hlo](HloComputation* computation,
                  tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
        // Adjust parameters.
        CHECK_EQ(operands.size(), computation->num_parameters());
        for (int64 i = 0; i < operands.size(); ++i) {
          auto parameter = computation->parameter_instruction(i);
          ShapeUtil::ForEachMutableSubshape(
              parameter->mutable_shape(),
              [this, i, hlo, &operands, parameter](Shape* subshape,
                                                   const ShapeIndex& index) {
                if (!ShapeUtil::IsLeafIndex(parameter->shape(), index)) {
                  return;
                }
                PrimitiveType operand_type =
                    ShapeUtil::GetSubshape(operands[i]->shape(), index)
                        .element_type();
                if (subshape->element_type() == operand_type) {
                  return;
                }
                CHECK(operand_type == F32 || operand_type == BF16);
                subshape->set_element_type(operand_type);
                changed_ = true;
                VLOG(2) << "Called computation parameter "
                        << parameter->ToString() << " at shape index " << index
                        << " adjusted to match operand in HLO "
                        << hlo->ToString();
              });
        }
      };

  switch (hlo->opcode()) {
    case HloOpcode::kFusion:
      adjust_computation(hlo->fused_instructions_computation(),
                         hlo->operands());
      break;
    case HloOpcode::kWhile:
      adjust_computation(hlo->while_condition(), hlo->operands());
      adjust_computation(hlo->while_body(), hlo->operands());
      break;
    default:
      break;
  }
}

void BFloat16Propagation::AdjustCalledComputationRoot(HloInstruction* hlo) {
  auto adjust_computation = [this, hlo](HloComputation* computation,
                                        const Shape& output_shape) {
    // Adjust root.
    HloInstruction* root = computation->root_instruction();
    ShapeUtil::ForEachMutableSubshape(
        root->mutable_shape(), [this, hlo, root, &output_shape](
                                   Shape* subshape, const ShapeIndex& index) {
          if (!ShapeUtil::IsLeafIndex(hlo->shape(), index)) {
            return;
          }
          const PrimitiveType output_type =
              ShapeUtil::GetSubshape(output_shape, index).element_type();
          if (subshape->element_type() == output_type) {
            return;
          }
          CHECK(output_type == F32 || output_type == BF16);
          subshape->set_element_type(output_type);
          // It's possible that output_type is F32, but the root instruction's
          // type is BF16; e.g., a fusion node's output was changed to BF16
          // initially but then adjusted back to F32, and the fusion computation
          // is now being adjusted after the fusion node.
          if (output_type == F32) {
            for (const auto* value :
                 dataflow_->GetValueSet(root, index).values()) {
              // We rely on the fact that this adjustment works in reverse
              // topological order so that called computation will be
              // processed later. Adding the value to
              // values_that_must_be_kept_as_f32_ will ensure the
              // correctness of the adjustment for HLOs that will be
              // processed later.
              values_that_must_be_kept_as_f32_.insert(value);
            }
          }
          changed_ = true;
          VLOG(2) << "Called computation root " << root->ToString()
                  << " at shape index " << index
                  << " adjusted to match output shape of " << hlo->ToString();
        });
  };

  switch (hlo->opcode()) {
    case HloOpcode::kFusion:
      adjust_computation(hlo->fused_instructions_computation(), hlo->shape());
      break;
    case HloOpcode::kWhile:
      adjust_computation(hlo->while_condition(), hlo->shape());
      adjust_computation(hlo->while_body(), hlo->shape());
      break;
    default:
      break;
  }
}

bool BFloat16Propagation::ResolveInconsistencyOfAliasingBuffersHelper(
    HloComputation* computation,
    tensorflow::gtl::FlatSet<const HloComputation*>* visited_computations) {
  bool parameter_changed = false;
  auto insts = computation->MakeInstructionPostOrder();
  // Do the adjustment on each instruction in the computation in reverse
  // topological order.
  for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
    auto hlo = *inst_it;
    auto adjust_hlo_output = [this, hlo, &parameter_changed](
                                 Shape* subshape, const ShapeIndex& index) {
      if (subshape->element_type() != F32 && subshape->element_type() != BF16) {
        return;
      }
      PrimitiveType type = BF16;
      for (const auto* value : dataflow_->GetValueSet(hlo, index).values()) {
        if (value->shape().element_type() == BF16) {
          continue;
        }
        CHECK_EQ(value->shape().element_type(), F32);
        type = F32;
        break;
      }
      // It's possible that a user has been changed from BF16 to F32
      // during this final adjustment pass, so we need to check
      // AllUsersConsumeBF16() again.
      if (type == BF16 && !AllUsersConsumeBF16(*hlo, index)) {
        type = F32;
      }
      if (type == F32) {
        for (const auto* value : dataflow_->GetValueSet(hlo, index).values()) {
          // We rely on the fact that this adjustment works in reverse
          // topological order. Adding the value to
          // values_that_must_be_kept_as_f32_ will ensure the correctness
          // of the adjustment for HLOs that will be processed later.
          values_that_must_be_kept_as_f32_.insert(value);
        }
      }
      if (type != subshape->element_type()) {
        subshape->set_element_type(type);
        VLOG(2) << "HloInstruction output at shape index " << index
                << " adjusted to " << *subshape << ": " << hlo->ToString();
        if (hlo->opcode() == HloOpcode::kParameter) {
          parameter_changed = true;
        }
      }
    };
    ShapeUtil::ForEachMutableSubshape(hlo->mutable_shape(), adjust_hlo_output);
    AdjustCalledComputationRoot(hlo);
    if (hlo->opcode() == HloOpcode::kWhile) {
      // We need to run on the while body and condition repeatedly until a fixed
      // point is reached, i.e., the parameters do not change any more. We may
      // need more than one iteration because the while input and output alias
      // each other, so changing one input parameter requires changing the
      // corresponding output element and thus may transitively require changing
      // another input parameter. A fixed point will be reached because the
      // parameters can only be changed from BF16 to F32, not the other way
      // around.
      tensorflow::gtl::FlatSet<const HloComputation*> visited_in_while;
      while (ResolveInconsistencyOfAliasingBuffersHelper(hlo->while_condition(),
                                                         &visited_in_while) ||
             ResolveInconsistencyOfAliasingBuffersHelper(hlo->while_body(),
                                                         &visited_in_while)) {
        visited_in_while.clear();
        ShapeUtil::ForEachMutableSubshape(hlo->mutable_shape(),
                                          adjust_hlo_output);
        AdjustCalledComputationRoot(hlo);
      }
      visited_computations->insert(visited_in_while.begin(),
                                   visited_in_while.end());
    }
  }
  // Now adjust parameters of called computations.
  for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
    AdjustCalledComputationParameters(*inst_it);
  }
  return parameter_changed;
}

Status BFloat16Propagation::ResolveInconsistencyOfAliasingBuffers(
    HloModule* module) {
  std::list<HloComputation*> computations_topological_order =
      module->MakeComputationPostOrder();
  tensorflow::gtl::FlatSet<const HloComputation*> resolved;
  for (auto comp_it = computations_topological_order.rbegin();
       comp_it != computations_topological_order.rend(); ++comp_it) {
    if (ContainsKey(resolved, *comp_it)) {
      continue;
    }
    ResolveInconsistencyOfAliasingBuffersHelper(*comp_it, &resolved);
  }

  // We could have changed a fusion computation's root shape to have a different
  // precision than the fusion node's output, if the fusion root does not
  // define a buffer (e.g., a tuple). Now we add conversions after such fusion
  // roots to make them match the fusion output. If the fusion output is a
  // (possibly nested) tuple, we first create get-tuple-elements, then convert
  // the unmatching leaf nodes, and finally create a new tuple as the fusion
  // computation's root. If tuples and get-tuple-elements are created, we will
  // run tuple simplifier and dead code elimination at the end (dead code is not
  // allowed in fusion computation). E.g.,
  //
  // (1)             (2)             (3)
  // a  b            a  b            a  b
  // |\ |            |\ |            |\ |
  // \ add   ->      |add    ->      | add
  //  \ |            \ |        convert |
  //  tuple         tuple             \ |
  //                 / \              tuple
  //               gte gte
  //                |   |
  //           convert  |
  //                 \  /
  //                 tuple
  // (1) a is F32 but tuple is BF16
  // (2) after adding conversion
  // (3) after tuple simplifier and DCE.
  bool needs_tuple_simplifier = false;
  for (auto computation : computations_topological_order) {
    auto insts = computation->MakeInstructionPostOrder();
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      auto hlo = *inst_it;
      if (hlo->opcode() != HloOpcode::kFusion) {
        continue;
      }
      auto fusion_computation = hlo->fused_instructions_computation();
      auto fusion_root = fusion_computation->root_instruction();
      if (ShapeUtil::Compatible(fusion_root->shape(), hlo->shape())) {
        continue;
      }
      ShapeTree<HloInstruction*> converted_outputs(hlo->shape());
      // Iterate through nodes in the shape tree in pre-order and initialize
      // each non-root node with a corresponding get-tuple-element. For a leaf
      // node, if its shape does not match the fusion output, create a
      // conversion node to overwrite the node value.
      for (auto it = converted_outputs.begin(); it != converted_outputs.end();
           ++it) {
        ShapeIndex output_index = it->first;
        HloInstruction*& output = it->second;
        const Shape subshape =
            ShapeUtil::GetSubshape(hlo->shape(), output_index);
        if (output_index.empty()) {
          output = fusion_root;
        } else {
          ShapeIndex parent_index = output_index;
          parent_index.pop_back();
          output = fusion_computation->AddInstruction(
              HloInstruction::CreateGetTupleElement(
                  subshape, converted_outputs.element(parent_index),
                  output_index.back()));
        }
        if (ShapeUtil::IsTuple(subshape)) {
          continue;
        }
        if (!ShapeUtil::Compatible(
                subshape,
                ShapeUtil::GetSubshape(fusion_root->shape(), output_index))) {
          output = fusion_computation->AddInstruction(
              HloInstruction::CreateConvert(subshape, output));
        }
      }
      // Iterate through nodes in the shape tree in reverse pre-order and create
      // a tuple instruction for each non-leaf node where the elements are the
      // values of its child nodes.
      for (auto it = converted_outputs.rbegin(); it != converted_outputs.rend();
           ++it) {
        ShapeIndex output_index = it->first;
        HloInstruction*& output = it->second;
        const Shape& subshape =
            ShapeUtil::GetSubshape(hlo->shape(), output_index);
        if (!ShapeUtil::IsTuple(subshape)) {
          continue;
        }
        std::vector<HloInstruction*> elements(
            ShapeUtil::TupleElementCount(subshape));
        ShapeIndex child_index = output_index;
        for (int64 i = 0; i < elements.size(); ++i) {
          child_index.push_back(i);
          elements[i] = converted_outputs.element(child_index);
          child_index.pop_back();
        }
        output = fusion_computation->AddInstruction(
            HloInstruction::CreateTuple(elements));
      }
      fusion_computation->set_root_instruction(converted_outputs.element({}));
      needs_tuple_simplifier |= ShapeUtil::IsTuple(hlo->shape());
    }
  }

  // We may have converted some constants from F32 to BF16, so adjust the
  // constant literals in such cases. We do this here instead of when the
  // constant node's is changed because 1) the HloInstruction interface does not
  // allow resetting the literal so we have to create a new kConstant
  // instruction to replace the old one, which invalidates dataflow analysis,
  // and 2) it's possible that a kConstant's output gets changed to BF16 at the
  // beginning but later on adjusted back to F32, so converting literals here
  // can avoid repeated conversions.
  //
  // TODO(b/73833576): Consider resetting literal in HloInstruction.
  bool needs_dce = needs_tuple_simplifier;
  for (auto computation : computations_topological_order) {
    for (auto hlo : computation->MakeInstructionPostOrder()) {
      if (hlo->opcode() != HloOpcode::kConstant) {
        continue;
      }
      if (!ShapeUtil::Equal(hlo->literal().shape(), hlo->shape())) {
        TF_ASSIGN_OR_RETURN(auto converted_literal,
                            hlo->literal().ConvertToShape(hlo->shape()));
        auto new_constant = computation->AddInstruction(
            HloInstruction::CreateConstant(std::move(converted_literal)));
        TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(new_constant));
        needs_dce = true;
      }
    }
  }

  if (needs_tuple_simplifier) {
    TupleSimplifier tuple_simplifier;
    TF_RETURN_IF_ERROR(tuple_simplifier.Run(module).status());
  }
  if (needs_dce) {
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }
  return Status::OK();
}

// The algorithm first does a forward pass (parameters to root) to determine a
// set of instructions to consider using bfloat16, then does a backward pass to
// determine the precisions of those instructions according to the need of
// their users.
StatusOr<bool> BFloat16Propagation::Run(HloModule* module) {
  TF_ASSIGN_OR_RETURN(dataflow_, HloDataflowAnalysis::Run(*module));

  std::list<HloComputation*> computations_topological_order =
      module->MakeComputationPostOrder();
  // The first step is a forward pass (parameters to root), where we determine
  // the potential candidate instructions to use bfloat16 in the outputs that
  // are not likely to cause overhead from extra explicit conversions. This is
  // done forwardly because we determine whether an HLO is a candidate partially
  // based on whether its operands are candidates.
  for (auto computation : computations_topological_order) {
    for (auto inst : computation->MakeInstructionPostOrder()) {
      if (InstructionIsCandidateForBF16Output(inst)) {
        consider_using_bfloat16_.insert(inst);
      }
    }
  }

  // The second step is a backward pass (root to parameters), where we modify
  // the precisions of the instructions identified in the first step when
  // feasible. This is done backwardly because we determine the precision of an
  // HLO's output based on how it is later used.
  //
  // The precision of an instruction is determined by its users, so we do the
  // propagation in reverse topological order.
  for (auto comp_it = computations_topological_order.rbegin();
       comp_it != computations_topological_order.rend(); ++comp_it) {
    if ((*comp_it)->IsFusionComputation()) {
      // Fusion computations are handled when visiting the fusion instruction.
      continue;
    }
    auto insts = (*comp_it)->MakeInstructionPostOrder();
    for (auto inst_it = insts.rbegin(); inst_it != insts.rend(); ++inst_it) {
      DetermineAndMutateInstructionPrecision(*inst_it,
                                             /*skip_parameters=*/true);
    }
  }

  if (!changed_) {
    return false;
  }

  // It's possible that an instruction does not define a buffer, but the
  // defining instruction's shape has changed. So we need to adjust the output
  // shapes of instructions according to the HLO values they refer to.
  TF_RETURN_IF_ERROR(ResolveInconsistencyOfAliasingBuffers(module));
  return true;
}

}  // namespace xla
