/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/while_loop_unroller.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/algorithm.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/while_loop_analysis.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_fix.h"
#include "xla/hlo/transforms/simplifiers/flatten_call_graph.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/overflow_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/constant_value.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_cse.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/scheduling_annotations_util.h"
#include "xla/service/value_range.h"
#include "xla/service/while_loop_constant_sinking.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using hlo_query::ContainsInstrWithOpcode;

// Helper function to create a condition for a single iteration while loop in
// the form of 'i <= init_value' where i is the induction variable.
std::unique_ptr<HloComputation> MakeTrivialLoopCondition(
    HloInstruction* while_op, absl::string_view name, int64_t induction_idx,
    int64_t init_value) {
  auto condition_builder = HloComputation::Builder(name);

  absl::StatusOr<HloInstruction*> param_instruction =
      condition_builder.AddParameter(
          while_op->while_condition()->parameter_instruction(0)->Clone());

  HloInstruction* indvar_instruction =
      condition_builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          param_instruction.value(), induction_idx));

  HloInstruction* init_value_constant = condition_builder.AddInstruction(
      MakeScalarConstantWithShape(indvar_instruction->shape(), init_value));

  return condition_builder.Build(
      condition_builder.AddInstruction(HloInstruction::CreateCompare(
          ShapeUtil::MakeValidatedShape(PrimitiveType::PRED, {}).value(),
          indvar_instruction, init_value_constant, ComparisonDirection::kLe)));
}

// Handle DynamicGte and DynamicTuple custom-calls created during unstacking
// pass.
absl::Status HandleDynamicGteOrTuple(HloInstruction* instr) {
  if (instr->IsCustomCall("DynamicGte")) {
    HloEvaluator evaluator(/*max_loop_iterations=*/0);
    TF_ASSIGN_OR_RETURN(
        Literal index_lit,
        evaluator.Evaluate(instr->mutable_operand(1),
                           /*precomputed_analyses=*/{},
                           /*recursively_evaluate_nonconstant_operands=*/true));
    auto index = LiteralUtil::LiteralAsScalarInt64(std::move(index_lit));
    // The index must have a compile-time integer value at this point.
    TF_RET_CHECK(index.has_value());
    return instr->parent()->ReplaceInstruction(
        instr, instr->AddInstruction(HloInstruction::CreateGetTupleElement(
                   instr->mutable_operand(0), index.value())));
  } else if (instr->IsCustomCall("DynamicTuple")) {
    HloEvaluator evaluator(/*max_loop_iterations=*/0);
    std::vector<HloInstruction*> tuple_operands;
    TF_ASSIGN_OR_RETURN(
        Literal index_lit,
        evaluator.Evaluate(instr->mutable_operand(2),
                           /*precomputed_analyses=*/{},
                           /*recursively_evaluate_nonconstant_operands=*/true));
    auto index = LiteralUtil::LiteralAsScalarInt64(std::move(index_lit));
    // The index must have a compile-time integer value at this point.
    TF_RET_CHECK(index.has_value());
    for (int64_t i = 0; i < instr->operand(0)->shape().tuple_shapes().size();
         i++) {
      if (i == index.value()) {
        tuple_operands.push_back(instr->mutable_operand(1));
      } else {
        HloInstruction* slice =
            instr->AddInstruction(HloInstruction::CreateGetTupleElement(
                instr->mutable_operand(0), i));
        tuple_operands.push_back(slice);
      }
    }
    return instr->parent()->ReplaceInstruction(
        instr,
        instr->AddInstruction(HloInstruction::CreateTuple(tuple_operands)));
  }
  return absl::OkStatus();
}

// Replaces all uses of the gte induction variable hlo (except the increment
// instruction) with a constant. We use induction_var_idx to find the gte
// instruction.
absl::Status ReplaceInductionVarUses(HloComputation* body,
                                     HloInstruction* induction_value_constant,
                                     int64_t induction_var_idx) {
  for (HloInstruction* body_inst : body->instructions()) {
    // We only consider induction variable instructions of the following form.
    if (!Match(body_inst,
               match::GetTupleElement(match::Parameter().WithParameterNum(0))
                   .WithTupleIndex(induction_var_idx))) {
      continue;
    }

    // Store users of the induction variable in a separate vector to go over.
    std::vector<HloInstruction*> indvar_uses;
    indvar_uses.reserve(body_inst->users().size());
    for (HloInstruction* indvar_use : body_inst->users()) {
      indvar_uses.push_back(indvar_use);
    }

    // Finds all the uses of induction var within the while body and replace it
    // with the constant.
    for (HloInstruction* indvar_use : indvar_uses) {
      // Skip the induction variable increment instruction. We need this
      // instruction to remain in the loop if we are doing wrapped unrolling. We
      // rely on this instruction to later find and remove these trivial loops.
      if (Match(indvar_use, match::Add(match::GetTupleElement().WithTupleIndex(
                                           induction_var_idx),
                                       match::Constant()))) {
        continue;
      }
      for (int64_t i = 0; i < indvar_use->operand_count(); ++i) {
        const HloInstruction* indvar_use_operand = indvar_use->operand(i);
        // Found the induction var user.
        if (indvar_use_operand == body_inst) {
          TF_RETURN_IF_ERROR(
              indvar_use->ReplaceOperandWith(i, induction_value_constant));
        }
      }
    }
  }
  return absl::OkStatus();
}

// Helper function that replaces a single iteration of a while loop with
// induction variable equal to induction_value.
absl::StatusOr<std::unique_ptr<HloComputation>>
UnrollSingleIterationOfTrivialLoop(HloInstruction* while_op,
                                   WhileLoopConfig config,
                                   const int64_t induction_value,
                                   int64_t& next_scheduling_id) {
  // We clone the body since we are changing the computation.
  std::unique_ptr<HloComputation> while_body_clone =
      while_op->while_body()->Clone(
          absl::StrCat(while_op->name(), induction_value));

  HloInstruction* induction_var_hlo =
      while_op->mutable_operand(0)->mutable_operand(config.induction_var_idx);

  // We record the next channel id to utilize when unrolling loops with
  // collective communication instructions. During unrolling a single iteration
  // of the body, we can reuse the same unique_channel_id. For the later
  // iterations, we obtain it again.
  int64_t unique_channel_id = hlo_query::NextChannelId(*while_op->GetModule());

  HloInstruction* induction_value_constant = while_body_clone->AddInstruction(
      MakeScalarConstantWithShape(induction_var_hlo->shape(), induction_value));
  TF_RETURN_IF_ERROR(ReplaceInductionVarUses(while_body_clone.get(),
                                             induction_value_constant,
                                             config.induction_var_idx));

  absl::flat_hash_set<int64_t> seen_scheduling_ids;
  for (HloInstruction* body_inst : while_body_clone->instructions()) {
    // We need to assign a unique channel_id for the collective ops that are
    // unrolled within the while loop body or fusions containing collectives.
    HloInstruction* collective = IsOrHasCollectiveWithChannelId(body_inst);
    if (collective != nullptr) {
      // To obtain the channel_id for the collective ops we only need to
      // increment the `unique_channel_id` since it records the next available
      // channel_id across the module.
      collective->set_channel_id(unique_channel_id++);
    }

    // We need to assign a unique id to each scheduling group (of instructions)
    // that are unrolled within the while loop body.
    TF_ASSIGN_OR_RETURN(std::optional<int64_t> scheduling_id,
                        GetSchedulingAnnotationGroupId(body_inst));
    if (scheduling_id.has_value()) {
      if (!seen_scheduling_ids.contains(scheduling_id.value())) {
        seen_scheduling_ids.insert(scheduling_id.value());
        next_scheduling_id++;
      }
      TF_RETURN_IF_ERROR(
          SetSchedulingAnnotationGroupId(body_inst, next_scheduling_id));
    }

    // Handle DynamicGte and DynamicTuple custom-calls created during unstacking
    // pass. All custom-calls must be replaced for the loop to be unrolled
    // successfully.
    TF_RETURN_IF_ERROR(HandleDynamicGteOrTuple(body_inst));
  }
  return while_body_clone;
}

// Checks the soft conditions of unrollability. Soft conditions are:
// 1. num instructions in loop body.
// 2. trip count.
// 3. unroll expansion limit (#_body_instructions * trip_count).
// These conditions can be changed per usecase.
bool InitialFeasibilityCheck(const HloInstruction* while_op,
                             const WhileLoopConfig config,
                             const UnrollConfig unroll_config) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);

  VLOG(5) << "Trying to unroll " << while_op->ToShortString();

  // We don't attempt to unroll loops where the body has more than
  // kUnrollInstructionCountThreshold instructions.
  if (while_op->while_body()->instruction_count() >
      unroll_config.instruction_count_threshold) {
    VLOG(5) << absl::StrCat(
        "Cannot unroll while loop. Too many instructions in the body: ",
        while_op->while_body()->instruction_count());
    return false;
  }

  // We only unroll loops up to a threshold.
  if (config.trip_count > unroll_config.trip_count_threshold) {
    VLOG(5) << absl::StrCat(
        "Cannot unroll while loop. The trip count is greater "
        "than the threshold: ",
        config.trip_count, " vs ", unroll_config.trip_count_threshold);
    return false;
  }

  // We don't unroll loops that increase the instruction count by more than
  // kUnrollExpandFactorThreshold.
  if (config.trip_count * while_op->while_body()->instruction_count() >
      unroll_config.expand_factor_threshold) {
    VLOG(5) << absl::StrCat(
        "Not attempting to unroll due to instruction count "
        "increase explosion. New instruction count: ",
        config.trip_count * while_op->while_body()->instruction_count(), " vs ",
        unroll_config.expand_factor_threshold);
    return false;
  }
  return true;
}

absl::StatusOr<bool> UnrollInternal(HloInstruction* while_op,
                                    WhileLoopConfig config) {
  VLOG(3) << "Unrolling while instruction " << while_op->ToShortString()
          << " with body instruction count "
          << while_op->while_body()->instruction_count();
  HloModule* module = while_op->GetModule();
  HloComputation* computation = while_op->parent();
  HloInstruction* unrolled_body_call_op;
  std::vector<HloInstruction*> call_operands = {while_op->operands().at(0)};

  TF_ASSIGN_OR_RETURN(int64_t next_scheduling_id,
                      NextSchedulingGroupId(*while_op->GetModule()));
  for (int64_t i = config.init; i < config.trip_count + config.init; ++i) {
    CHECK(OverflowSafeAdd(i, (int64_t)1).has_value());

    HloComputation* unrolled_body = module->AddEmbeddedComputation(
        UnrollSingleIterationOfTrivialLoop(while_op, config, i,
                                           next_scheduling_id)
            .value());
    unrolled_body_call_op =
        computation->AddInstruction(HloInstruction::CreateCall(
            while_op->shape(), call_operands, unrolled_body));
    call_operands.clear();
    call_operands.push_back(unrolled_body_call_op);
  }
  TF_RETURN_IF_ERROR(
      computation->ReplaceInstruction(while_op, unrolled_body_call_op));

  // Needed for the nested while loops in which the outer loop has been
  // unrolled which leaves the call graph non-flat.
  TF_RETURN_IF_ERROR(FlattenCallGraph().Run(module).status());
  return true;
}

absl::StatusOr<UnrollResult> UnrollInternalWrappedAndReturnReplacement(
    HloInstruction* while_op, WhileLoopConfig config) {
  VLOG(3) << "Unrolling (wrapped) while instruction "
          << while_op->ToShortString() << " with body instruction count "
          << while_op->while_body()->instruction_count();
  HloModule* module = while_op->GetModule();

  HloComputation* computation = while_op->parent();
  HloInstruction* unrolled_body_call_op;
  std::vector<HloInstruction*> call_operands;

  auto body_builder =
      HloComputation::Builder(absl::StrCat("unrolled-body-", while_op->name()));
  absl::StatusOr<HloInstruction*> p = body_builder.AddParameter(
      while_op->while_body()->parameter_instruction(0)->Clone());

  // We assume while has only one tuple parameter
  call_operands.emplace_back(std::move(p.value()));

  TF_ASSIGN_OR_RETURN(int64_t next_scheduling_id,
                      NextSchedulingGroupId(*while_op->GetModule()));
  for (int64_t i = config.init; i < config.trip_count + config.init; ++i) {
    CHECK(OverflowSafeAdd(i, (int64_t)1).has_value());

    HloComputation* unrolled_body = module->AddEmbeddedComputation(
        UnrollSingleIterationOfTrivialLoop(while_op, config, i,
                                           next_scheduling_id)
            .value());

    unrolled_body_call_op = body_builder.AddInstruction(
        HloInstruction::CreateCall(while_op->shape(), call_operands,
                                   unrolled_body),
        absl::StrCat(while_op->name(), "-unrolled-body-call-", i));

    call_operands.clear();
    call_operands.push_back(unrolled_body_call_op);
  }
  HloComputation* new_body =
      module->AddEmbeddedComputation(body_builder.Build(unrolled_body_call_op));
  HloComputation* new_cond =
      module->AddEmbeddedComputation(MakeTrivialLoopCondition(
          while_op, absl::StrCat("unrolled", while_op->name(), "-cond"),
          config.induction_var_idx, config.init));

  HloInstruction* new_while_op =
      computation->AddInstruction(HloInstruction::CreateWhile(
          while_op->shape(), new_cond, new_body, while_op->mutable_operand(0)));
  while_op->SetupDerivedInstruction(new_while_op);
  CHECK_OK(computation->ReplaceInstruction(while_op, new_while_op));

  // Needed for the nested while loops in which the outer loop has been
  // unrolled which leaves the call graph non-flat.
  TF_RETURN_IF_ERROR(FlattenCallGraph().Run(module).status());
  UnrollResult result;
  result.unrolled = true;
  result.new_while_op = new_while_op;
  return result;
}

absl::StatusOr<bool> UnrollInternalWrapped(HloInstruction* while_op,
                                           WhileLoopConfig config) {
  TF_ASSIGN_OR_RETURN(
      UnrollResult result,
      UnrollInternalWrappedAndReturnReplacement(while_op, config));
  return result.unrolled;
}

// Recursively checks if the given instruction points to the induction var of
// the given loop config.
bool IsLoopInductionVar(const HloInstruction* instr,
                        const WhileLoopConfig& config) {
  if (!instr->parent()->IsFusionComputation()) {
    return Match(instr, match::GetTupleElement(match::Parameter(),
                                               config.induction_var_idx));
  } else {
    if (!Match(instr, match::Parameter())) {
      return false;
    }
    HloInstruction* caller_fusion = instr->parent()->FusionInstruction();
    return IsLoopInductionVar(caller_fusion->operand(instr->parameter_number()),
                              config);
  }
}

// Recursively checks if the given instruction inside a while loop can be
// expressed as a value range, possibly depending on the loop induction variable
// of that while loop.
std::optional<Range> IdentifyRangeAsFunctionOfInductionVar(
    const HloInstruction* instr, const WhileLoopConfig& config) {
  if (instr->parent()->IsFusionComputation()) {
    if (!Match(instr, match::Parameter())) {
      return std::nullopt;
    }
    HloInstruction* caller_fusion = instr->parent()->FusionInstruction();
    return IdentifyRangeAsFunctionOfInductionVar(
        caller_fusion->operand(instr->parameter_number()), config);
  }

  std::optional<Range> loop_range = MatchTrivialLoopRange(config.while_instr);
  if (loop_range == std::nullopt) {
    return std::nullopt;
  }

  const HloComputation* while_body = config.while_instr->while_body();
  absl::flat_hash_map<const HloInstruction*, Range> predefined_ranges;
  HloInstruction* while_body_input_tuple = while_body->parameter_instruction(0);
  for (HloInstruction* user : while_body_input_tuple->users()) {
    if (Match(user, match::GetTupleElement(match::Parameter(0),
                                           config.induction_var_idx))) {
      predefined_ranges[user] = loop_range.value();
    }
  }

  Range instr_range =
      RecursivelyIdentifyRange(instr, predefined_ranges, nullptr);
  return instr_range;
}

};  // namespace

// Recursively checks if the given instruction is effectively static by checking
// if it is a constant or a parameter that points to the induction var of the
// given loop config.
bool IsEffectivelyStatic(const HloInstruction* instr,
                         const WhileLoopConfig& config) {
  switch (instr->opcode()) {
    case HloOpcode::kConstant:
      return true;
    case HloOpcode::kParameter: {
      if (instr->parent()->IsFusionComputation()) {
        HloInstruction* caller_fusion = instr->parent()->FusionInstruction();
        return IsEffectivelyStatic(
            caller_fusion->operand(instr->parameter_number()), config);
      }
      return false;
    }
    case HloOpcode::kGetTupleElement: {
      if (instr->parent() != config.while_instr->while_body()) {
        return false;
      }
      if (!Match(instr, match::GetTupleElement(match::Parameter(),
                                               config.induction_var_idx))) {
        return false;
      }
      return true;
    }
    default: {
      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        if (!IsEffectivelyStatic(instr->operand(i), config)) {
          return false;
        }
      }
      return true;
    }
  }
}

std::optional<int64_t> MatchEffectivelyStaticDynamicSliceInsideLoop(
    const HloInstruction* instr, const HloInstruction* input,
    const WhileLoopConfig& config) {
  if (instr->opcode() != HloOpcode::kDynamicSlice) {
    return std::nullopt;
  }
  int64_t start_indices_offset = 1;
  const HloInstruction* operand = instr->operand(0);
  if (operand != input) {
    VLOG(3) << "Input of dynamic index instruction is not the given operand.";
    return std::nullopt;
  }

  int64_t dynamic_index = -1;
  for (int64_t start_index = start_indices_offset;
       start_index < instr->operand_count(); ++start_index) {
    const HloInstruction* index = instr->operand(start_index);
    // All constants must be zero in order to slice the entire shape.
    if (Match(index, match::ConstantScalar())) {
      std::optional<int64_t> offset =
          LiteralUtil::LiteralAsScalarInt64(index->literal());
      if (offset.has_value() && offset.value() != 0) {
        VLOG(3) << "Constant index " << start_index << " is not zero.";
        return std::nullopt;
      }
      continue;
    }
    if (IsEffectivelyStatic(index, config)) {
      if (dynamic_index != -1) {
        VLOG(3) << "Multiple non-constant indices.";
        return std::nullopt;
      }
      dynamic_index = start_index - start_indices_offset;
    }
  }

  if (dynamic_index == -1) {
    VLOG(3) << "No dynamic index found.";
    return std::nullopt;
  }

  return dynamic_index;
}

std::optional<int64_t> MatchShapeCoveringDynamicIndexInstruction(
    const HloInstruction* instr, const HloInstruction* input, HloOpcode opcode,
    const WhileLoopConfig& config) {
  if (instr->opcode() != opcode) {
    return std::nullopt;
  }
  // Based on the instruction type, start indices start from index 1 or 2 of the
  // operands.
  int64_t start_indices_offset;
  if (instr->opcode() == HloOpcode::kDynamicSlice) {
    start_indices_offset = 1;
  } else if (instr->opcode() == HloOpcode::kDynamicUpdateSlice) {
    start_indices_offset = 2;
  } else {
    return std::nullopt;
  }
  const HloInstruction* operand = instr->operand(0);
  if (input != nullptr && operand != input) {
    VLOG(3) << "Input of dynamic index instruction is not the given operand.";
    return std::nullopt;
  }

  int64_t dynamic_index = -1;
  for (int64_t start_index = start_indices_offset;
       start_index < instr->operand_count(); ++start_index) {
    const HloInstruction* index = instr->operand(start_index);
    // All constants must be zero in order to slice the entire shape.
    if (Match(index, match::ConstantScalar())) {
      std::optional<int64_t> offset =
          LiteralUtil::LiteralAsScalarInt64(index->literal());
      if (offset.has_value() && offset.value() != 0) {
        VLOG(3) << "Constant index " << start_index << " is not zero.";
        return std::nullopt;
      }
      continue;
    }

    // Check that the instruction's dynamic index points to the loop induction
    // variable.
    if (IsLoopInductionVar(index, config)) {
      // In order to cover the whole shape only a single non-constant index is
      // allowed.
      if (dynamic_index != -1) {
        VLOG(3) << "Multiple non-constant indices.";
        return std::nullopt;
      }
      dynamic_index = start_index - start_indices_offset;
    }
  }

  if (dynamic_index == -1) {
    VLOG(3) << "No dynamic index found.";
    return std::nullopt;
  }

  if (operand->shape().dimensions(dynamic_index) != config.trip_count) {
    VLOG(3) << "The dynamic_index dimension size of the operand must be equal "
               "to the loop trip count.";
    return std::nullopt;
  }

  if (opcode == HloOpcode::kDynamicSlice) {
    const Shape& result_shape = instr->shape();
    if (result_shape.dimensions(dynamic_index) != 1) {
      VLOG(3) << "The slice size on the dynamic_index dimension must be 1.";
      return std::nullopt;
    }

    const Shape& operand_shape = operand->shape();
    CHECK_EQ(result_shape.dimensions().size(),
             operand_shape.dimensions().size());
    for (int64_t i = 0; i < result_shape.dimensions().size(); ++i) {
      if (i != dynamic_index &&
          result_shape.dimensions(i) != operand_shape.dimensions(i)) {
        VLOG(3) << "The slice sizes must match the operand-shape on "
                   "non-dynamic-index dimensions.";
        return std::nullopt;
      }
    }
  }

  return dynamic_index;
}

// TODO(b/393399049): Replace MatchShapeCoveringDynamicInstruction with this
// one.
// Compared to the MatchShapeCoveringDynamicInstruction() method above, this
// implementation determines whether the (single) dynamic dimension is fully
// coverd by simulating the loop and noting which indices have been covered at
// any point.
std::optional<int64_t> AdvancedMatchShapeCoveringDynamicIndexInstruction(
    const HloInstruction* instr, const HloInstruction* input, HloOpcode opcode,
    const WhileLoopConfig& config) {
  if (instr->opcode() != opcode) {
    return std::nullopt;
  }
  // Based on the instruction type, start indices start from index 1 or 2 of the
  // operands and the slice shape is either the shape of instr (i.e. its output
  // shape) or the shape of its operand at index 1.
  int64_t start_indices_offset;
  const Shape* slice_shape;
  if (instr->opcode() == HloOpcode::kDynamicSlice) {
    start_indices_offset = 1;
    slice_shape = &instr->shape();
  } else if (instr->opcode() == HloOpcode::kDynamicUpdateSlice) {
    start_indices_offset = 2;
    slice_shape = &instr->operand(1)->shape();
  } else {
    return std::nullopt;
  }

  if (input != nullptr && input != instr->operand(0)) {
    VLOG(3) << "Input of dynamic index instruction is not the given operand.";
    return std::nullopt;
  }
  input = instr->operand(0);
  const Shape& input_shape = input->shape();

  const int64_t num_indices = slice_shape->dimensions().size();
  CHECK_EQ(num_indices, input_shape.dimensions().size());
  CHECK_EQ(num_indices, instr->operand_count() - start_indices_offset);

  std::vector<int64_t> dynamic_indices;
  for (int64_t index = 0; index < num_indices; ++index) {
    int64_t start_index_offset = start_indices_offset + index;
    const HloInstruction* start_index = instr->operand(start_index_offset);

    if (!Match(start_index, match::ConstantScalar())) {
      dynamic_indices.push_back(index);
      continue;
    }
    // This is a non-dynamic index. It must start at zero and have a slice
    // size matching the input size.
    if (!Match(start_index, match::ConstantScalar(0))) {
      VLOG(3) << "Non-dynamic-index dimensions must start at zero; "
                 "nonzero at index "
              << index;
      return std::nullopt;
    }
    if (slice_shape->dimensions(index) != input_shape.dimensions(index)) {
      VLOG(3) << "The slice sizes must match the input shape on "
                 "non-dynamic-index dimensions; mismatch at index "
              << index;
      return std::nullopt;
    }
  }

  if (dynamic_indices.empty()) {
    VLOG(3) << "No dynamic index found.";
    return std::nullopt;
  }
  if (dynamic_indices.size() >= 2) {
    VLOG(3) << "Too many dynamic indices; found " << dynamic_indices.size();
    return std::nullopt;
  }

  std::optional<int64_t> dynamic_index = dynamic_indices[0];
  std::optional<Range> dynamic_index_range =
      IdentifyRangeAsFunctionOfInductionVar(
          instr->operand(start_indices_offset + dynamic_indices[0]), config);
  if (dynamic_index_range == std::nullopt ||
      !dynamic_index_range->IsBounded() ||
      !dynamic_index_range->IsStepKnown()) {
    VLOG(3) << "Could not compute compact dynamic index range.";
    return std::nullopt;
  }

  const int64_t dimension_size = input_shape.dimensions(dynamic_index.value());
  // We keep a boolean per possible index of the dynamic dimension, initially
  // false.
  std::vector<bool> indices_covered(dimension_size);
  const int64_t slice_size = slice_shape->dimensions(dynamic_index.value());

  // Here, we simulate the loop based on the xla::Range that we have computed
  // to represent the input to the DS/DUS.
  for (int64_t start_index_value = dynamic_index_range->min().GetSignedValue();
       start_index_value <= dynamic_index_range->max()->GetSignedValue();
       start_index_value += dynamic_index_range->step()->GetSignedValue()) {
    // DS and DUS clamp start indices so that the entire region is in-bounds.
    int64_t clamped_start_index_value = std::min(
        std::max<int64_t>(start_index_value, 0), dimension_size - slice_size);
    // The DS/DUS covers `slice_size` many indices.
    for (int64_t index = clamped_start_index_value;
         index < clamped_start_index_value + slice_size; ++index) {
      indices_covered[index] = true;
    }
  }

  for (int index = 0; index < indices_covered.size(); ++index) {
    if (!indices_covered[index]) {
      VLOG(3) << "Index " << index << " was not covered.";
      return std::nullopt;
    }
  }
  return dynamic_index;
}

/*static*/ std::optional<WhileLoopConfig> WhileLoopUnroller::IsLoopUnrollable(
    HloInstruction* while_op) {
  CHECK_EQ(while_op->opcode(), HloOpcode::kWhile);

  // While loop must have a single tuple operand.
  CHECK_EQ(while_op->operands().size(), 1);
  if (while_op->operands().size() != 1) {
    VLOG(5) << absl::StrCat(
        "Cannot unroll while loop ", while_op->name(),
        ". While loop must have a single "
        "tuple operand, instead has more than one operand: ",
        while_op->operands().size());
    return std::nullopt;
  }

  // TODO(b/300668690): Add support for unrolling loops with control dependency.
  // For now, we bail.
  //
  // Finding all the while loops where other instructions have explicit control
  // dependencies on them.
  std::vector<HloInstruction*> while_dependees;
  for (HloComputation* comp : while_op->GetModule()->computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      for (HloInstruction* control_dep : instr->control_predecessors()) {
        if (control_dep->opcode() == HloOpcode::kWhile) {
          while_dependees.push_back(control_dep);
        }
      }
    }
  }
  if (absl::linear_search(while_dependees.begin(), while_dependees.end(),
                          while_op)) {
    VLOG(2) << "Not attempting to unroll " << while_op->name()
            << " due to control dependency: " << while_op->ToShortString();
    return std::nullopt;
  }

  // We can't remove while loops that contain send/recv nodes, because we
  // rely on the particular loop structure around the node matching on the
  // send and recv sides.
  if (ContainsInstrWithOpcode(while_op->while_body(),
                              {HloOpcode::kSend, HloOpcode::kSendDone,
                               HloOpcode::kRecv, HloOpcode::kRecvDone}) ||
      ContainsInstrWithOpcode(while_op->while_condition(),
                              {HloOpcode::kSend, HloOpcode::kSendDone,
                               HloOpcode::kRecv, HloOpcode::kRecvDone})) {
    VLOG(2) << "Not attempting to unroll " << while_op->name()
            << " because it contains a send/recv node: "
            << while_op->ToShortString();
    return std::nullopt;
  }

  if (while_op->operand(0)->opcode() != HloOpcode::kTuple) {
    VLOG(2) << "Not attempting to unroll " << while_op->name()
            << " because the operand is not a tuple: "
            << while_op->ToShortString();
    return std::nullopt;
  }

  // We cannot unroll loops that have side effecting condition because the
  // condition will be removed after unrolling. This might be relaxed
  // later when we add partial unrolling.
  if (while_op->while_condition()->HasSideEffect()) {
    VLOG(2) << "Not attempting to remove while loop whose condition contains "
               "side-effecting instructions: "
            << while_op->ToShortString();
    return std::nullopt;
  }
  std::optional<int64_t> indvar_tuple_idx =
      GetLoopInductionVarTupleIdx(while_op);
  if (!indvar_tuple_idx.has_value()) {
    VLOG(2) << "Not attempting to unroll because induction variable could not "
               "be found.";
    return std::nullopt;
  }

  HloEvaluator evaluator(/*max_loop_iterations=*/0);
  const HloInstruction* while_init = while_op->operand(0);
  const HloInstruction* indvar_init = while_init->operand(*indvar_tuple_idx);
  absl::StatusOr<Literal> indvar_init_result = evaluator.Evaluate(indvar_init);
  if (!indvar_init_result.ok()) {
    VLOG(2) << "Couldn't evaluate induction variable init, "
            << indvar_init_result.status() << ", " << indvar_init->ToString();
    return std::nullopt;
  }
  Literal indvar_iter_val = std::move(indvar_init_result).value();
  std::optional<int64_t> trip_count =
      MatchTrivialLoopTripCount(while_op, *indvar_tuple_idx, indvar_iter_val);
  if (!trip_count.has_value()) {
    VLOG(3) << "Loop doesn't have trivial trip count";
    return std::nullopt;
  }

  VLOG(3) << "Loop trip count " << trip_count.value();

  WhileLoopConfig config;
  config.while_instr = while_op;
  config.init =
      LiteralUtil::LiteralAsScalarInt64(std::move(indvar_iter_val)).value();
  config.trip_count = trip_count.value();
  config.induction_var_idx = *indvar_tuple_idx;
  return config;
}

/*static*/ absl::StatusOr<bool> WhileLoopUnroller::PrepareModuleForUnrolling(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  TF_ASSIGN_OR_RETURN(
      bool applied_cse,
      HloCSE(/*is_layout_sensitive=*/true, /*only_fusion_computations=*/false,
             /*ignore_control_dependencies=*/false, /*only_scalars=*/true)
          .Run(module, execution_threads));
  if (applied_cse) {
    changed = true;
    VLOG(3) << "Applied hlo cse to module " << module->name();
  }
  TF_ASSIGN_OR_RETURN(bool applied_tuple_simplifier,
                      TupleSimplifier{}.Run(module, execution_threads));
  if (applied_tuple_simplifier) {
    changed = true;
    VLOG(3) << "Applied tuple simplifier to module " << module->name();
  }

  // We apply constant sinking to fix point.
  HloPassFix<WhileLoopConstantSinking> constant_sinking(
      /*sink_broadcast_of_constants=*/true,
      /*sink_only_scalar_constants=*/true);
  TF_ASSIGN_OR_RETURN(bool applied_constant_sinking,
                      constant_sinking.Run(module, execution_threads));
  if (applied_constant_sinking) {
    changed = true;
    VLOG(3) << "Applied constant sinking to module " << module->name();
  }
  return changed;
}

/*static*/ std::vector<std::pair<HloInstruction*, WhileLoopConfig>>
WhileLoopUnroller::GetUnrollableLoops(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    std::optional<UnrollConfig> unroll_config) {
  // Processing the while loops in the reverse topological order. If the body
  // of while loop A calls while loop B, B comes before A.
  std::vector<HloInstruction*> all_while_ops;
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(all_while_ops),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  std::vector<std::pair<HloInstruction*, WhileLoopConfig>> while_loop_configs;
  for (HloInstruction* instr : all_while_ops) {
    std::optional<WhileLoopConfig> config = IsLoopUnrollable(instr);
    if (!config.has_value()) {
      continue;
    }
    if (unroll_config.has_value() &&
        !InitialFeasibilityCheck(instr, config.value(),
                                 unroll_config.value())) {
      VLOG(3) << "Initial feasibility check failed for " << instr->name();
      continue;
    }
    while_loop_configs.emplace_back(instr, config.value());
  }
  return while_loop_configs;
}

/*static*/ absl::StatusOr<UnrollResult>
WhileLoopUnroller::UnrollAndReturnReplacement(
    HloInstruction* while_op, int64_t unroll_factor, bool wrap_in_trivial_loop,
    bool force_unroll, bool prepare, const UnrollConfig& unroll_config) {
  UnrollResult result;

  HloModule* module = while_op->GetModule();
  // TODO(b/288130138): For now, we only support full unrolling. Will add
  // partial unrolling if needed.
  if (unroll_factor != -1) {
    VLOG(5) << absl::StrCat(
        "Currently, only full unrolling is supported, unroll factor: ",
        unroll_factor);
    return result;
  }

  if (prepare) {
    // Make sure all the necessary passes are executed before unrolling in order
    // to unroll every possible loop.
    TF_RETURN_IF_ERROR(
        PrepareModuleForUnrolling(module, /*execution_threads=*/{}).status());
  }

  // Construct the loop config
  std::optional<WhileLoopConfig> config = IsLoopUnrollable(while_op);
  if (!config.has_value()) {
    VLOG(5) << "Not attempting to unroll " << while_op->name()
            << " because it is not unrollable.";
    return result;
  }

  if (!force_unroll &&
      !InitialFeasibilityCheck(while_op, config.value(), unroll_config)) {
    return result;
  }
  if (wrap_in_trivial_loop) {
    TF_ASSIGN_OR_RETURN(result, UnrollInternalWrappedAndReturnReplacement(
                                    while_op, config.value()));
  } else {
    TF_ASSIGN_OR_RETURN(result.unrolled,
                        UnrollInternal(while_op, config.value()));
  }

  // We need to inline the calls created for unrolling since later passes rely
  // on the calls to be inlined.
  if (result.unrolled) {
    TF_RETURN_IF_ERROR(CallInliner().Run(module).status());
  }

  return result;
}

absl::StatusOr<bool> WhileLoopUnroller::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // TODO(b/288130138) For now, we only support full unrolling. Will add partial
  // unrolling if needed.
  if (unroll_factor_ != -1) {
    return false;
  }
  XLA_VLOG_LINES(3, "WhileLoopUnroller::Run(), before:\n" + module->ToString());
  bool changed = false;
  // Make sure all the necessary passes are executed before unrolling in order
  // to unroll every possible loop.
  TF_ASSIGN_OR_RETURN(changed,
                      PrepareModuleForUnrolling(module, execution_threads));
  // Processing the while loops in the reverse of topological order. If the body
  // of while loop A calls while loop B, B comes before A.
  std::vector<HloInstruction*> all_while_ops;
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    absl::c_copy_if(comp->instructions(), std::back_inserter(all_while_ops),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  // Gather a preliminary vector of all the while ops that we think we can
  // unroll. We do this ahead of time so we don't have to worry about mutating
  // the lists of computations or instructions while we iterate.
  std::vector<std::pair<HloInstruction*, WhileLoopConfig>>
      unrollable_while_ops = GetUnrollableLoops(
          module, execution_threads, /*unroll_config=*/unroll_config_);
  VLOG(3) << "Number of while instructions in the module to unroll: "
          << unrollable_while_ops.size();

  bool unrolled = false;
  for (auto& [while_op, config] : unrollable_while_ops) {
    if (wrap_in_trivial_loop_) {
      TF_ASSIGN_OR_RETURN(unrolled, UnrollInternalWrapped(while_op, config));
    } else {
      TF_ASSIGN_OR_RETURN(unrolled, UnrollInternal(while_op, config));
    }
    changed |= unrolled;
  }

  // We need to inline the calls created for unrolling since later passes rely
  // on the calls to be inlined.
  if (changed) {
    TF_RETURN_IF_ERROR(CallInliner().Run(module, execution_threads).status());
  }

  XLA_VLOG_LINES(3, "WhileLoopUnroller::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
