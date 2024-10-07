/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/while_util.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <stack>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/tuple_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

using absl::StrCat;

static absl::StatusOr<
    std::pair<HloComputation*, CallInliner::InlinedInstructionMap>>
WidenWhileCondition(HloComputation* narrow_condition, const Shape& wide_shape) {
  const Shape& narrow_shape =
      narrow_condition->parameter_instruction(0)->shape();

  HloComputation* wide_while_cond = [&]() {
    HloComputation::Builder builder(StrCat("wide.", narrow_condition->name()));
    builder.AddInstruction(HloInstruction::CreateParameter(
        0, wide_shape,
        absl::StrCat("wide.",
                     narrow_condition->parameter_instruction(0)->name())));

    // This is needed so that the root instruction is shaped as a PRED[] -- we
    // need to get this right to begin with since we can't mutate the type of
    // the root instruction later.  We later change the root instruction to
    // something more appropriate.
    builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(false)));
    return narrow_condition->parent()->AddEmbeddedComputation(builder.Build());
  }();

  HloInstruction* truncated_parameter = TupleUtil::ExtractPrefix(
      wide_while_cond->parameter_instruction(0),
      narrow_shape.tuple_shapes_size(),
      absl::StrCat("renarrowed.",
                   wide_while_cond->parameter_instruction(0)->name()));
  HloInstruction* call_narrow_cond = wide_while_cond->AddInstruction(
      HloInstruction::CreateCall(ShapeUtil::MakeShape(PRED, {}),
                                 {truncated_parameter}, narrow_condition));

  wide_while_cond->set_root_instruction(call_narrow_cond);

  TF_ASSIGN_OR_RETURN(auto inlined_instructions_map,
                      CallInliner::Inline(call_narrow_cond));
  return {{wide_while_cond, std::move(inlined_instructions_map)}};
}

static absl::StatusOr<
    std::pair<HloComputation*, CallInliner::InlinedInstructionMap>>
WidenWhileBody(HloComputation* narrow_body, const Shape& wide_shape) {
  const Shape& narrow_shape = narrow_body->parameter_instruction(0)->shape();

  HloComputation* wide_while_body = [&]() {
    HloComputation::Builder builder(StrCat("wide.", narrow_body->name()));
    builder.AddInstruction(HloInstruction::CreateParameter(
        0, wide_shape,
        absl::StrCat("wide.", narrow_body->parameter_instruction(0)->name())));
    return narrow_body->parent()->AddEmbeddedComputation(builder.Build());
  }();

  HloInstruction* wide_parameter = wide_while_body->parameter_instruction(0);
  HloInstruction* truncated_parameter = TupleUtil::ExtractPrefix(
      wide_parameter, narrow_shape.tuple_shapes_size(),
      absl::StrCat("renarrowed.",
                   wide_while_body->parameter_instruction(0)->name()));
  HloInstruction* call_narrow_body =
      wide_while_body->AddInstruction(HloInstruction::CreateCall(
          narrow_shape, {truncated_parameter}, narrow_body));

  std::vector<HloInstruction*> live_through_values;
  for (int i = narrow_shape.tuple_shapes_size();
       i < wide_shape.tuple_shapes_size(); i++) {
    live_through_values.push_back(wide_while_body->AddInstruction(
        HloInstruction::CreateGetTupleElement(wide_shape.tuple_shapes(i),
                                              wide_parameter, i),
        absl::StrCat(wide_while_body->name(), ".through.",
                     i - narrow_shape.tuple_shapes_size())));
  }

  wide_while_body->set_root_instruction(
      TupleUtil::AppendSuffix(call_narrow_body, live_through_values));

  TF_ASSIGN_OR_RETURN(auto inlined_instructions_map,
                      CallInliner::Inline(call_narrow_body));
  return {{wide_while_body, std::move(inlined_instructions_map)}};
}

/*static*/ absl::StatusOr<WhileUtil::MakeInstructionsLiveInResult>
WhileUtil::MakeInstructionsLiveIn(
    HloInstruction* while_instr,
    absl::Span<HloInstruction* const> instructions) {
  CHECK(while_instr->shape().IsTuple());

  int elements_in_old_while_shape = while_instr->shape().tuple_shapes_size();
  Shape new_while_shape = while_instr->shape();
  for (auto* instruction : instructions) {
    *new_while_shape.add_tuple_shapes() = instruction->shape();
  }

  HloComputation* new_while_condition;
  CallInliner::InlinedInstructionMap inlined_condition_instructions_map;
  TF_ASSIGN_OR_RETURN(
      std::tie(new_while_condition, inlined_condition_instructions_map),
      WidenWhileCondition(while_instr->while_condition(), new_while_shape));

  HloComputation* new_while_body;
  CallInliner::InlinedInstructionMap inlined_instructions_map;
  TF_ASSIGN_OR_RETURN(
      std::tie(new_while_body, inlined_instructions_map),
      WidenWhileBody(while_instr->while_body(), new_while_shape));

  HloInstruction* new_while_init =
      TupleUtil::AppendSuffix(while_instr->mutable_operand(0), instructions);
  HloComputation* containing_computation = while_instr->parent();
  HloInstruction* new_while = containing_computation->AddInstruction(
      HloInstruction::CreateWhile(new_while_shape, new_while_condition,
                                  new_while_body, new_while_init));

  // We want to get rid of the old while instruction even if it has side
  // effecting operations so we do a manual HloComputation::RemoveInstruction
  // instead of relying on HloComputation::ReplaceInstruction.
  HloInstruction* replacement_instr = TupleUtil::ExtractPrefix(
      new_while, while_instr->shape().tuple_shapes_size());
  TF_RETURN_IF_ERROR(while_instr->ReplaceAllUsesWith(replacement_instr));
  TF_RETURN_IF_ERROR(containing_computation->RemoveInstruction(while_instr));

  HloInstruction* while_body_param = new_while_body->parameter_instruction(0);
  std::vector<HloInstruction*> live_in_instructions;
  for (int64_t i = elements_in_old_while_shape;
       i < new_while_shape.tuple_shapes_size(); i++) {
    live_in_instructions.push_back(new_while_body->AddInstruction(
        HloInstruction::CreateGetTupleElement(
            instructions[i - elements_in_old_while_shape]->shape(),
            while_body_param, i),
        absl::StrCat(new_while_body->name(), ".in.",
                     i - elements_in_old_while_shape)));
  }

  WhileUtil::MakeInstructionsLiveInResult result;

  result.new_while_instr = new_while;
  result.replacement_instr = replacement_instr;
  result.while_body_live_in_values = std::move(live_in_instructions);
  result.while_body_instruction_map = std::move(inlined_instructions_map);
  result.while_condition_instruction_map =
      std::move(inlined_condition_instructions_map);

  return std::move(result);
}

static absl::StatusOr<std::unique_ptr<HloComputation>>
MakeCountedLoopConditionComputation(const Shape& loop_state_shape,
                                    int32_t trip_count) {
  Shape scalar_pred = ShapeUtil::MakeShape(PRED, {});

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloComputation> cond_computation,
                      CreateComputationWithSignature(
                          {&loop_state_shape}, scalar_pred, "while_cond"));

  HloInstruction* trip_count_constant =
      cond_computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32_t>(trip_count)));

  HloInstruction* param = cond_computation->parameter_instruction(0);
  TF_ASSIGN_OR_RETURN(HloInstruction * indvar,
                      MakeGetTupleElementHlo(param, 0));

  TF_ASSIGN_OR_RETURN(
      HloInstruction * compare,
      MakeCompareHlo(ComparisonDirection::kLt, indvar, trip_count_constant));
  cond_computation->set_root_instruction(compare);
  return std::move(cond_computation);
}

static absl::StatusOr<std::unique_ptr<HloComputation>>
MakeCountedLoopBodyComputation(
    const Shape& loop_state_shape,
    absl::FunctionRef<absl::StatusOr<WhileUtil::LoopStateTy>(
        HloInstruction*, const WhileUtil::LoopStateTy&)>
        loop_body_generator) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloComputation> body_computation,
                      CreateComputationWithSignature(
                          {&loop_state_shape}, loop_state_shape, "while_body"));
  HloInstruction* one = body_computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));
  HloInstruction* param = body_computation->parameter_instruction(0);
  TF_ASSIGN_OR_RETURN(HloInstruction * indvar,
                      MakeGetTupleElementHlo(param, 0));
  TF_ASSIGN_OR_RETURN(HloInstruction * next_indvar,
                      MakeBinaryHlo(HloOpcode::kAdd, indvar, one));

  std::vector<HloInstruction*> loop_body_generator_args;
  for (int i = 1, e = loop_state_shape.tuple_shapes_size(); i < e; i++) {
    TF_ASSIGN_OR_RETURN(HloInstruction * tuple_element,
                        MakeGetTupleElementHlo(param, i));
    loop_body_generator_args.push_back(tuple_element);
  }
  TF_ASSIGN_OR_RETURN(std::vector<HloInstruction*> next_state,
                      loop_body_generator(indvar, loop_body_generator_args));
  next_state.insert(next_state.begin(), next_indvar);
  HloInstruction* next_state_tuple =
      body_computation->AddInstruction(HloInstruction::CreateTuple(next_state));
  body_computation->set_root_instruction(next_state_tuple);

  return std::move(body_computation);
}

static std::pair<std::unique_ptr<HloInstruction>,
                 std::unique_ptr<HloInstruction>>
MakeInitTupleFromInitValues(const WhileUtil::LoopStateTy& init_values) {
  std::vector<HloInstruction*> init_values_with_indvar;
  init_values_with_indvar.reserve(init_values.size() + 1);
  std::unique_ptr<HloInstruction> zero =
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0));
  init_values_with_indvar.push_back(zero.get());
  absl::c_copy(init_values, std::back_inserter(init_values_with_indvar));
  return std::make_pair(std::move(zero),
                        HloInstruction::CreateTuple(init_values_with_indvar));
}

// Returns a tuple shape containing a S32, and a shape from each value in
// `init_values`. If a shape from a value in `init_values` doesn't have a
// layout, use a default layout for the shape.
static Shape MakeLoopStateShapeWithLayout(
    const WhileUtil::LoopStateTy& init_values) {
  std::vector<Shape> loop_state_shape_components;
  loop_state_shape_components.reserve(init_values.size() + 1);
  loop_state_shape_components.push_back(ShapeUtil::MakeShape(S32, {}));
  absl::c_transform(init_values,
                    std::back_inserter(loop_state_shape_components),
                    [](HloInstruction* instr) {
                      Shape shape = instr->shape();
                      if (!shape.has_layout()) {
                        LayoutUtil::SetToDefaultLayout(&shape);
                      }
                      return shape;
                    });
  return ShapeUtil::MakeTupleShape(loop_state_shape_components);
}

/*static*/ absl::StatusOr<WhileUtil::OwningLoopStateTy>
WhileUtil::MakeCountedLoop(HloModule* module, int32_t trip_count,
                           const WhileUtil::LoopStateTy& init_values,
                           WhileUtil::LoopBodyGeneratorTy loop_body_generator,
                           const OpMetadata& metadata) {
  CHECK_GE(trip_count, 0);

  // Both MakeCountedLoopConditionComputation and MakeCountedLoopBodyComputation
  // use loop_state_shape to create a literal, which requires loop_state_shape
  // to have a layout.
  Shape loop_state_shape = MakeLoopStateShapeWithLayout(init_values);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloComputation> cond,
      MakeCountedLoopConditionComputation(loop_state_shape, trip_count));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloComputation> body,
      MakeCountedLoopBodyComputation(loop_state_shape, loop_body_generator));
  std::unique_ptr<HloInstruction> owned_indvar;
  std::unique_ptr<HloInstruction> owned_init_tuple;
  std::tie(owned_indvar, owned_init_tuple) =
      MakeInitTupleFromInitValues(init_values);
  std::unique_ptr<HloInstruction> owned_while = HloInstruction::CreateWhile(
      loop_state_shape, module->AddEmbeddedComputation(std::move(cond)),
      module->AddEmbeddedComputation(std::move(body)), owned_init_tuple.get());
  owned_while->set_metadata(metadata);
  HloInstruction* while_instr = owned_while.get();

  std::vector<std::unique_ptr<HloInstruction>> owned;
  owned.push_back(std::move(owned_indvar));
  owned.push_back(std::move(owned_init_tuple));
  owned.push_back(std::move(owned_while));
  std::vector<HloInstruction*> while_results;
  for (int64_t i = 0, e = init_values.size(); i < e; i++) {
    std::unique_ptr<HloInstruction> user_state =
        HloInstruction::CreateGetTupleElement(init_values[i]->shape(),
                                              while_instr, i + 1);
    while_results.push_back(user_state.get());
    owned.push_back(std::move(user_state));
  }
  return WhileUtil::OwningLoopStateTy{std::move(owned), while_results};
}

/*static*/ absl::StatusOr<WhileUtil::LoopStateTy> WhileUtil::MakeCountedLoop(
    HloComputation* computation, int32_t trip_count,
    const WhileUtil::LoopStateTy& init_values,
    WhileUtil::LoopBodyGeneratorTy loop_body_generator,
    const OpMetadata& metadata) {
  TF_ASSIGN_OR_RETURN(
      auto owning_loop_state,
      MakeCountedLoop(computation->parent(), trip_count, init_values,
                      loop_body_generator, metadata));
  for (auto& instruction_to_add : owning_loop_state.instructions_to_add) {
    computation->AddInstruction(std::move(instruction_to_add));
  }
  return owning_loop_state.while_results;
}

/*static*/ std::vector<HloInstruction*> WhileUtil::GetInvariantGTEsForWhileBody(
    const HloComputation& while_body) {
  std::vector<HloInstruction*> result;
  const HloInstruction::InstructionVector root_operands =
      while_body.root_instruction()->operands();
  for (int i = 0; i < root_operands.size(); i++) {
    HloInstruction* instr = root_operands[i];
    if (instr->opcode() == HloOpcode::kGetTupleElement &&
        instr->tuple_index() == i &&
        instr->operand(0) == while_body.parameter_instruction(0)) {
      result.push_back(instr);
    }
  }
  return result;
}

/*static*/ absl::flat_hash_map<int64_t, absl::InlinedVector<HloInstruction*, 1>>
WhileUtil::GetGTEsMapForWhileConditional(
    const HloComputation& while_conditional) {
  absl::flat_hash_map<int64_t, absl::InlinedVector<HloInstruction*, 1>> result;
  for (HloInstruction* user :
       while_conditional.parameter_instruction(0)->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement) {
      result[user->tuple_index()].push_back(user);
    }
  }
  return result;
}

/*static*/
absl::Status WhileUtil::IncrementWhileLoopTripCount(
    HloComputation* while_conditional, int32_t increment) {
  HloInstruction* compare = while_conditional->root_instruction();
  if (compare->opcode() != HloOpcode::kCompare) {
    return absl::InvalidArgumentError("While condition root is not a compare");
  }
  HloInstruction* trip_count;
  if ((compare->comparison_direction() == ComparisonDirection::kGt) ||
      (compare->comparison_direction() == ComparisonDirection::kGe)) {
    trip_count = compare->mutable_operand(0);
  } else if ((compare->comparison_direction() == ComparisonDirection::kLt) ||
             (compare->comparison_direction() == ComparisonDirection::kLe)) {
    trip_count = compare->mutable_operand(1);
  } else {
    return absl::InvalidArgumentError("Unhandled comparison direction");
  }
  if (trip_count->user_count() > 1) {
    return absl::InvalidArgumentError(
        "Not able to handle trip count with multiple users");
  }
  HloInstruction* increment_constant =
      while_conditional->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32_t>(increment)));
  HloInstruction* incremented_trip_count = while_conditional->AddInstruction(
      HloInstruction::CreateBinary(trip_count->shape(), HloOpcode::kAdd,
                                   trip_count, increment_constant));
  return trip_count->ReplaceAllUsesWith(incremented_trip_count);
}

/*static*/
int64_t WhileUtil::ComputeWhileLoopPipelineDepth(
    const HloComputation& while_body) {
  CHECK(while_body.IsWhileBodyComputation());

  // Look for pattern param -> gte -> root, where indices in the param and
  // root tuples are missmatching.
  absl::flat_hash_map<int64_t, int64_t> loop_permutations;
  HloInstruction* while_param = while_body.parameter_instruction(0);
  HloInstruction* while_root = while_body.root_instruction();
  for (int64_t output_index = 0; output_index < while_root->operand_count();
       output_index++) {
    const HloInstruction* operand = while_root->operand(output_index);
    if (operand->opcode() == HloOpcode::kGetTupleElement &&
        operand->operand(0) == while_param) {
      int64_t input_index = operand->tuple_index();
      if (input_index != output_index) {
        // Don't try to analyze loops with complicated permutation patterns.
        if (loop_permutations.contains(input_index)) {
          return 1;
        }
        loop_permutations.emplace(input_index, output_index);
      }
    }
  }

  // Find all indices at which the pipelined chains start from.
  std::vector<int64_t> start_indices;
  for (auto&& [input_index, _] : loop_permutations) {
    bool is_chain_start = true;
    for (auto&& [_, output_index] : loop_permutations) {
      if (input_index == output_index) {
        is_chain_start = false;
      }
    }
    if (is_chain_start) {
      start_indices.push_back(input_index);
    }
  }

  // Find all pipelining chains.
  std::vector<std::vector<int64_t>> pipelined_chains;
  for (int64_t start_index : start_indices) {
    std::stack<std::pair<int64_t, std::vector<int64_t>>> stack;
    stack.push({start_index, {start_index}});
    while (!stack.empty()) {
      auto [current_index, current_chain] = stack.top();
      stack.pop();
      if (!loop_permutations.contains(current_index)) {
        pipelined_chains.push_back(std::move(current_chain));
      } else {
        int64_t next_index = loop_permutations[current_index];
        current_chain.push_back(next_index);
        stack.emplace(next_index, std::move(current_chain));
      }
    }
  }

  // Compute the pipeline depth of the loop body.
  // https://en.wikipedia.org/wiki/Permutation#Order_of_a_permutation
  int64_t pipeline_depth = 1;
  for (auto&& pipelined_chain : pipelined_chains) {
    pipeline_depth =
        std::lcm<int64_t>(pipelined_chain.size() + 1, pipeline_depth);
  }

  return pipeline_depth;
}

}  // namespace xla
