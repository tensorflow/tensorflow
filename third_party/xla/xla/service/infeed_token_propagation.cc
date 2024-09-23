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

#include "xla/service/infeed_token_propagation.h"

#include <cstdint>
#include <string_view>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {
bool IsDanglingInfeed(HloInstruction* infeed) {
  CHECK(infeed->opcode() == HloOpcode::kInfeed);
  if (infeed->has_sharding()) {
    // TODO: b/368327832 - Skip handling sharding until it is removed.
    return false;
  }

  // Check for dangling input token.
  if (const HloInstruction* after_all = infeed->operand(0);
      after_all->opcode() != HloOpcode::kAfterAll ||
      after_all->operand_count() != 0) {
    return false;
  }

  // Check for dangling output token.
  for (const HloInstruction* user : infeed->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == 1) {
      return false;
    }
  }

  return true;
}

bool IsDanglingOutfeed(HloInstruction* outfeed) {
  CHECK(outfeed->opcode() == HloOpcode::kOutfeed);
  if (outfeed->has_sharding()) {
    // TODO: b/368327832 - Skip handling sharding until it is removed.
    return false;
  }

  // Check for dangling input token.
  if (const HloInstruction* after_all = outfeed->operand(1);
      after_all->opcode() != HloOpcode::kAfterAll ||
      after_all->operand_count() != 0) {
    return false;
  }

  // Check for dangling output token.
  if (outfeed->user_count() != 0) {
    return false;
  }

  return true;
}

HloInstruction* ReconstructTuple(HloInstruction* tuple) {
  CHECK(tuple->shape().IsTuple());
  HloComputation* computation = tuple->parent();

  std::vector<HloInstruction*> gtes;
  gtes.resize(tuple->shape().tuple_shapes_size());
  for (int64_t idx = 0; idx < gtes.size(); ++idx) {
    gtes[idx] = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(tuple, idx));
  }

  return computation->AddInstruction(HloInstruction::CreateTuple(gtes));
}

absl::StatusOr<HloInstruction*> InsertTokenIntoTuple(HloInstruction* tuple,
                                                     bool add_token_operand) {
  CHECK(tuple->shape().IsTuple());
  HloComputation* computation = tuple->parent();

  // Recreate the original tuple, we'll need to pass this to all the users.
  std::vector<HloInstruction*> original_users = tuple->users();
  HloInstruction* original_tuple = ReconstructTuple(tuple);
  for (HloInstruction* original_user : original_users) {
    int64_t idx = original_user->operand_index(tuple);
    TF_RETURN_IF_ERROR(original_user->ReplaceOperandWith(idx, original_tuple));
  }

  // Append the token to the parameter tuple.
  *tuple->mutable_shape()->add_tuple_shapes() = ShapeUtil::MakeTokenShape();
  if (add_token_operand) {
    tuple->AppendOperand(
        computation->AddInstruction(HloInstruction::CreateToken()));
  }

  HloInstruction* input_token_gte =
      computation->AddInstruction(HloInstruction::CreateGetTupleElement(
          tuple, tuple->shape().tuple_shapes_size() - 1));
  return input_token_gte;
}

absl::Status CanonicalizeConditionalBranch(HloComputation* branch) {
  CHECK(branch->IsConditionalBranchComputation());
  CHECK_EQ(branch->num_parameters(), 1);

  // Tuplify the branch parameter if needed.
  HloInstruction* parameter = branch->parameter_instruction(0);
  if (!parameter->shape().IsTuple()) {
    *parameter->mutable_shape() =
        ShapeUtil::MakeTupleShape({parameter->shape()});
    HloInstruction* original = branch->AddInstruction(
        HloInstruction::CreateGetTupleElement(parameter, 0));
    TF_RETURN_IF_ERROR(parameter->ReplaceAllUsesWithDifferentShape(original));
  }

  // Tuplify the branch tuple if needed.
  HloInstruction* conditional = branch->ConditionalCallInstruction();
  int64_t branch_operand_idx = conditional->branch_index(branch) + 1;
  HloInstruction* branch_tuple =
      conditional->mutable_operand(branch_operand_idx);
  if (!branch_tuple->shape().IsTuple()) {
    branch_tuple = conditional->parent()->AddInstruction(
        HloInstruction::CreateTuple({branch_tuple}));
    TF_RETURN_IF_ERROR(conditional->ReplaceOperandWithDifferentShape(
        branch_operand_idx, branch_tuple));
  }

  // Explicitly disjoin computation parameters from branch inputs, so we can
  // insert tokens into the input tuple.
  if (branch_tuple->opcode() == HloOpcode::kParameter) {
    branch_tuple = ReconstructTuple(branch_tuple);
    TF_RETURN_IF_ERROR(
        conditional->ReplaceOperandWith(branch_operand_idx, branch_tuple));
  }

  // If the computation root is a also a computation parameter, explicitly split
  // them, as the input and output tokens cannot be part of the same
  // instruction.
  HloInstruction* root = branch->root_instruction();
  if (root->opcode() == HloOpcode::kParameter) {
    root = ReconstructTuple(root);
    branch->set_root_instruction(root);
  }

  // ConditionalCanonicalizer should have already turned the conditional output
  // to be a tuple.
  CHECK(conditional->shape().IsTuple());
  return absl::OkStatus();
}

absl::Status CanonicalizeWhileBody(HloComputation* body) {
  CHECK(body->IsWhileBodyComputation());
  CHECK_EQ(body->num_parameters(), 1);

  // Tuplify the body parameter if needed.
  HloInstruction* parameter = body->parameter_instruction(0);
  if (!parameter->shape().IsTuple()) {
    *parameter->mutable_shape() =
        ShapeUtil::MakeTupleShape({parameter->shape()});
    HloInstruction* original = body->AddInstruction(
        HloInstruction::CreateGetTupleElement(parameter, 0));
    TF_RETURN_IF_ERROR(parameter->ReplaceAllUsesWithDifferentShape(original));
  }

  // Tuplify the body root if needed.
  HloInstruction* root = body->root_instruction();
  if (!root->shape().IsTuple()) {
    root = body->AddInstruction(HloInstruction::CreateTuple({root}));
    body->set_root_instruction(root, /*accept_different_shape=*/true);
  }

  // Tuplify the condition parameter if needed.
  HloInstruction* loop = body->WhileCallInstruction();
  HloComputation* cond = loop->while_condition();
  HloInstruction* cond_parameter = cond->parameter_instruction(0);
  if (!cond_parameter->shape().IsTuple()) {
    *cond_parameter->mutable_shape() =
        ShapeUtil::MakeTupleShape({cond_parameter->shape()});
    HloInstruction* original = cond->AddInstruction(
        HloInstruction::CreateGetTupleElement(cond_parameter, 0));
    TF_RETURN_IF_ERROR(
        cond_parameter->ReplaceAllUsesWithDifferentShape(original));
  }

  // Tuplify the while instruction if needed.
  if (!loop->shape().IsTuple()) {
    *loop->mutable_shape() = ShapeUtil::MakeTupleShape({loop->shape()});
    HloInstruction* original = loop->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(loop, 0));
    TF_RETURN_IF_ERROR(loop->ReplaceAllUsesWithDifferentShape(original));
  }

  // Tuplify the while tuple if needed.
  HloInstruction* loop_tuple = loop->mutable_operand(0);
  if (!loop_tuple->shape().IsTuple()) {
    loop_tuple = loop->parent()->AddInstruction(
        HloInstruction::CreateTuple({loop_tuple}));
    TF_RETURN_IF_ERROR(loop->ReplaceOperandWithDifferentShape(0, loop_tuple));
  }

  // Explicitly disjoin computation parameters from loop inputs, so we can
  // insert tokens into the input tuple.
  if (loop_tuple->opcode() == HloOpcode::kParameter) {
    loop_tuple = ReconstructTuple(loop_tuple);
    TF_RETURN_IF_ERROR(loop->ReplaceOperandWith(0, loop_tuple));
  }

  // If the computation root is a also a computation parameter, explicitly
  // split them, as the input and output tokens cannot be part of the same
  // instruction.
  if (root->opcode() == HloOpcode::kParameter) {
    root = ReconstructTuple(root);
    body->set_root_instruction(root);
  }

  return absl::OkStatus();
}

absl::StatusOr<std::tuple<HloInstruction*, HloInstruction*, HloInstruction*>>
PropagateTokenThroughConditionalBranch(HloInstruction* instruction,
                                       HloInstruction* input_token,
                                       HloInstruction* output_token) {
  // Conditional branches can diverge in inputs, but must converge on outputs.

  // Fixup the branch.
  HloComputation* comp = instruction->parent();
  TF_RETURN_IF_ERROR(CanonicalizeConditionalBranch(comp));
  HloInstruction* next_instruction = comp->ConditionalCallInstruction();

  // Insert the output token into each branch.
  for (HloComputation* branch : next_instruction->branch_computations()) {
    HloInstruction* root = branch->root_instruction();
    if (branch == comp) {
      TF_RETURN_IF_ERROR(
          InsertTokenIntoTuple(root, /*add_token_operand=*/false).status());
      root->AppendOperand(output_token);
    } else {
      TF_RETURN_IF_ERROR(
          InsertTokenIntoTuple(root, /*add_token_operand=*/true).status());
    }
  }

  // Insert the input token into the branch parameter.
  HloInstruction* parameter = comp->parameter_instruction(0);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * input_token_gte,
      InsertTokenIntoTuple(parameter, /*add_token_operand=*/false));
  TF_RETURN_IF_ERROR(input_token->ReplaceAllUsesWith(input_token_gte));

  // Insert the input token into the branch tuple.
  int64_t branch_operand_idx = next_instruction->branch_index(comp) + 1;
  HloInstruction* branch_tuple =
      next_instruction->mutable_operand(branch_operand_idx);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * next_input_token_gte,
      InsertTokenIntoTuple(branch_tuple, /*add_token_operand=*/true));
  TF_RETURN_IF_ERROR(next_instruction->ReplaceOperandWithDifferentShape(
      branch_operand_idx, branch_tuple));
  HloInstruction* next_input_token =
      branch_tuple->mutable_operand(next_input_token_gte->tuple_index());

  // Insert the output token into conditional instruction.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * next_output_token,
      InsertTokenIntoTuple(next_instruction, /*add_token_operand=*/false));

  return std::make_tuple(next_instruction, next_input_token, next_output_token);
}

absl::StatusOr<std::tuple<HloInstruction*, HloInstruction*, HloInstruction*>>
PropagateTokenThroughWhileBody(HloInstruction* instruction,
                               HloInstruction* input_token,
                               HloInstruction* output_token) {
  // While loops need to converge on input and output.

  // Fixup the while body.
  HloComputation* comp = instruction->parent();
  TF_RETURN_IF_ERROR(CanonicalizeWhileBody(comp));
  HloInstruction* next_instruction = comp->WhileCallInstruction();

  // Insert the output token into the body root.
  HloInstruction* root = comp->root_instruction();
  TF_RETURN_IF_ERROR(
      InsertTokenIntoTuple(root, /*add_token_operand=*/false).status());
  root->AppendOperand(output_token);

  // Insert the input token into the body parameter.
  HloInstruction* body_parameter = comp->parameter_instruction(0);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * input_token_gte,
      InsertTokenIntoTuple(body_parameter, /*add_token_operand=*/false));
  TF_RETURN_IF_ERROR(input_token->ReplaceAllUsesWith(input_token_gte));

  // Insert the input token into the condition parameter.
  HloComputation* cond = next_instruction->while_condition();
  HloInstruction* cond_parameter = cond->parameter_instruction(0);
  TF_RETURN_IF_ERROR(
      InsertTokenIntoTuple(cond_parameter, /*add_token_operand=*/false)
          .status());

  // Insert the input token into the while tuple.
  HloInstruction* while_tuple = next_instruction->mutable_operand(0);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * next_input_token,
      InsertTokenIntoTuple(while_tuple, /*add_token_operand=*/true));
  TF_RETURN_IF_ERROR(
      next_instruction->ReplaceOperandWithDifferentShape(0, while_tuple));

  // Insert the input token into the while instruction.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * next_output_token,
      InsertTokenIntoTuple(next_instruction, /*add_token_operand=*/false));

  return std::make_tuple(next_instruction, next_input_token, next_output_token);
}

absl::Status PropagateToken(HloInstruction* instruction,
                            HloInstruction* input_token,
                            HloInstruction* output_token) {
  HloComputation* comp = instruction->parent();
  if (comp->IsEntryComputation()) {
    // If we propagate through the root instruction, reconstruct the original
    // tuple and set that to be root.
    if (instruction->IsRoot() &&
        (instruction->opcode() == HloOpcode::kWhile ||
         instruction->opcode() == HloOpcode::kConditional)) {
      std::vector<HloInstruction*> gtes;
      int64_t output_token_idx = output_token->tuple_index();
      for (int64_t idx = 0; idx < instruction->shape().tuple_shapes_size();
           idx++) {
        if (idx != output_token_idx) {
          gtes.push_back(comp->AddInstruction(
              HloInstruction::CreateGetTupleElement(instruction, idx)));
        }
      }
      HloInstruction* original_tuple =
          comp->AddInstruction(HloInstruction::CreateTuple(gtes));
      comp->set_root_instruction(original_tuple,
                                 /*accept_different_shape=*/true);
    }
    return absl::OkStatus();
  }

  HloInstruction* next_instruction = nullptr;
  HloInstruction* next_input_token = nullptr;
  HloInstruction* next_output_token = nullptr;
  if (comp->IsConditionalBranchComputation()) {
    // TODO: b/368327832 - Skip handling sharding until it is removed.
    if (comp->ConditionalCallInstruction()->has_sharding()) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(
        std::tie(next_instruction, next_input_token, next_output_token),
        PropagateTokenThroughConditionalBranch(instruction, input_token,
                                               output_token));
  } else if (comp->IsWhileBodyComputation()) {
    // TODO: b/368327832 - Skip handling sharding until it is removed.
    if (comp->WhileCallInstruction()->has_sharding()) {
      return absl::OkStatus();
    }
    TF_ASSIGN_OR_RETURN(
        std::tie(next_instruction, next_input_token, next_output_token),
        PropagateTokenThroughWhileBody(instruction, input_token, output_token));
  } else {
    // We only expect to encounter computations behind while and conditional
    // instructions. In the case of it being behind a while condition, there is
    // no way to propagate the output token, as the root only returns a
    // predicate. All other computations that could possibly contain infeed
    // or outfeed ops should have already been inlined.
    VLOG(2) << "Unhandled computation: " << comp->name();
    return absl::OkStatus();
  }
  CHECK_NE(next_instruction, nullptr);
  CHECK_NE(next_input_token, nullptr);
  CHECK_NE(next_output_token, nullptr);

  return PropagateToken(next_instruction, next_input_token, next_output_token);
}
}  // namespace

absl::StatusOr<bool> InfeedTokenPropagation::Run(
    HloModule* module,
    const absl::flat_hash_set<std::string_view>& execution_threads) {
  VLOG(5) << "Before InfeedTokenPropagation:";
  XLA_VLOG_LINES(5, module->ToString());

  std::vector<HloInstruction*> dangling_infeeds;
  std::vector<HloInstruction*> dangling_outfeeds;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (!computation->IsEntryComputation()) {
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kInfeed &&
            IsDanglingInfeed(instruction)) {
          VLOG(1) << "Found dangling infeed: " << instruction->ToString();
          dangling_infeeds.push_back(instruction);
        } else if (instruction->opcode() == HloOpcode::kOutfeed &&
                   IsDanglingOutfeed(instruction)) {
          VLOG(1) << "Found dangling outfeed: " << instruction->ToString();
          dangling_outfeeds.push_back(instruction);
        }
      }
    }
  }

  for (HloInstruction* dangling_infeed : dangling_infeeds) {
    HloInstruction* input_token = dangling_infeed->mutable_operand(0);
    HloInstruction* output_token = dangling_infeed->AddInstruction(
        HloInstruction::CreateGetTupleElement(dangling_infeed, 1));
    TF_RETURN_IF_ERROR(
        PropagateToken(dangling_infeed, input_token, output_token));
  }
  for (HloInstruction* dangling_outfeed : dangling_outfeeds) {
    HloInstruction* input_token = dangling_outfeed->mutable_operand(1);
    HloInstruction* output_token = dangling_outfeed;
    TF_RETURN_IF_ERROR(
        PropagateToken(dangling_outfeed, input_token, output_token));
  }

  bool changed = !dangling_infeeds.empty() || !dangling_outfeeds.empty();
  if (changed) {
    TF_RETURN_IF_ERROR(
        TupleSimplifier().Run(module, execution_threads).status());
    TF_RETURN_IF_ERROR(HloDCE().Run(module, execution_threads).status());
  }

  VLOG(5) << "After InfeedTokenPropagation:";
  XLA_VLOG_LINES(5, module->ToString());
  return changed;
}
}  // namespace xla
