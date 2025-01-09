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

#include "xla/hlo/transforms/collectives/infeed_token_propagation.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/hlo/transforms/simplifiers/tuple_simplifier.h"
#include "xla/service/call_graph.h"
#include "xla/service/tuple_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {
HloInstruction* InfeedToken(HloInstruction* infeed) {
  CHECK_EQ(infeed->opcode(), HloOpcode::kInfeed);
  for (HloInstruction* user : infeed->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == 1) {
      return user;
    }
  }
  return nullptr;
}

HloInstruction* InfeedChainBegin(HloInstruction* infeed) {
  CHECK_EQ(infeed->opcode(), HloOpcode::kInfeed);
  HloInstruction* begin = infeed;
  while (begin->operand(0)->opcode() == HloOpcode::kGetTupleElement &&
         begin->operand(0)->operand(0)->opcode() == HloOpcode::kInfeed) {
    begin = begin->mutable_operand(0)->mutable_operand(0);
  }
  return begin;
}

HloInstruction* InfeedChainEnd(HloInstruction* infeed) {
  CHECK_EQ(infeed->opcode(), HloOpcode::kInfeed);
  HloInstruction* end = infeed;
  HloInstruction* token = InfeedToken(end);
  while (token != nullptr && token->user_count() == 1) {
    if (token->users()[0]->opcode() == HloOpcode::kInfeed) {
      end = token->users()[0];
      token = InfeedToken(end);
    } else {
      break;
    }
  }
  return end;
}

HloInstruction* OutfeedChainBegin(HloInstruction* outfeed) {
  CHECK_EQ(outfeed->opcode(), HloOpcode::kOutfeed);
  HloInstruction* begin = outfeed;
  while (begin->operand(1)->opcode() == HloOpcode::kOutfeed) {
    begin = begin->mutable_operand(1);
  }
  return begin;
}

HloInstruction* OutfeedChainEnd(HloInstruction* outfeed) {
  CHECK_EQ(outfeed->opcode(), HloOpcode::kOutfeed);
  HloInstruction* end = outfeed;
  while (end->user_count() == 1 &&
         end->users()[0]->opcode() == HloOpcode::kOutfeed) {
    end = end->users()[0];
  }
  return end;
}

HloInstruction* ChainBegin(HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kInfeed) {
    return InfeedChainBegin(instruction);
  } else if (instruction->opcode() == HloOpcode::kOutfeed) {
    return OutfeedChainBegin(instruction);
  } else {
    LOG(FATAL) << "Unexpected opcode";
  }
  return nullptr;
}

HloInstruction* ChainEnd(HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kInfeed) {
    return InfeedChainEnd(instruction);
  } else if (instruction->opcode() == HloOpcode::kOutfeed) {
    return OutfeedChainEnd(instruction);
  } else {
    LOG(FATAL) << "Unexpected opcode";
  }
  return nullptr;
}

bool IsDanglingInfeed(HloInstruction* infeed) {
  CHECK(infeed->opcode() == HloOpcode::kInfeed);
  if (infeed->has_sharding()) {
    // TODO: b/368327832 - Skip handling sharding until it is removed.
    return false;
  }

  // Check for dangling input token.
  if (const HloInstruction* after_all = ChainBegin(infeed)->operand(0);
      after_all->opcode() != HloOpcode::kAfterAll ||
      after_all->operand_count() != 0) {
    return false;
  }

  // Check for dangling output token.
  for (const HloInstruction* user : ChainEnd(infeed)->users()) {
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
  if (const HloInstruction* after_all = OutfeedChainBegin(outfeed)->operand(1);
      after_all->opcode() != HloOpcode::kAfterAll ||
      after_all->operand_count() != 0) {
    return false;
  }

  // Check for dangling output token.
  if (OutfeedChainEnd(outfeed)->user_count() != 0) {
    return false;
  }

  return true;
}

absl::StatusOr<HloInstruction*> InsertTokenIntoTuple(HloInstruction* tuple,
                                                     bool add_token_operand) {
  CHECK(tuple->shape().IsTuple());
  HloComputation* computation = tuple->parent();

  // Recreate the original tuple, we'll need to pass this to all the users.
  // Trying to use tuple->ReplaceAllUsesWith(original_tuple) cause a cycle.
  std::vector<HloInstruction*> original_users = tuple->users();
  HloInstruction* original_tuple = TupleUtil::Duplicate(tuple);
  for (HloInstruction* original_user : original_users) {
    for (int64_t idx : original_user->operand_indices(tuple)) {
      TF_RETURN_IF_ERROR(
          original_user->ReplaceOperandWith(idx, original_tuple));
    }
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
}  // namespace

absl::Status CanonicalizeConditionalInstruction(HloInstruction* conditional) {
  CHECK_EQ(conditional->opcode(), HloOpcode::kConditional);

  for (HloComputation* branch : conditional->branch_computations()) {
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
      branch_tuple = TupleUtil::Duplicate(branch_tuple);
      TF_RETURN_IF_ERROR(
          conditional->ReplaceOperandWith(branch_operand_idx, branch_tuple));
    }

    // Explicitly make the root of the branch a tuple.
    HloInstruction* root = branch->root_instruction();
    if (root->opcode() != HloOpcode::kTuple) {
      root = TupleUtil::Duplicate(root);
      branch->set_root_instruction(root);
    }
  }

  // ConditionalCanonicalizer should have already turned the conditional output
  // to be a tuple.
  CHECK(conditional->shape().IsTuple());

  // Explicitly disjoin the conditional from being a computation root, so that
  // we can insert tokens into, while preserving the original computation shape.
  if (conditional->IsRoot()) {
    HloInstruction* new_root = TupleUtil::Duplicate(conditional);
    conditional->parent()->set_root_instruction(new_root);
  }

  return absl::OkStatus();
}

absl::Status CanonicalizeWhileInstruction(HloInstruction* loop) {
  CHECK_EQ(loop->opcode(), HloOpcode::kWhile);
  HloComputation* body = loop->while_body();
  HloComputation* cond = loop->while_condition();

  // Tuplify the body parameter if needed.
  HloInstruction* body_parameter = body->parameter_instruction(0);
  if (!body_parameter->shape().IsTuple()) {
    *body_parameter->mutable_shape() =
        ShapeUtil::MakeTupleShape({body_parameter->shape()});
    HloInstruction* original = body->AddInstruction(
        HloInstruction::CreateGetTupleElement(body_parameter, 0));
    TF_RETURN_IF_ERROR(
        body_parameter->ReplaceAllUsesWithDifferentShape(original));
  }

  // Tuplify the body root if needed.
  HloInstruction* root = body->root_instruction();
  if (!root->shape().IsTuple()) {
    root = body->AddInstruction(HloInstruction::CreateTuple({root}));
    body->set_root_instruction(root, /*accept_different_shape=*/true);
  }

  // Tuplify the condition parameter if needed.
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
    loop_tuple = TupleUtil::Duplicate(loop_tuple);
    TF_RETURN_IF_ERROR(loop->ReplaceOperandWith(0, loop_tuple));
  }

  // Explicitly make the root of the body a tuple.
  if (root->opcode() != HloOpcode::kTuple) {
    root = TupleUtil::Duplicate(root);
    body->set_root_instruction(root);
  }

  // Explicitly disjoin the loop from being a computation root, so that
  // we can insert tokens into, while preserving the original computation shape.
  if (loop->IsRoot()) {
    HloInstruction* new_root = TupleUtil::Duplicate(loop);
    loop->parent()->set_root_instruction(new_root);
  }

  return absl::OkStatus();
}

absl::Status InfeedTokenPropagation::PropagateTokenThroughConditionalBranch() {
  // Conditional branches can diverge in inputs, but must converge on outputs.

  HloComputation* comp = dangling_instruction_->parent();
  dangling_instruction_ = call_graph_->GetComputationCallers(comp)[0];
  CHECK_EQ(dangling_instruction_->opcode(), HloOpcode::kConditional);

  // Insert the output token into each branch.
  for (HloComputation* branch : dangling_instruction_->branch_computations()) {
    HloInstruction* root = branch->root_instruction();
    if (branch == comp) {
      TF_RETURN_IF_ERROR(
          InsertTokenIntoTuple(root, /*add_token_operand=*/false).status());
      root->AppendOperand(output_token_);
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
  TF_RETURN_IF_ERROR(input_token_->ReplaceAllUsesWith(input_token_gte));

  // Insert the input token into the branch tuple.
  int64_t branch_operand_idx = dangling_instruction_->branch_index(comp) + 1;
  HloInstruction* branch_tuple =
      dangling_instruction_->mutable_operand(branch_operand_idx);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * next_input_token_gte,
      InsertTokenIntoTuple(branch_tuple, /*add_token_operand=*/true));
  TF_RETURN_IF_ERROR(dangling_instruction_->ReplaceOperandWithDifferentShape(
      branch_operand_idx, branch_tuple));
  input_token_ =
      branch_tuple->mutable_operand(next_input_token_gte->tuple_index());

  // Insert the output token into conditional instruction.
  TF_ASSIGN_OR_RETURN(
      output_token_,
      InsertTokenIntoTuple(dangling_instruction_, /*add_token_operand=*/false));

  return absl::OkStatus();
}

absl::Status InfeedTokenPropagation::PropagateTokenThroughWhileBody() {
  // While loops need to converge on input and output.

  HloComputation* comp = dangling_instruction_->parent();
  dangling_instruction_ = call_graph_->GetComputationCallers(comp)[0];
  CHECK_EQ(dangling_instruction_->opcode(), HloOpcode::kWhile);

  // Insert the output token into the body root.
  HloInstruction* root = comp->root_instruction();
  TF_RETURN_IF_ERROR(
      InsertTokenIntoTuple(root, /*add_token_operand=*/false).status());
  root->AppendOperand(output_token_);

  // Insert the input token into the body parameter.
  HloInstruction* body_parameter = comp->parameter_instruction(0);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * input_token_gte,
      InsertTokenIntoTuple(body_parameter, /*add_token_operand=*/false));
  TF_RETURN_IF_ERROR(input_token_->ReplaceAllUsesWith(input_token_gte));

  // Insert the input token into the condition parameter.
  HloComputation* cond = dangling_instruction_->while_condition();
  HloInstruction* cond_parameter = cond->parameter_instruction(0);
  TF_RETURN_IF_ERROR(
      InsertTokenIntoTuple(cond_parameter, /*add_token_operand=*/false)
          .status());

  // Insert the input token into the while tuple.
  HloInstruction* while_tuple = dangling_instruction_->mutable_operand(0);
  TF_ASSIGN_OR_RETURN(
      input_token_,
      InsertTokenIntoTuple(while_tuple, /*add_token_operand=*/true));
  // Retrieve the actual token added to the tuple.
  input_token_ = input_token_->mutable_operand(0)->mutable_operand(
      input_token_->tuple_index());
  TF_RETURN_IF_ERROR(
      dangling_instruction_->ReplaceOperandWithDifferentShape(0, while_tuple));

  // Insert the input token into the while instruction.
  TF_ASSIGN_OR_RETURN(
      output_token_,
      InsertTokenIntoTuple(dangling_instruction_, /*add_token_operand=*/false));

  return absl::OkStatus();
}

absl::Status InfeedTokenPropagation::PropagateToken(
    const HloOrdering& ordering) {
  HloComputation* comp = dangling_instruction_->parent();
  if (dangling_instruction_->opcode() != HloOpcode::kInfeed &&
      dangling_instruction_->opcode() != HloOpcode::kOutfeed) {
    for (HloInstruction* instruction : comp->instructions()) {
      if (instruction->opcode() == original_opcode_) {
        HloInstruction* begin = ChainBegin(instruction);
        HloInstruction* end = ChainEnd(instruction);
        if (ordering.ExecutesBefore(end, dangling_instruction_)) {
          // Parent infeed happens before child infeed. Stitch via parent result
          // token.
          CHECK_EQ(begin->opcode(), HloOpcode::kInfeed);
          HloInstruction* parent_output_token = comp->AddInstruction(
              HloInstruction::CreateGetTupleElement(end, 1));
          TF_RETURN_IF_ERROR(
              input_token_->ReplaceAllUsesWith(parent_output_token));
          input_token_ = begin->mutable_operand(0);
        } else if (ordering.ExecutesBefore(dangling_instruction_, begin)) {
          // Parent outfeed happens after child infeed. Stitch via parent input
          // token.
          CHECK_EQ(begin->opcode(), HloOpcode::kOutfeed);
          TF_RETURN_IF_ERROR(begin->ReplaceOperandWith(1, output_token_));
          output_token_ = end;
        } else {
          LOG(WARNING) << absl::StrFormat(
              "Execution order of %s, %s and %s is undefined. This may lead to "
              "incorrect results",
              begin->name(), end->name(), dangling_instruction_->name());
        }
        // We assume that a well defined HLO graph only contains a single
        // infeed chain per computation.
        break;
      }
    }
  }
  if (comp->IsEntryComputation()) {
    return absl::OkStatus();
  }
  VLOG(2) << "Propagating tokens for: " << dangling_instruction_->name();

  HloInstruction* caller = call_graph_->GetComputationCallers(comp)[0];
  // TODO: b/368327832 - Skip handling sharding until it is removed.
  if (caller->has_sharding()) {
    return absl::OkStatus();
  }
  if (caller->opcode() == HloOpcode::kConditional) {
    TF_RETURN_IF_ERROR(CanonicalizeConditionalInstruction(caller));
    TF_RETURN_IF_ERROR(PropagateTokenThroughConditionalBranch());
  } else if (caller->opcode() == HloOpcode::kWhile &&
             comp == caller->while_body()) {
    TF_RETURN_IF_ERROR(CanonicalizeWhileInstruction(caller));
    TF_RETURN_IF_ERROR(PropagateTokenThroughWhileBody());
  } else {
    // We only expect to encounter computations behind while and conditional
    // instructions. In the case of it being behind a while condition, there is
    // no way to propagate the output token, as the root only returns a
    // predicate. All other computations that could possibly contain infeed
    // or outfeed ops should have already been inlined.
    VLOG(2) << "Unhandled computation: " << comp->name();
    return absl::OkStatus();
  }

  return PropagateToken(ordering);
}

absl::StatusOr<bool> InfeedTokenPropagation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
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
          break;
        }
      }
      for (HloInstruction* instruction : computation->instructions()) {
        if (instruction->opcode() == HloOpcode::kOutfeed &&
            IsDanglingOutfeed(instruction)) {
          VLOG(1) << "Found dangling outfeed: " << instruction->ToString();
          dangling_outfeeds.push_back(instruction);
          break;
        }
      }
    }
  }
  bool changed = !dangling_infeeds.empty() || !dangling_outfeeds.empty();

  if (changed) {
    call_graph_ = CallGraph::Build(module, execution_threads);
    if (!call_graph_->IsFlattened()) {
      return FailedPrecondition(
          "Call graph must be flattened before infeed token propagation.");
    }
    DependencyHloOrdering ordering = DependencyHloOrdering(module);

    for (HloInstruction* dangling_infeed : dangling_infeeds) {
      // In the process of token propagation, we might have stitched two
      // previously dangling infeeds token, causing both to no longer be
      // dangling.
      if (!IsDanglingInfeed(dangling_infeed)) {
        continue;
      }
      dangling_instruction_ = dangling_infeed;
      original_opcode_ = HloOpcode::kInfeed;
      input_token_ = ChainBegin(dangling_infeed)->mutable_operand(0);
      output_token_ =
          ChainEnd(dangling_infeed)
              ->AddInstruction(
                  HloInstruction::CreateGetTupleElement(dangling_infeed, 1));
      TF_RETURN_IF_ERROR(PropagateToken(ordering));
    }
    for (HloInstruction* dangling_outfeed : dangling_outfeeds) {
      // In the process of token propagation, we might have stitched two
      // previously dangling outfeeds token, causing both to no longer be
      // dangling.
      if (!IsDanglingOutfeed(dangling_outfeed)) {
        continue;
      }
      dangling_instruction_ = dangling_outfeed;
      original_opcode_ = HloOpcode::kOutfeed;
      input_token_ = ChainBegin(dangling_outfeed)->mutable_operand(1);
      output_token_ = ChainEnd(dangling_outfeed);
      TF_RETURN_IF_ERROR(PropagateToken(ordering));
    }

    TF_RETURN_IF_ERROR(
        TupleSimplifier().Run(module, execution_threads).status());
    TF_RETURN_IF_ERROR(HloDCE().Run(module, execution_threads).status());
  }

  VLOG(5) << "After InfeedTokenPropagation:";
  XLA_VLOG_LINES(5, module->ToString());
  return changed;
}
}  // namespace xla
