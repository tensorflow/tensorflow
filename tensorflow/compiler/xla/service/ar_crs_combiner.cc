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

#include "tensorflow/compiler/xla/service/ar_crs_combiner.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/call_graph.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

namespace {

namespace m = match;

// If the argument instruction is a CRS in the sequence
// AR -> Convert -> Add -> CRS
// then return the AR in the sequence.
// TODO(b/117554291): Rewrite this to recognize more general patterns,
// not just the specific one of AR -> Add -> Convert -> CRS.
absl::optional<HloInstruction*> MatchesArCrsPattern(
    HloInstruction* instruction) {
  HloInstruction *ar, *convert, *add, *crs;
  if (Match(instruction,
            m::CrossReplicaSum(
                &crs, m::Add(&add, m::Op(),
                             m::Convert(&convert,
                                        m::CrossReplicaSum(&ar, m::Op()))))) &&
      ar->users().size() == 1 && ar->shape().element_type() == BF16 &&
      convert->shape().element_type() == F32 && !crs->all_reduce_id()) {
    return ar;
  }
  return absl::optional<HloInstruction*>();
}

}  // namespace

absl::optional<HloInstruction*> ArCrsCombiner::WhileFromBodyParameter(
    HloInstruction* instruction) {
  CHECK_EQ(HloOpcode::kParameter, instruction->opcode());
  HloComputation* computation = instruction->parent();
  auto caller_instructions = call_graph_->GetComputationCallers(computation);
  if (caller_instructions.size() == 1) {
    auto caller_instruction = caller_instructions[0];
    if (caller_instruction->opcode() == HloOpcode::kWhile) {
      return caller_instruction;
    }
  }
  return absl::optional<HloInstruction*>();
}

std::vector<HloInstruction*> ArCrsCombiner::GetAllTuples(
    HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kTuple) {
    return {instruction};
  }
  if (instruction->opcode() == HloOpcode::kDomain) {
    return GetAllTuples(instruction->operands()[0]);
  }
  if (instruction->opcode() == HloOpcode::kParameter) {
    auto maybe_while = WhileFromBodyParameter(instruction);
    if (!maybe_while) {
      return {};
    }
    auto while_instr = *maybe_while;
    auto init_tuples = GetAllTuples(while_instr->while_init());
    auto body_tuples =
        GetAllTuples(while_instr->while_body()->root_instruction());
    if (init_tuples.empty() || body_tuples.empty()) {
      return {};
    }
    init_tuples.insert(init_tuples.end(), body_tuples.begin(),
                       body_tuples.end());
    return init_tuples;
  }
  if (instruction->opcode() == HloOpcode::kGetTupleElement) {
    std::vector<HloInstruction*> result_tuples;
    for (auto tuple : GetAllTuples(instruction->operands()[0])) {
      auto tmp_tuples =
          GetAllTuples(tuple->mutable_operand(instruction->tuple_index()));
      if (tmp_tuples.empty()) {
        return {};
      }
      result_tuples.insert(result_tuples.end(), tmp_tuples.begin(),
                           tmp_tuples.end());
    }
    return result_tuples;
  }
  return {};
}

bool ArCrsCombiner::TupleElementsComputeSameValue(
    HloInstruction* tuple_shaped_instruction, int64 i1, int64 i2,
    absl::flat_hash_map<int64, int64>* visited_pairs) {
  auto tuples = GetAllTuples(tuple_shaped_instruction);
  if (tuples.empty()) {
    return false;
  }
  for (auto tuple : tuples) {
    CHECK_EQ(tuple->opcode(), HloOpcode::kTuple);
    if (!InstructionsComputeSameValue(tuple->mutable_operand(i1),
                                      tuple->mutable_operand(i2),
                                      visited_pairs)) {
      return false;
    }
  }
  return true;
}

/* static */
bool ArCrsCombiner::TestInstructionsComputeSameValue(HloInstruction* i1,
                                                     HloInstruction* i2) {
  ArCrsCombiner combiner(/*num_spatial_partitions=*/2);
  auto module = i1->parent()->parent();
  CHECK_EQ(module, i2->parent()->parent());
  combiner.call_graph_ = CallGraph::Build(module);
  absl::flat_hash_map<int64, int64> visited_pairs;
  return combiner.InstructionsComputeSameValue(i1, i2, &visited_pairs);
}

bool ArCrsCombiner::InstructionsComputeSameValue(
    HloInstruction* i1, HloInstruction* i2,
    absl::flat_hash_map<int64, int64>* visited_pairs) {
  if (i1 == i2) {
    return true;
  }
  auto uid1 = i1->unique_id();
  auto uid2 = i2->unique_id();
  auto min_uid = std::min(uid1, uid2);
  auto max_uid = std::max(uid1, uid2);
  auto it = visited_pairs->find(min_uid);
  if (it != visited_pairs->end() && max_uid == it->second) {
    return true;
  }
  auto opcode1 = i1->opcode();
  auto operands1 = i1->operands();
  if (opcode1 != i2->opcode() || operands1.size() != i2->operands().size()) {
    return false;
  }
  visited_pairs->emplace(min_uid, max_uid);
  for (int i = 0; i < operands1.size(); ++i) {
    auto operand1 = operands1[i];
    auto operand2 = i2->operands()[i];
    if (!InstructionsComputeSameValue(operand1, operand2, visited_pairs)) {
      return false;
    }
  }
  if (opcode1 == HloOpcode::kParameter) {
    // In the general case, we don't try to prove equality of parameters.
    // We only try in the context of get-tuple-element
    // (see TupleElementsComputeSameValue).
    return false;
  }
  if (opcode1 == HloOpcode::kGetTupleElement) {
    return i1->tuple_index() == i2->tuple_index() ||
           TupleElementsComputeSameValue(operands1[0], i1->tuple_index(),
                                         i2->tuple_index(), visited_pairs);
  }
  // Don't check that the operands are identical, because Identical can
  // return false for instructions that compute the same value but are not
  // identical, which we don't want. We have checked the arguments with
  // InstructionsComputeSameValue earlier.
  auto eq_instructions = [](const HloInstruction* i1,
                            const HloInstruction* i2) -> bool { return true; };
  auto eq_computations = [](const HloComputation* a, const HloComputation* b) {
    return *a == *b;
  };
  return i1->Identical(*i2, eq_instructions, eq_computations,
                       /*layout_sensitive=*/false);
}

void ArCrsCombiner::GroupAllReducesById(HloModule* module) {
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      auto ar = MatchesArCrsPattern(instruction);
      if (ar) {
        all_reduce_map_[*((*ar)->all_reduce_id())].push_back(*ar);
      }
    }
  }
}

void ArCrsCombiner::KeepProvablyEqualInstructionGroups() {
  for (auto it : all_reduce_map_) {
    auto instruction_vec = it.second;
    CHECK_EQ(instruction_vec.size(), num_spatial_partitions_);

    auto instr_0 = instruction_vec[0];
    auto add_0 = instr_0->users()[0]->users()[0];
    CHECK_EQ(HloOpcode::kAdd, add_0->opcode());

    for (int i = 1; i < instruction_vec.size(); ++i) {
      auto instr_i = instruction_vec[i];
      auto add_i = instr_i->users()[0]->users()[0];
      CHECK_EQ(HloOpcode::kAdd, add_i->opcode());
      absl::flat_hash_map<int64, int64> visited_pairs;
      if (!InstructionsComputeSameValue(add_0, add_i, &visited_pairs)) {
        all_reduce_map_.erase(it.first);
      }
    }
  }
}

StatusOr<bool> ArCrsCombiner::RewriteGraph() {
  if (all_reduce_map_.empty()) {
    return false;
  }

  auto computation_is_addition = [](HloComputation* c) {
    return c->instruction_count() == 3 &&
           Match(c->root_instruction(), m::Add(m::Parameter(), m::Parameter()));
  };

  for (auto it : all_reduce_map_) {
    auto instruction_vec = it.second;
    for (auto all_reduce : instruction_vec) {
      auto parent_computation = all_reduce->parent();
      auto convert = all_reduce->users()[0];
      auto add = convert->users()[0];
      auto crs = add->users()[0];

      if (!computation_is_addition(all_reduce->called_computations()[0]) ||
          !computation_is_addition(crs->called_computations()[0])) {
        continue;
      }
      HloInstruction* other_summand = (add->operands()[0] == convert)
                                          ? add->operands()[1]
                                          : add->operands()[0];
      // To move the AR past the addition, we need to divide other_summand by
      // the number of spatial partitions.
      CHECK_EQ(all_reduce->user_count(), 1);
      TF_CHECK_OK(
          all_reduce->ReplaceAllUsesWith(all_reduce->mutable_operand(0)));
      auto shape = other_summand->shape();
      Literal lit(shape);
      lit.PopulateWithValue<float>(num_spatial_partitions_);
      auto divisor = parent_computation->AddInstruction(
          HloInstruction::CreateConstant(lit.Clone()));
      auto division =
          parent_computation->AddInstruction(HloInstruction::CreateBinary(
              shape, HloOpcode::kDivide, other_summand, divisor));
      TF_CHECK_OK(other_summand->ReplaceUseWith(add, division));
      // The AllReduce and the CRS are combined to an all-core AllReduce.
      crs->set_all_reduce_id(all_reduce->all_reduce_id());
      TF_CHECK_OK(parent_computation->RemoveInstruction(all_reduce));
    }
  }

  return true;
}

StatusOr<bool> ArCrsCombiner::Run(HloModule* module) {
  call_graph_ = CallGraph::Build(module);

  GroupAllReducesById(module);

  KeepProvablyEqualInstructionGroups();

  return RewriteGraph();
}

}  // namespace xla
