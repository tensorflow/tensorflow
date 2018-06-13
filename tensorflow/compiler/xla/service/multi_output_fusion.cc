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

#include "tensorflow/compiler/xla/service/multi_output_fusion.h"

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

StatusOr<bool> MultiOutputFusion::Run(HloModule* module) {
  bool changed = false;

  for (auto* computation : module->MakeNonfusionComputations()) {
    computation_ = computation;
    reachability_ = computation_->ComputeReachability();
    candidates_.clear();
    candidates_index_.clear();
    all_fusion_candidates_.clear();

    int64 index = 0;
    for (auto it : computation_->MakeInstructionPostOrder()) {
      candidates_.emplace_back(it);
      InsertOrDie(&candidates_index_, it, index++);
    }

    // Create the initial candidate list for each Node.
    for (auto& node : candidates_) {
      HloInstruction* instruction = node.hlo;
      int64 instruction_id = get_candidate_id(instruction);
      FusionCandidate& instr_node = candidates_[instruction_id];
      if (!IsFusible(instruction)) {
        continue;
      }
      all_fusion_candidates_.push_back(instruction);

      std::vector<HloInstruction*> candidates;
      tensorflow::gtl::FlatSet<HloInstruction*> candidates_set;
      VLOG(10) << "Looking at instruction: " << instruction->name();
      for (auto operand : instruction->operands()) {
        // Filter out the non-interesting instructions -- they
        // will not generate the savings.
        if (!IsProfitableOperand(operand)) {
          VLOG(10) << "Operand not profitable: " << operand->name();
          continue;
        }
        VLOG(10) << "Operand profitable: " << operand->name();
        for (auto user : operand->users()) {
          VLOG(10) << "User: " << user->name();
          if (user == instruction || !IsFusible(user)) {
            VLOG(10) << "User is not fusible, or is the instruction itself: "
                     << user->name();
            continue;
          }
          int64 user_id = get_candidate_id(user);
          if (is_connected(instruction, user)) {
            VLOG(10) << "User is connected: " << user->name();
            continue;
          }
          if (instruction_id < user_id &&
              user->opcode() == HloOpcode::kFusion) {
            VLOG(10) << "User ID for user: " << user->name() << " is "
                     << user_id << " which is higher than " << instruction_id;
            continue;
          }
          if (!LegalToFuse(instruction, user)) {
            VLOG(10) << "User not legal to fuse: " << user->name();
            continue;
          }
          if (candidates_set.insert(user).second) {
            VLOG(10) << "User added to candidate list: " << user->name();
            candidates.push_back(user);
          }
        }
      }

      // Iterate over candidates rather than candidates_set to avoid
      // nondeterminism.
      for (auto candidate : candidates) {
        int64 profit = GetProfit(instruction, candidate);
        if (profit > 0) {
          FusionCandidate& candidate_node =
              candidates_[get_candidate_id(candidate)];
          instr_node.fusibles.emplace_back(candidate, profit);
          candidate_node.fusibles.emplace_back(instruction, profit);
          worklist_.emplace(instruction, candidate, profit);
        }
      }
    }
    if (Perform()) {
      changed = true;
    }
  }
  return changed;
}

HloInstruction* MultiOutputFusion::Fuse(HloInstruction* instr1,
                                        HloInstruction* instr2) {
  HloInstruction* remaining = instr1;
  HloInstruction* fused = instr2;
  // Make sure that if only one of the instructions is a fusion, or if only one
  // of the instructions is a multi-output fusion, it's what will be fused into.
  //
  // An invariant is that no bitcast nodes will show up in the middle of a
  // fusion node. This invariant must hold in order for us to lower it. Given
  // that, we require that during multi-output fusion, a fusion node ending with
  // bitcast to preserve its structure as a nested fusion instead being
  // merged and flattened.
  if (fused->opcode() == HloOpcode::kFusion &&
      fused->fused_expression_root()->opcode() != HloOpcode::kBitcast) {
    std::swap(remaining, fused);
  }
  if (fused->IsMultiOutputFusion()) {
    std::swap(remaining, fused);
  }

  if (fused->opcode() == HloOpcode::kFusion &&
      fused->fused_expression_root()->opcode() != HloOpcode::kBitcast) {
    remaining->MergeFusionInstructionIntoMultiOutput(fused);
  } else {
    if (remaining->opcode() == HloOpcode::kFusion &&
        remaining->fused_expression_root()->opcode() == HloOpcode::kBitcast) {
      auto parent_computation = remaining->parent();
      // Create a nested fusion node.
      auto remaining_nested_fused =
          parent_computation->AddInstruction(HloInstruction::CreateFusion(
              remaining->shape(), HloInstruction::FusionKind::kLoop,
              remaining));
      TF_CHECK_OK(parent_computation->ReplaceInstruction(
          remaining, remaining_nested_fused));
      remaining = remaining_nested_fused;
    }
    remaining->FuseInstructionIntoMultiOutput(fused);
  }

  return remaining;
}

bool MultiOutputFusion::IsProfitableOperand(HloInstruction* instr) {
  // kConstant instruction will not have memory reads, so it won't be a profit
  // source. Skip them.
  if (instr->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsEffectiveScalar(instr->shape())) {
    return false;
  }
  // We don't target to fuse producer/consumer instructions -- this should
  // be taken care of by the instruction_fusion pass. If instr has only
  // one user, it will not have sibling instructions. We won't consider it.
  if (instr->user_count() < 2) {
    return false;
  }
  return true;
}

void MultiOutputFusion::Update(HloInstruction* instr1, HloInstruction* instr2) {
  HloInstruction* fusion = instr1;
  HloInstruction* fused = instr2;
  if (is_fused(instr1)) {
    fusion = instr2;
    fused = instr1;
  }

  // Insert the newly created instruction (if any), to candidates_.
  for (auto use : fusion->users()) {
    if (candidates_index_.find(use) == candidates_index_.end()) {
      int64 index = candidates_.size();
      candidates_.emplace_back(use);
      InsertOrDie(&candidates_index_, use, index++);
    }
  }
  FusionCandidate& fusion_node = candidates_[get_candidate_id(fusion)];
  FusionCandidate& fused_node = candidates_[get_candidate_id(fused)];

  // Update the reachability graph.
  UpdateReachability(fusion, fused, all_fusion_candidates_,
                     [this](HloInstruction* instr) { return is_fused(instr); });

  // Update the fusible list for fusion. Variable new_fusibles keeps
  // track of the new or changed entries.
  std::vector<std::pair<HloInstruction*, int64>> new_fusibles;
  tensorflow::gtl::FlatSet<HloInstruction*> in_list;
  auto it = fusion_node.fusibles.begin();
  while (it != fusion_node.fusibles.end()) {
    HloInstruction* instr = it->first;
    if (is_fused(instr) || is_connected(fusion, instr)) {
      it = fusion_node.fusibles.erase(it);
      continue;
    }
    in_list.insert(instr);
    int64 profit = GetProfit(instr, fusion);
    if (profit > it->second) {
      it->second = profit;
      new_fusibles.emplace_back(instr, profit);
    }
    ++it;
  }

  // Fused_node has been fused into fusion_node. Take the fusion candidates
  // (fusibles) from fused_nodes and add them to the fusion_node's. Filter
  // out those fusibles that no longer valid (or already in the list).
  for (const auto& it : fused_node.fusibles) {
    HloInstruction* instr = it.first;
    if (instr == fusion || is_fused(instr) || is_connected(fusion, instr)) {
      continue;
    }
    if (in_list.count(instr) > 0) {
      continue;
    }
    int64 profit = GetProfit(instr, fusion);
    fusion_node.fusibles.emplace_back(instr, profit);
    new_fusibles.emplace_back(instr, profit);
  }
  fused_node.fusibles.clear();

  // Update the worklist_.
  for (auto it : new_fusibles) {
    worklist_.emplace(fusion, it.first, it.second);
  }
}

bool MultiOutputFusion::LegalToFuse(HloInstruction* instr1,
                                    HloInstruction* instr2) {
  if (instr1 == instr2) {
    return false;
  }
  if (instr1->opcode() != HloOpcode::kFusion) {
    return false;
  }

  // Fusing nodes with 0 user makes no sense and the rest of the implementation
  // doesn't support it either.
  if (instr1->user_count() == 0 || instr2->user_count() == 0) {
    return false;
  }

  // Check if the users of multioutput fusion is not a get-tuple-element.
  // If this is the case, we bail out because the transformation assumes
  // the users are get-tuple-element.
  auto multioutput_user_is_not_gte = [](HloInstruction* instr) {
    if (!instr->IsMultiOutputFusion()) {
      return false;
    }
    for (auto user : instr->users()) {
      if (user->opcode() != HloOpcode::kGetTupleElement) {
        return true;
      }
    }
    return false;
  };
  if (multioutput_user_is_not_gte(instr1) ||
      multioutput_user_is_not_gte(instr2)) {
    return false;
  }

  if (is_connected(instr1, instr2)) {
    return false;
  }
  if (!ShapesCompatibleForFusion(instr1, instr2)) {
    return false;
  }

  return true;
}

void MultiOutputFusion::UpdateReachability(
    HloInstruction* instr1, HloInstruction* instr2,
    tensorflow::gtl::ArraySlice<HloInstruction*> instrs_to_update,
    const std::function<bool(HloInstruction*)>& skip) {
  for (auto instr : instrs_to_update) {
    if (skip != nullptr && skip(instr)) {
      continue;
    }
    if (reachability_->IsReachable(instr2, instr) &&
        reachability_->IsReachable(instr1, instr)) {
      // If a candidate was already reachable by both, no update needed.
      continue;
    }
    if (reachability_->IsReachable(instr2, instr)) {
      reachability_->FastSetReachabilityToUnion({instr, instr1}, instr);
    }
    if (reachability_->IsReachable(instr1, instr)) {
      reachability_->FastSetReachabilityToUnion({instr, instr2}, instr);
    }
  }
}

bool MultiOutputFusion::Perform() {
  int changed = false;
  // Pick the top candidate from queue and try to merge.
  while (!worklist_.empty()) {
    if (fuel_ <= 0) {
      VLOG(2) << "No fusing: run out of fuel.";
      break;
    }
    ToBeFused candidate = worklist_.top();
    worklist_.pop();

    HloInstruction* instr1 = candidate.instr1;
    HloInstruction* instr2 = candidate.instr2;

    if (is_fused(instr1) || is_fused(instr2)) {
      continue;
    }

    VLOG(1) << "Considering candidate profit_score=" << candidate.score
            << "\n\t\tinstr1 = " << instr1->ToString()
            << "\n\t\tinstr2 = " << instr2->ToString();

    if (LegalToFuse(instr1, instr2)) {
      VLOG(1) << "Fuse!";
      VLOG(2) << "Before multi_output_fusion:";
      VLOG(2) << "instr1: " << instr1->ToString();
      VLOG(2) << "\n"
              << instr1->fused_instructions_computation()->ToString(
                     HloPrintOptions().set_indent_amount(1));
      VLOG(2) << "instr2: " << instr2->ToString();
      if (instr2->opcode() == HloOpcode::kFusion) {
        VLOG(2) << "\n"
                << instr2->fused_instructions_computation()->ToString(
                       HloPrintOptions().set_indent_amount(1));
      }
      HloInstruction* ret = Fuse(instr1, instr2);
      set_is_fused(ret == instr1 ? instr2 : instr1);
      Update(instr1, instr2);
      changed = true;
      VLOG(2) << "After fusion, \t this: " << ret->name() << "\n"
              << ret->fused_instructions_computation()->ToString(
                     HloPrintOptions().set_indent_amount(1));
      auto users = ret->users();
      --fuel_;
    }
  }
  if (DoProducerConsumerMultiOutputFusion(computation_)) {
    changed = true;
  }
  return changed;
}

bool MultiOutputFusion::DoProducerConsumerMultiOutputFusion(
    HloComputation* /*computation*/) {
  return false;
}
}  // namespace xla
