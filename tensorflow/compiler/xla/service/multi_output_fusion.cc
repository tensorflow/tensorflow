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

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

StatusOr<bool> MultiOutputFusion::Run(HloModule* module) {
  bool changed = false;

  for (auto* computation : module->MakeNonfusionComputations()) {
    computation_ = computation;
    candidates_.clear();
    candidates_index_.clear();
    all_fusion_candidates_.clear();
    RecomputeReachability();

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
      absl::flat_hash_set<HloInstruction*> candidates_set;
      VLOG(10) << "Looking at instruction: " << instruction->name();
      for (auto operand : instruction->operands()) {
        // Filter out the non-interesting instructions -- they
        // will not generate the savings.
        if (!IsProfitableOperand(operand)) {
          VLOG(10) << "Operand not profitable: " << operand->name();
          continue;
        }
        VLOG(10) << "Operand profitable: " << operand->name();
        // We don't look at all users of operands as it's quadratic. Only look
        // at one slice of users.
        const int64 kUserSliceSize = 128;

        const int64 user_slice_begin =
            RoundDownToNearest(operand->UserId(instruction), kUserSliceSize);

        const int64 user_slice_end =
            std::min(static_cast<int64>(operand->users().size()),
                     user_slice_begin + kUserSliceSize);

        for (int64 i = user_slice_begin; i < user_slice_end; ++i) {
          HloInstruction* user = operand->users()[i];
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
  // Clean up state in case this pass is wrapped in an HloPassPipeline.
  candidates_.clear();
  candidates_index_.clear();
  all_fusion_candidates_.clear();
  reachability_.reset();
  return changed;
}

HloInstruction* MultiOutputFusion::Fuse(HloInstruction* instr1,
                                        HloInstruction* instr2) {
  HloInstruction* remaining = instr1;
  HloInstruction* fused = instr2;
  // Make sure that if only one of the instructions is a fusion, or if only one
  // of the instructions is a multi-output fusion, it's what will be fused into.
  if (fused->opcode() == HloOpcode::kFusion) {
    std::swap(remaining, fused);
  }
  if (fused->IsMultiOutputFusion()) {
    std::swap(remaining, fused);
  }
  if (fused->opcode() == HloOpcode::kFusion) {
    remaining->MergeFusionInstructionIntoMultiOutput(fused);
  } else {
    remaining->FuseInstructionIntoMultiOutput(fused);
    CHECK_EQ(0, fused->user_count());
    TF_CHECK_OK(computation()->RemoveInstruction(fused));
  }
  return remaining;
}

HloInstruction* MultiOutputFusion::CreateFusion(HloInstruction* base,
                                                HloInstruction* to_fuse) {
  HloInstruction* input_fusion =
      computation()->AddInstruction(HloInstruction::CreateFusion(
          base->shape(), HloInstruction::FusionKind::kLoop, base));

  // Update candidate_ and all_fusion_candidates_.
  std::vector<std::pair<HloInstruction*, int64>> new_fusibles =
      GetNewFusibles(base, to_fuse);
  int64 index;
  if (candidates_index_.contains(input_fusion)) {
    index = candidates_index_[input_fusion];
  } else {
    index = candidates_.size();
    InsertOrDie(&candidates_index_, input_fusion, index);
    candidates_.emplace_back(input_fusion);
    all_fusion_candidates_.push_back(input_fusion);
  }

  // Update the worklist_.
  FusionCandidate& candidate_node = candidates_[index];
  for (auto it : new_fusibles) {
    candidate_node.fusibles.emplace_back(it.first, it.second);
    worklist_.emplace(input_fusion, it.first, it.second);
  }

  reachability_->Replace(base, input_fusion);
  TF_CHECK_OK(computation()->ReplaceInstruction(base, input_fusion));
  return input_fusion;
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

std::vector<std::pair<HloInstruction*, int64>>
MultiOutputFusion::GetNewFusibles(HloInstruction* fusion,
                                  HloInstruction* fused) {
  FusionCandidate& fusion_node = candidates_[get_candidate_id(fusion)];
  FusionCandidate& fused_node = candidates_[get_candidate_id(fused)];

  // Update the fusible list for fusion. Variable new_fusibles keeps
  // track of the new or changed entries.
  std::vector<std::pair<HloInstruction*, int64>> new_fusibles;
  absl::flat_hash_set<HloInstruction*> in_list;
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
    if (in_list.contains(instr)) {
      continue;
    }
    int64 profit = GetProfit(instr, fusion);
    fusion_node.fusibles.emplace_back(instr, profit);
    new_fusibles.emplace_back(instr, profit);
  }
  fused_node.fusibles.clear();

  return new_fusibles;
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

  // Update the reachability graph.
  UpdateReachability(fusion, fused, all_fusion_candidates_,
                     [this](HloInstruction* instr) { return is_fused(instr); });

  std::vector<std::pair<HloInstruction*, int64>> new_fusibles =
      GetNewFusibles(fusion, fused);

  // Update the worklist_.
  for (auto it : new_fusibles) {
    worklist_.emplace(fusion, it.first, it.second);
  }
}

bool MultiOutputFusion::LegalToFuse(HloInstruction* instr1,
                                    HloInstruction* instr2) {
  if (instr1->opcode() != HloOpcode::kFusion) {
    return false;
  }
  return LegalToFuseMainConstraints(instr1, instr2);
}

bool MultiOutputFusion::LegalToFuseMainConstraints(HloInstruction* instr1,
                                                   HloInstruction* instr2) {
  if (instr1 == instr2) {
    return false;
  }

  // Fusing nodes with 0 users makes no sense and the rest of the implementation
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

void MultiOutputFusion::RecomputeReachability() {
  // Free the memory used for the reachability map before computing a new one.
  reachability_.reset();
  reachability_ = HloReachabilityMap::Build(computation_);
}

void MultiOutputFusion::UpdateReachability(
    HloInstruction* instr1, HloInstruction* instr2,
    absl::Span<HloInstruction* const> instrs_to_update,
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
      if (!ConsumeFuel(name(), [&] {
            return absl::StrFormat("Not fusing %s and %s.", instr1->ToString(),
                                   instr2->ToString());
          })) {
        break;
      }
      VLOG(1) << "Fuse!";
      VLOG(2) << "Before multi_output_fusion:";
      VLOG(2) << "instr1: " << instr1->ToString();
      if (instr1->opcode() == HloOpcode::kFusion) {
        VLOG(2) << "\n"
                << instr1->fused_instructions_computation()->ToString(
                       HloPrintOptions().set_indent_amount(1));
      }
      VLOG(2) << "instr2: " << instr2->ToString();
      if (instr2->opcode() == HloOpcode::kFusion) {
        VLOG(2) << "\n"
                << instr2->fused_instructions_computation()->ToString(
                       HloPrintOptions().set_indent_amount(1));
      }
      Update(instr1, instr2);
      HloInstruction* ret = Fuse(instr1, instr2);
      if (ret != instr1) {
        set_is_fused(instr1);
      }
      if (ret != instr2) {
        set_is_fused(instr2);
      }
      changed = true;
      VLOG(2) << "After fusion, \t this: " << ret->name() << "\n"
              << ret->fused_instructions_computation()->ToString(
                     HloPrintOptions().set_indent_amount(1));
    }
  }
  if (DoProducerConsumerMultiOutputFusion()) {
    changed = true;
  }
  return changed;
}

bool MultiOutputFusion::DoProducerConsumerMultiOutputFusion() { return false; }

}  // namespace xla
