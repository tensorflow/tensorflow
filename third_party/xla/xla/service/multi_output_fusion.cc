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

#include "xla/service/multi_output_fusion.h"

#include <optional>

#include "absl/container/flat_hash_set.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/hlo_dce.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<bool> MultiOutputFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (auto* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    // Do not operate over async computations (computations of async
    // instructions).
    if (computation->IsAsyncComputation()) {
      continue;
    }
    computation_ = computation;
    candidates_.clear();
    candidates_index_.clear();
    all_fusion_candidates_.clear();
    RecomputeReachability();

    int64_t index = 0;
    for (auto it : computation_->MakeInstructionPostOrder()) {
      candidates_.emplace_back(it);
      InsertOrDie(&candidates_index_, it, index++);
    }

    // Create the initial candidate list for each Node.
    for (auto& node : candidates_) {
      HloInstruction* instruction = node.hlo;
      int64_t instruction_id = get_candidate_id(instruction);
      FusionCandidate& instr_node = candidates_[instruction_id];
      if (!IsFusible(instruction)) {
        continue;
      }
      all_fusion_candidates_.emplace_back(instruction,
                                          reachability_->GetIndex(instruction));

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
        const int64_t kUserSliceSize = 128;

        const int64_t user_slice_begin =
            RoundDownTo(operand->UserId(instruction), kUserSliceSize);

        const int64_t user_slice_end =
            std::min(static_cast<int64_t>(operand->users().size()),
                     user_slice_begin + kUserSliceSize);

        for (int64_t i = user_slice_begin; i < user_slice_end; ++i) {
          HloInstruction* user = operand->users()[i];
          VLOG(10) << "User: " << user->name();
          if (user == instruction || !IsFusible(user)) {
            VLOG(10) << "User is not fusible, or is the instruction itself: "
                     << user->name();
            continue;
          }
          int64_t user_id = get_candidate_id(user);
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
        int64_t profit = GetProfit(instruction, candidate);
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
  if (changed) {
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module, execution_threads).status());
  }
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
  int64_t index = candidates_.size();
  InsertOrDie(&candidates_index_, input_fusion, index);
  candidates_.emplace_back(input_fusion);
  reachability_->Replace(base, input_fusion);
  all_fusion_candidates_.emplace_back(input_fusion,
                                      reachability_->GetIndex(input_fusion));
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

std::vector<std::pair<HloInstruction*, int64_t>>
MultiOutputFusion::GetNewFusibles(HloInstruction* instr1,
                                  HloInstruction* instr2) {
  HloInstruction* fusion = instr1;
  HloInstruction* fused = instr2;
  if (is_fused(instr1)) {
    fusion = instr2;
    fused = instr1;
  }

  FusionCandidate& fusion_node = candidates_[get_candidate_id(fusion)];
  FusionCandidate& fused_node = candidates_[get_candidate_id(fused)];

  // The second entry of the pair is an old profit value.
  std::vector<std::pair<HloInstruction*, int64_t>> new_fusibles;
  absl::flat_hash_set<HloInstruction*> in_list;
  auto it = fusion_node.fusibles.begin();
  while (it != fusion_node.fusibles.end()) {
    HloInstruction* instr = it->first;
    if (is_fused(instr) || is_connected(fusion, instr)) {
      it = fusion_node.fusibles.erase(it);
      continue;
    }
    in_list.insert(instr);
    new_fusibles.emplace_back(instr, it->second);
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
    // Set old profit to zero because instr is not originally fusible to
    // fusion_node.
    new_fusibles.emplace_back(instr, 0);
  }
  fused_node.fusibles.clear();

  return new_fusibles;
}

void MultiOutputFusion::UpdateBeforeFuse(HloInstruction* instr1,
                                         HloInstruction* instr2) {
  HloInstruction* fusion = instr1;
  HloInstruction* fused = instr2;
  if (is_fused(instr1)) {
    fusion = instr2;
    fused = instr1;
  }

  // Insert the newly created instruction (if any), to candidates_.
  for (auto use : fusion->users()) {
    if (candidates_index_.find(use) == candidates_index_.end()) {
      int64_t index = candidates_.size();
      candidates_.emplace_back(use);
      InsertOrDie(&candidates_index_, use, index++);
    }
  }

  // Update the reachability graph.
  UpdateReachability(fusion, fused, all_fusion_candidates_,
                     [this](HloInstruction* instr) { return is_fused(instr); });
}

void MultiOutputFusion::UpdateAfterFuse(
    HloInstruction* fusion,
    const std::vector<std::pair<HloInstruction*, int64_t>>& new_fusibles,
    bool new_fusion_node) {
  FusionCandidate& candidate_node = candidates_[candidates_index_[fusion]];
  for (auto it : new_fusibles) {
    int64_t profit = GetProfit(it.first, fusion);
    if (new_fusion_node) {
      // If `fusion' is a new fusion node, then add all fusibles.
      if (profit > 0) {
        candidate_node.fusibles.emplace_back(it.first, profit);
        worklist_.emplace(fusion, it.first, profit);
      }
    } else {
      if (profit > it.second) {
        // If the new profit is higher than the old profit, add the fusible
        // into worklist.
        worklist_.emplace(fusion, it.first, profit);
      }
      if (it.second == 0) {
        // If the old profit is zero, that means `it.first' is not
        // originally fusible to the base op of `fusion', so we must add it
        // to candidate_node.fusibles.
        candidate_node.fusibles.emplace_back(it.first, profit);
      }
    }
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
  if (instr1->IsDead() || instr2->IsDead()) {
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

  // If both nodes are in-place operations and they use a common in-place
  // operand, we can't fuse these two.
  for (const auto& operand_and_output_index1 :
       HloDataflowAnalysis::GetInPlaceInputOutputPairs(instr1)) {
    const HloInstruction* operand =
        instr1->operand(operand_and_output_index1.first.operand_number);
    for (const auto& operand_and_output_index2 :
         HloDataflowAnalysis::GetInPlaceInputOutputPairs(instr2)) {
      if (operand ==
          instr2->operand(operand_and_output_index2.first.operand_number)) {
        return false;
      }
    }
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
    absl::Span<const std::pair<HloInstruction*, HloReachabilityMap::Index>>
        instrs_to_update,
    std::optional<absl::FunctionRef<bool(HloInstruction*)>> skip) {
  auto instr1_i = reachability_->GetIndex(instr1);
  auto instr2_i = reachability_->GetIndex(instr2);
  for (auto& instr_and_index : instrs_to_update) {
    HloInstruction* instr = instr_and_index.first;
    if (skip != std::nullopt && (*skip)(instr)) {
      continue;
    }
    auto instr_i = instr_and_index.second;
    bool instr2_instr = reachability_->IsReachable(instr2_i, instr_i);
    bool instr1_instr = reachability_->IsReachable(instr1_i, instr_i);
    if (instr2_instr && instr1_instr) {
      // If a candidate was already reachable by both, no update needed.
      continue;
    }
    if (instr2_instr) {
      reachability_->FastSetReachabilityToUnion({instr_i, instr1_i}, instr_i);
    }
    if (reachability_->IsReachable(instr1_i, instr_i)) {
      reachability_->FastSetReachabilityToUnion({instr_i, instr2_i}, instr_i);
    }
  }
}

bool MultiOutputFusion::Perform() {
  int changed = false;
  // Pick the top candidate from queue and try to merge.
  while (!worklist_.empty()) {
    ToBeFused candidate = worklist_.pop();

    HloInstruction* instr1 = candidate.instr1;
    HloInstruction* instr2 = candidate.instr2;

    // Candidates are already fused.
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
      UpdateBeforeFuse(instr1, instr2);
      std::vector<std::pair<HloInstruction*, int64_t>> new_fusibles =
          GetNewFusibles(instr1, instr2);
      HloInstruction* fusion = Fuse(instr1, instr2);
      if (fusion != instr1) {
        set_is_fused(instr1);
      }
      if (fusion != instr2) {
        set_is_fused(instr2);
      }
      UpdateAfterFuse(
          fusion, new_fusibles,
          /*new_fusion_node=*/(fusion != instr1) && (fusion != instr2));

      changed = true;
      VLOG(2) << "After fusion, \t this: " << fusion->name() << "\n"
              << fusion->fused_instructions_computation()->ToString(
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
