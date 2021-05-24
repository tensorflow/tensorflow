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

#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"

#include <stdint.h>

#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/hlo_graph_dumper.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

namespace {

bool IsProfitableOperand(HloInstruction* instr) {
  // kConstant instruction will not have memory reads, so it won't be a profit
  // source. Skip them.
  if (instr->opcode() == HloOpcode::kConstant &&
      ShapeUtil::IsEffectiveScalar(instr->shape())) {
    return false;
  }
  return true;
}

bool LegalToFuse(HloInstruction* instr1, HloInstruction* instr2) {
  // If we're fusing fusions only do it if the fusion kind matches. Loop fusions
  // merge into bigger loop fusions and input (reduce) fusions become fusions
  // with multiple reduce outputs. We could fuse reduce and loop fusions
  // together too (the result being an input fusion) if we find cases where this
  // improves things. Also disable fusing standalone input-fusible reduces into
  // loop fusions.
  CHECK(instr1->opcode() == HloOpcode::kFusion);
  if ((instr2->opcode() == HloOpcode::kFusion &&
       instr1->fusion_kind() != instr2->fusion_kind()) ||
      (IsReductionFromOrToContiguousDimensions(*instr2) &&
       instr1->IsLoopFusion())) {
    return false;
  }
  // The emitter only supports in-place DUS for fusions with a single DUS at the
  // root. Don't sibling fuse DUS for now.
  // TODO(b/119178699): Multi-output fusing DUS can improve performance if we
  // share the input and output buffers and add support to the emitter.
  if (instr1->fused_expression_root()->opcode() ==
          HloOpcode::kDynamicUpdateSlice ||
      (instr2->opcode() == HloOpcode::kFusion &&
       instr2->fused_expression_root()->opcode() ==
           HloOpcode::kDynamicUpdateSlice)) {
    return false;
  }
  // Do this check last, as it may be expensive.
  return !FusionWouldBeTooLarge(*instr1, *instr2);
}

// We prefer multi-output fusions over other fusions over unfused ops, because
// we want to preserve fusion opportunities if possible.
int FusionPriority(const HloInstruction* instr) {
  if (instr->IsMultiOutputFusion()) {
    return 2;
  }
  if (instr->opcode() == HloOpcode::kFusion) {
    return 1;
  }
  return 0;
}

HloInstruction* SelectPreferredFusionCandidate(
    const std::vector<HloInstruction*> candidates) {
  if (candidates.empty()) {
    return nullptr;
  }
  return *std::max_element(
      candidates.begin(), candidates.end(),
      [](const HloInstruction* a, const HloInstruction* b) {
        return FusionPriority(a) < FusionPriority(b);
      });
}

std::vector<HloInstruction*> GetProducerConsumerMultiOutputFusionCandidates(
    const HloInstruction* producer, const HloReachabilityMap& reachability) {
  std::vector<HloInstruction*> fusion_candidates;
  // If there is only one user, and it is not a multi-output fusion node, this
  // fusion possibility was already considered and rejected by the FusionMerger
  // pass. No need to try again!
  if (producer->user_count() == 1 &&
      !producer->users()[0]->IsMultiOutputFusion()) {
    return fusion_candidates;
  }
  for (HloInstruction* consumer : producer->users()) {
    VLOG(3) << "Looking at producer " << producer->name()
            << " and its consumer " << consumer->name();
    if (!IsFusibleAsMultiOutputFusionRoot(*consumer)) {
      VLOG(3) << "Consumer " << consumer->name()
              << " is not eligible as multi-output fusion root.";
      continue;
    }
    if (!IsProducerConsumerMultiOutputFusible(*producer, *consumer)) {
      VLOG(3) << producer->name() << " and " << consumer->name()
              << " are not fusible.";
      continue;
    }
    // Do not fuse a producer if the other operands of the fusion are
    // reachable from the producer, this would create a cycle.
    auto operand_reachable_from_producer = [&](const HloInstruction* operand) {
      // If a get-tuple-element instruction is not in the reachability
      // map, it has been created by fusion in this pass. Simply move
      // on to its operand, which is in the reachability map.
      if (!reachability.IsPresent(operand) &&
          operand->opcode() == HloOpcode::kGetTupleElement) {
        operand = operand->operand(0);
      }
      CHECK(reachability.IsPresent(operand) && reachability.IsPresent(producer))
          << "Reachability map is incomplete. This should never "
             "happen.";
      return producer != operand && reachability.IsReachable(producer, operand);
    };
    if (absl::c_any_of(consumer->operands(), operand_reachable_from_producer)) {
      VLOG(3) << producer->name() << " would introduce a cycle when fused.";
      continue;
    }
    if (FusionWouldBeTooLarge(*producer, *consumer)) {
      VLOG(3) << producer->name() << " and " << consumer->name()
              << " would be too large of a fusion.";
      continue;
    }
    // Make sure the emitter can codegen the fusion op efficiently. We currently
    // can have exponential time/memory requirements for emitting certain fusion
    // ops, in which case we don't want to fuse.
    // TODO(b/119692968): Remove this once fixed in the emitter.
    if (FusedIrEmitter::IsFusedIrEmitterInefficient(consumer, producer)) {
      VLOG(3) << "Fusion of " << producer->name() << " into "
              << consumer->name()
              << " would result in overly large code duplication.";
      continue;
    }
    fusion_candidates.push_back(consumer);
  }
  return fusion_candidates;
}

bool IsSiblingFusionCandidate(const HloInstruction* instr) {
  if (instr->user_count() == 0) {
    return false;
  }
  if (!IsFusibleAsMultiOutputFusionRoot(*instr)) {
    return false;
  }
  // Check if the users of multioutput fusion is not a get-tuple-element.
  // If this is the case, we bail out because the transformation assumes
  // the users are get-tuple-element.
  if (instr->IsMultiOutputFusion()) {
    for (auto user : instr->users()) {
      if (user->opcode() != HloOpcode::kGetTupleElement) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

void GpuMultiOutputFusion::RecomputeReachability() {
  reachability_ = HloReachabilityMap::Build(computation_);
}

bool GpuMultiOutputFusion::FuseSiblings(HloInstruction* parent) {
  if (!IsProfitableOperand(parent)) {
    return false;
  }
  bool changed = false;
  std::vector<HloInstruction*> siblings = parent->users();
  // Sort the siblings such that multi-output fusion ops occur first, followed
  // by fusion ops, followed by unfused ops.
  absl::c_stable_sort(siblings,
                      [](const HloInstruction* a, const HloInstruction* b) {
                        return FusionPriority(a) > FusionPriority(b);
                      });
  for (auto i = siblings.begin(); i != siblings.end();) {
    VLOG(3) << "Considering " << (*i)->name();
    if ((*i)->opcode() != HloOpcode::kFusion || !IsSiblingFusionCandidate(*i)) {
      ++i;
      continue;
    }
    for (auto j = i + 1; j != siblings.end();) {
      VLOG(3) << "Considering " << (*i)->name() << " and " << (*j)->name();
      if (!IsSiblingFusionCandidate(*j) || reachability_->IsConnected(*i, *j) ||
          !ShapesCompatibleForMultiOutputFusion(*(*i), *(*j)) ||
          !LegalToFuse(*i, *j)) {
        ++j;
        continue;
      }
      if (!ConsumeFuel(name(), [&] {
            return absl::StrFormat("Not fusing siblings %s and %s.",
                                   (*i)->name(), (*j)->name());
          })) {
        ++j;
        continue;
      }
      VLOG(2) << "Fuse siblings " << (*i)->name() << " and " << (*j)->name();
      HloInstruction* remaining = *i;
      HloInstruction* fused = *j;
      if (fused->opcode() == HloOpcode::kFusion) {
        remaining->MergeFusionInstructionIntoMultiOutput(fused);
      } else {
        remaining->FuseInstructionIntoMultiOutput(fused);
        CHECK_EQ(0, fused->user_count());
        TF_CHECK_OK(computation_->RemoveInstruction(fused));
      }
      changed = true;
      siblings.erase(j);
      RecomputeReachability();
    }
    ++i;
  }
  return changed;
}

StatusOr<bool> GpuMultiOutputFusion::DoMultiOutputFusion() {
  bool changed = false;
  RecomputeReachability();
  std::vector<HloInstruction*> defs_before_uses =
      computation_->MakeInstructionPostOrder();

  auto dump_fusion_state = [&] {
    if (computation_->parent()
            ->config()
            .debug_options()
            .xla_dump_fusion_visualization()) {
      TF_RETURN_IF_ERROR(
          RegisterFusionState(*computation_, "GpuMultiOutputFusion"));
    }
    return Status::OK();
  };

  while (!defs_before_uses.empty()) {
    // Traverse the HLO in uses-before-defs order by removing instruction from
    // the back of the vector.
    HloInstruction* producer = defs_before_uses.back();
    defs_before_uses.pop_back();
    // Never multi-output fuse constants.  To the extent that we want to fuse
    // constants, that should be handled by the regular fusion pass.
    if (producer->opcode() == HloOpcode::kConstant) {
      VLOG(3) << producer->name() << " is a constant.";
      continue;
    }
    // First, fuse the consumer ops of the current op, which are siblings.
    if (FuseSiblings(/*parent=*/producer)) {
      changed = true;
    }
    // Second, perform producer-consumer multi-output fusion. This order will
    // ensure that all get-tuple-element ops inserted as a by-product of
    // multi-output fusion will occur before the current op in the order of
    // traversal, and hence, not get into the way of subsequent fusion attempts.
    const auto candidates = GetProducerConsumerMultiOutputFusionCandidates(
        producer, *reachability_);
    auto* consumer_for_fusion = SelectPreferredFusionCandidate(candidates);
    if (consumer_for_fusion == nullptr) {
      continue;
    }
    if (!ConsumeFuel(name(), [&] {
          return absl::StrFormat("Not fusing %s and %s.", producer->name(),
                                 consumer_for_fusion->name());
        })) {
      continue;
    }
    changed = true;
    if (consumer_for_fusion->opcode() == HloOpcode::kFusion) {
      VLOG(2) << "Fuse producer " << producer->name() << " into its consumer "
              << consumer_for_fusion->name();
      if (producer->opcode() == HloOpcode::kFusion) {
        consumer_for_fusion->MergeFusionInstructionIntoMultiOutput(producer);
      } else {
        consumer_for_fusion->FuseInstructionIntoMultiOutput(producer);
        CHECK_EQ(0, producer->user_count());
        TF_CHECK_OK(computation_->RemoveInstruction(producer));
      }

      TF_RETURN_IF_ERROR(dump_fusion_state());
      RecomputeReachability();
      continue;
    }
    HloInstruction* input_fusion =
        computation_->AddInstruction(HloInstruction::CreateFusion(
            consumer_for_fusion->shape(),
            ChooseFusionKind(*producer, *consumer_for_fusion),
            consumer_for_fusion));
    VLOG(2) << "Fuse producer " << producer->name() << " and its consumer "
            << consumer_for_fusion->name() << " into " << input_fusion->name();
    TF_CHECK_OK(
        computation_->ReplaceInstruction(consumer_for_fusion, input_fusion));
    if (producer->opcode() == HloOpcode::kFusion) {
      input_fusion->MergeFusionInstructionIntoMultiOutput(producer);
    } else {
      input_fusion->FuseInstructionIntoMultiOutput(producer);
      CHECK_EQ(0, producer->user_count());
      TF_CHECK_OK(computation_->RemoveInstruction(producer));
    }

    TF_RETURN_IF_ERROR(dump_fusion_state());
    RecomputeReachability();
  }
  return changed;
}

StatusOr<bool> GpuMultiOutputFusion::Run(HloModule* module) {
  bool changed = false;
  for (auto* computation : module->MakeNonfusionComputations()) {
    computation_ = computation;
    TF_ASSIGN_OR_RETURN(bool fusion_changed, DoMultiOutputFusion());
    if (fusion_changed) {
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
