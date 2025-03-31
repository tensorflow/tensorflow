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

#include "xla/service/gpu/transforms/multi_output_fusion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/hlo_dfs_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/instruction_fusion.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

bool IsProfitableOperand(HloInstruction* instr) {
  // Effective scalars are not a profitable shared operand. Skip them.
  return !ShapeUtil::IsEffectiveScalar(instr->shape());
}

// Finds and returns the unique `slice` op where `parent` is used in `instr`.
// Returns `nullptr` if no such `slice` exists.
const HloSliceInstruction* FindUniqueSlice(const HloInstruction* parent,
                                           const HloInstruction* instr) {
  if (const auto* slice = DynCast<HloSliceInstruction>(instr)) {
    return slice;
  } else if (const auto* fusion = DynCast<HloFusionInstruction>(instr)) {
    const HloSliceInstruction* result = nullptr;
    for (size_t i = 0; i < fusion->operand_count(); ++i) {
      if (fusion->operand(i) == parent) {
        // Parameter used more than once -> there's no unique slice.
        if (result) return nullptr;

        auto* called_param = fusion->fused_parameter(i);
        if (called_param->user_count() != 1) return nullptr;

        result = FindUniqueSlice(called_param, called_param->users()[0]);
        if (!result) return nullptr;
      }
    }
    return result;
  } else {
    return nullptr;
  }
}

FusionDecision ParameterSlicesAreNonOverlapping(const HloInstruction& instr1,
                                                const HloInstruction& instr2,
                                                const HloInstruction* parent) {
  if (parent->shape().IsTuple()) return FusionDecision::Allow();
  // Allow MOF if the parameter is small, even if there's no overlap. 1024 bytes
  // were arbitrarily chosen as the threshold.
  if (ShapeUtil::ByteSizeOfElements(parent->shape()) < 1024) {
    return FusionDecision::Allow();
  }

  const HloSliceInstruction* slice1 = FindUniqueSlice(parent, &instr1);
  const HloSliceInstruction* slice2 = FindUniqueSlice(parent, &instr2);
  if (!slice1 || !slice2) return FusionDecision::Allow();

  // TODO(jreiffers): Check strides as well.
  auto& starts1 = slice1->slice_starts();
  auto& starts2 = slice2->slice_starts();
  auto& limits1 = slice1->slice_limits();
  auto& limits2 = slice2->slice_limits();

  for (int64_t dim = 0; dim < parent->shape().dimensions_size(); ++dim) {
    bool overlap = starts1[dim] < limits2[dim] && starts2[dim] < limits1[dim];
    if (!overlap) {
      return FusionDecision::Forbid("slices are non-overlapping");
    }
  }
  return FusionDecision::Allow();
}

FusionDecision LegalToFuse(const HloInstruction& instr1,
                           const HloInstruction& instr2,
                           const se::DeviceDescription& device_info,
                           FusionInfoCache* fusion_info_cache) {
  CHECK(instr1.opcode() == HloOpcode::kFusion);

  // The emitter only supports in-place DUS for fusions with a single DUS at the
  // root. Don't sibling fuse DUS for now.
  // TODO(b/119178699): Multi-output fusing DUS can improve performance if we
  // share the input and output buffers and add support to the emitter.
  if (instr1.fused_expression_root()->opcode() ==
          HloOpcode::kDynamicUpdateSlice ||
      (instr2.opcode() == HloOpcode::kFusion &&
       instr2.fused_expression_root()->opcode() ==
           HloOpcode::kDynamicUpdateSlice)) {
    return FusionDecision::Forbid("can't fuse multiple DUSs");
  }

  // Do this check last, as it may be expensive.
  return FusionFitsInBudget(instr1, instr2, device_info,
                            /*is_consumer_producer_fusion=*/false,
                            fusion_info_cache);
}

// We prefer multi-output fusions over other fusions over unfused ops, because
// we want to preserve fusion opportunities if possible.
int FusionPriority(const HloInstruction* instr) {
  if (instr->IsMultiOutputFusion()) {
    return 2;
  }
  if (HloPredicateIsOp<HloOpcode::kFusion>(instr)) {
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

// Do not fuse a producer if the other operands of the fusion are
// reachable from the producer, this would create a cycle.
FusionDecision OperandReachableFromProducer(
    const HloInstruction& producer, const HloInstruction& consumer,
    const HloDfsReachability& reachability) {
  for (const auto* operand : consumer.operands()) {
    // If a get-tuple-element instruction is not in the reachability
    // map, it has been created by fusion in this pass. Simply move
    // on to its operand, which is in the reachability map.
    if (!reachability.IsPresent(operand) &&
        HloPredicateIsOp<HloOpcode::kGetTupleElement>(operand)) {
      operand = operand->operand(0);
    }
    CHECK(reachability.IsPresent(operand) && reachability.IsPresent(&producer))
        << "Reachability map is incomplete. This should never "
           "happen.";
    if (&producer != operand && reachability.IsReachable(&producer, operand)) {
      return FusionDecision::Forbid(
          absl::StrCat(producer.name(), " would introduce a cycle when fused"));
    }
  }
  return FusionDecision::Allow();
}

FusionDecision ProducerCandidateIsFusible(
    const HloInstruction& producer, const HloInstruction& consumer,
    const HloDfsReachability& reachability, FusionInfoCache* fusion_info_cache,
    const se::DeviceDescription& device_info,
    GpuHloCostAnalysis* cost_analysis) {
  if (!IsFusibleAsMultiOutputFusionRoot(consumer, device_info)) {
    return FusionDecision::Forbid(
        "consumer not eligible as multi-output fusion root.");
  }

  RETURN_IF_NOT_FUSIBLE(
      ShapesCompatibleForMultiOutputFusion(consumer, producer, device_info));

  RETURN_IF_NOT_FUSIBLE(
      OperandReachableFromProducer(producer, consumer, reachability));

  RETURN_IF_NOT_FUSIBLE(FusionFitsInBudget(
      producer, consumer, device_info,
      /*is_consumer_producer_fusion=*/false, fusion_info_cache));

  if (cost_analysis->ProducerConsumerMergedTooLarge(producer, consumer)) {
    return FusionDecision::Forbid("will generate too large IR");
  }

  GpuPerformanceModel::RunTimes t =
      GpuPerformanceModel::EstimateRunTimesForMultiOutputFusion(
          &producer, &consumer, device_info, cost_analysis);
  if (t.time_fused > t.time_unfused) {
    return FusionDecision::Forbid("will execute slower if fused");
  }

  return FusionDecision::Allow();
}

std::vector<HloInstruction*> GetProducerConsumerMultiOutputFusionCandidates(
    const HloInstruction* producer, const HloDfsReachability& reachability,
    FusionInfoCache* fusion_info_cache,
    const se::DeviceDescription& device_info,
    GpuHloCostAnalysis* cost_analysis) {
  std::vector<HloInstruction*> fusion_candidates;
  const HloComputation* computation = producer->parent();
  const HloModule* module = computation->parent();
  bool dump_fusion =
      module->config().debug_options().xla_dump_fusion_visualization();

  // If the producer is not a valid candidate for MOF, no need to check any of
  // its users.
  if (!IsProducerMultiOutputFusible(*producer, device_info)) {
    return fusion_candidates;
  }

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

    if (auto decision = ProducerCandidateIsFusible(
            *producer, *consumer, reachability, fusion_info_cache, device_info,
            cost_analysis)) {
      fusion_candidates.push_back(consumer);
    } else if (dump_fusion) {
      RegisterFusionState(
          *computation,
          absl::StrCat("Not considering fusion of producer |", producer->name(),
                       "| into consumer |", consumer->name(),
                       "| due to: ", decision.Explain()),
          *consumer, producer);
    }
  }
  return fusion_candidates;
}

bool IsSiblingFusionCandidate(const HloInstruction* instr,
                              const se::DeviceDescription& device_info) {
  if (instr->users().empty() ||
      !IsFusibleAsMultiOutputFusionRoot(*instr, device_info) ||
      IsNestableVariadicReduction(*instr, device_info)) {
    return false;
  }
  // Check if the users of multioutput fusion is not a get-tuple-element.
  // If this is the case, we bail out because the transformation assumes
  // the users are get-tuple-element.
  return (!instr->IsMultiOutputFusion() ||
          absl::c_all_of(instr->users(),
                         HloPredicateIsOp<HloOpcode::kGetTupleElement>));
}

FusionDecision CanFuseSiblings(const HloInstruction& sibling_consumer_1,
                               const HloInstruction& sibling_consumer_2,
                               const HloInstruction& common_producer,
                               const HloDfsReachability& reachability,
                               FusionInfoCache* fusion_info_cache,
                               const se::DeviceDescription& device_info) {
  if (reachability.IsConnected(&sibling_consumer_1, &sibling_consumer_2)) {
    return FusionDecision::Forbid(
        absl::StrCat(sibling_consumer_1.name(), " and ",
                     sibling_consumer_2.name(), " are connected"));
  }

  RETURN_IF_NOT_FUSIBLE(ShapesCompatibleForMultiOutputFusion(
      sibling_consumer_1, sibling_consumer_2, device_info));

  // Technically, this check is order-dependent (e.g. siblings A, B, C where
  // {A, B} and {B, C} overlap, but {A, C} do not. If the priority order is
  // [C, A, B], only {C, B} will be fused, and A will only be fused in the
  // next iteration of the fusion pipeline, potentially requiring several
  // iterations to converge. We assume this case to be very rare in
  // practice.
  RETURN_IF_NOT_FUSIBLE(ParameterSlicesAreNonOverlapping(
      sibling_consumer_1, sibling_consumer_2, &common_producer));

  // This check should be last, as it may be expensive.
  RETURN_IF_NOT_FUSIBLE(LegalToFuse(sibling_consumer_1, sibling_consumer_2,
                                    device_info, fusion_info_cache));
  return FusionDecision::Allow();
}

}  // namespace

void MultiOutputFusion::RecomputeReachability() {
  reachability_ = HloDfsReachability::Build(computation_);
}

bool MultiOutputFusion::FuseSiblings(HloInstruction* parent,
                                     FusionInfoCache* fusion_info_cache,
                                     GpuHloCostAnalysis* cost_analysis) {
  const HloComputation* computation = parent->parent();
  const HloModule* module = computation->parent();
  bool dump_fusion =
      module->config().debug_options().xla_dump_fusion_visualization();

  if (!IsProfitableOperand(parent)) {
    VLOG(3) << "Operand " << parent->ToShortString() << " is not profitable";
    return false;
  }
  bool changed = false;
  std::vector<HloInstruction*> siblings;
  // Only consider siblings that are fusion candidates.
  absl::c_copy_if(parent->users(), std::back_inserter(siblings),
                  [&](const HloInstruction* instr) {
                    return IsSiblingFusionCandidate(instr, device_info_);
                  });
  // Sort the siblings such that multi-output fusion ops occur first, followed
  // by fusion ops, followed by unfused ops.
  absl::c_stable_sort(siblings,
                      [](const HloInstruction* a, const HloInstruction* b) {
                        return FusionPriority(a) > FusionPriority(b);
                      });

  for (auto i = siblings.begin(); i != siblings.end(); ++i) {
    VLOG(3) << "Considering " << (*i)->name();
    if ((*i)->opcode() != HloOpcode::kFusion) {
      continue;
    }
    for (auto j = i + 1; j != siblings.end();) {
      VLOG(3) << "Considering " << (*i)->name() << " and " << (*j)->name();

      if (auto fusible = CanFuseSiblings(**i, **j, *parent, *reachability_,
                                         fusion_info_cache, device_info_);
          !fusible) {
        // We pick `j` arbitrarily as a consumer.
        if (dump_fusion) {
          RegisterFusionState(
              *computation,
              absl::StrCat("Not fusing siblings |", (**i).name(), "| and |",
                           (**j).name(), "| due to: ", fusible.Explain()),
              // Randomly pick one consumer.
              /*consumer=*/**i,
              /*producer=*/parent);
        }
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
      fusion_info_cache->Invalidate(*i);
      fusion_info_cache->Invalidate(*j);
      HloInstruction* remaining = *i;
      HloInstruction* fused = *j;
      TF_CHECK_OK(cost_analysis->RemoveInstruction(remaining));
      TF_CHECK_OK(cost_analysis->RemoveInstruction(fused));

      DumpFusionState(*remaining,
                      absl::StrCat("About to fuse sibling |", fused->name(),
                                   "| into sibling |", remaining->name(),
                                   "| inside multi-output fusion"),
                      /*producer=*/fused);

      if (HloPredicateIsOp<HloOpcode::kFusion>(fused)) {
        remaining->MergeFusionInstructionIntoMultiOutput(fused);
        if (fused->IsInputFusion()) {
          remaining->set_fusion_kind(HloInstruction::FusionKind::kInput);
        }
      } else {
        remaining->FuseInstructionIntoMultiOutput(fused);
        CHECK_EQ(0, fused->user_count());
        TF_CHECK_OK(computation_->RemoveInstruction(fused));
      }
      DumpFusionState(*remaining,
                      absl::StrCat("Fused into |", remaining->name(),
                                   "| inside multi-output fusion"));
      TF_CHECK_OK(cost_analysis->RevisitInstruction(remaining));
      changed = true;
      siblings.erase(j);
      RecomputeReachability();
    }
  }
  return changed;
}

absl::StatusOr<bool> MultiOutputFusion::DoMultiOutputFusion() {
  bool changed = false;
  RecomputeReachability();
  GpuHloCostAnalysis cost_analysis({shape_size_function_,
                                    /*per_second_rates=*/{},
                                    /*min_latencies_seconds=*/{},
                                    /*count_multiple_input_accesses=*/true},
                                   device_info_);
  TF_RETURN_IF_ERROR(computation_->Accept(&cost_analysis));
  std::vector<HloInstruction*> defs_before_uses =
      computation_->MakeInstructionPostOrder();

  FusionInfoCache fusion_info_cache(device_info_);
  // Traverse the HLO in uses-before-defs order.
  for (auto it = defs_before_uses.rbegin(); it != defs_before_uses.rend();
       ++it) {
    auto* producer = *it;
    // Never multi-output fuse constants.  To the extent that we want to fuse
    // constants, that should be handled by the regular fusion pass.
    if (HloPredicateIsOp<HloOpcode::kConstant>(producer)) {
      VLOG(3) << producer->name() << " is a constant.";
      continue;
    }
    if (producer->IsCustomFusion()) {
      continue;
    }
    // First, fuse the consumer ops of the current op, which are siblings.
    if (FuseSiblings(/*parent=*/producer, &fusion_info_cache, &cost_analysis)) {
      changed = true;
    }
    // Second, perform producer-consumer multi-output fusion. This order will
    // ensure that all get-tuple-element ops inserted as a by-product of
    // multi-output fusion will occur before the current op in the order of
    // traversal, and hence, not get into the way of subsequent fusion attempts.
    const auto candidates = GetProducerConsumerMultiOutputFusionCandidates(
        producer, *reachability_, &fusion_info_cache, device_info_,
        &cost_analysis);
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
    fusion_info_cache.Invalidate(producer);
    fusion_info_cache.Invalidate(consumer_for_fusion);
    TF_RETURN_IF_ERROR(cost_analysis.RemoveInstruction(producer));
    TF_RETURN_IF_ERROR(cost_analysis.RemoveInstruction(consumer_for_fusion));

    HloInstruction* input_fusion;
    if (HloPredicateIsOp<HloOpcode::kFusion>(consumer_for_fusion)) {
      input_fusion = consumer_for_fusion;
      VLOG(2) << "Fuse producer " << producer->name() << " into its consumer "
              << consumer_for_fusion->name();
    } else {
      input_fusion = computation_->AddInstruction(HloInstruction::CreateFusion(
          consumer_for_fusion->shape(),
          ChooseFusionKind(*producer, *consumer_for_fusion, device_info_),
          consumer_for_fusion));
      VLOG(2) << "Fuse producer " << producer->name() << " and its consumer "
              << consumer_for_fusion->name() << " into "
              << input_fusion->name();
      TF_CHECK_OK(
          computation_->ReplaceInstruction(consumer_for_fusion, input_fusion));
    }

    DumpFusionState(*input_fusion,
                    absl::StrCat("About to fuse producer |", producer->name(),
                                 "| into consumer |", input_fusion->name(),
                                 "| inside multi-output fusion"),
                    /*producer=*/producer);

    if (HloPredicateIsOp<HloOpcode::kFusion>(producer)) {
      input_fusion->MergeFusionInstructionIntoMultiOutput(producer);
    } else {
      input_fusion->FuseInstructionIntoMultiOutput(producer);
      CHECK_EQ(0, producer->user_count());
      TF_CHECK_OK(computation_->RemoveInstruction(producer));
    }
    TF_RETURN_IF_ERROR(cost_analysis.RevisitInstruction(input_fusion));

    DumpFusionState(*input_fusion,
                    absl::StrCat("Fused into |", input_fusion->name(),
                                 "| inside multi-output fusion"));
    RecomputeReachability();
  }
  return changed;
}

void MultiOutputFusion::DumpFusionState(const HloInstruction& consumer,
                                        absl::string_view label,
                                        const HloInstruction* producer) {
  if (consumer.GetModule()
          ->config()
          .debug_options()
          .xla_dump_fusion_visualization()) {
    RegisterFusionState(*computation_, label, consumer, producer);
  }
}

absl::StatusOr<bool> MultiOutputFusion::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (auto* computation : GetFusibleComputations(*module, execution_threads)) {
    computation_ = computation;
    TF_ASSIGN_OR_RETURN(bool computation_changed, DoMultiOutputFusion());
    changed |= computation_changed;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
