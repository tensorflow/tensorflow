/* Copyright 2025 The OpenXLA Authors.


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

#include "xla/hlo/analysis/hlo_dimension_analysis.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

bool HloDimensionAnalysis::IsInstructionWeight(
    const HloInstruction* instruction) const {
  auto it = info_map_.find(instruction);
  if (it == info_map_.end()) {
    return false;
  }
  return absl::c_any_of(it->second.leaves(),
                        [](const std::pair<ShapeIndex, WeightInfo>& leaf) {
                          return leaf.second == WeightInfo::kWeight;
                        });
}

std::optional<ShapeTree<WeightInfo>> HloDimensionAnalysis::GetWeightInfo(
    const HloInstruction* instruction) const {
  auto it = info_map_.find(instruction);
  if (it == info_map_.end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::Status HloDimensionAnalysis::SetInstructionAsWeight(
    HloInstruction* instruction) {
  auto [it, success] = info_map_.emplace(
      std::piecewise_construct, std::forward_as_tuple(instruction),
      std::forward_as_tuple(instruction->shape(), WeightInfo::kUnknown));

  if (!success) {
    return absl::InternalError(absl::StrCat(
        "Instruction ", instruction->ToString(), " already has weight info."));
  }

  ShapeTree<WeightInfo>& weight_tree = it->second;
  weight_tree.ForEachMutableElement(
      [&](const ShapeIndex& index, WeightInfo* weight_info) {
        if (weight_tree.IsLeaf(index)) {
          *weight_info = WeightInfo::kWeight;
          return;
        }
        *weight_info = WeightInfo::kTuple;
      });
  return absl::OkStatus();
}

absl::Status HloDimensionAnalysis::SetWeightInfo(
    const HloInstruction* target, ShapeTree<WeightInfo> weight_annotation) {
  auto [it, success] = info_map_.emplace(target, std::move(weight_annotation));
  if (!success) {
    return absl::InternalError(absl::StrCat("Instruction ", target->ToString(),
                                            " already has weight info."));
  }
  return absl::OkStatus();
}

// Annotates the entry computation parameters as weights.
absl::Status HloDimensionAnalysis::AnnotateEntryComputationParameters(
    const HloModule& module) {
  const auto& params = module.entry_computation()->parameter_instructions();
  info_map_.reserve(params.size());
  for (HloInstruction* instruction : params) {
    TF_RETURN_IF_ERROR(SetInstructionAsWeight(instruction));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<HloDimensionAnalysis>> HloDimensionAnalysis::Run(
    const HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<HloDimensionAnalysis> weight_analysis =
      absl::WrapUnique(new HloDimensionAnalysis(module, execution_threads));
  TF_RETURN_IF_ERROR(
      weight_analysis->AnnotateEntryComputationParameters(module));
  TF_RETURN_IF_ERROR(
      weight_analysis->RunOnComputation(*module.entry_computation()));
  return weight_analysis;
}

absl::Status HloDimensionAnalysis::RunOnComputation(
    const HloComputation& computation) {
  if (HloInstruction::IsThreadIncluded(computation.execution_thread(),
                                       execution_threads_)) {
    HloWeightPropagation propagation(this);
    return propagation.Run(computation);
  }
  return absl::OkStatus();
}

absl::Status HloDimensionAnalysis::RunOnComputation(
    const HloComputation& computation,
    absl::Span<const HloInstruction* const> operands) {
  CHECK_EQ(computation.num_parameters(), operands.size());
  for (int i = 0; i < computation.num_parameters(); ++i) {
    auto weight_info_iter = info_map_.find(operands[i]);
    if (weight_info_iter == info_map_.end()) {
      continue;
    }
    TF_RETURN_IF_ERROR(SetWeightInfo(computation.parameter_instructions()[i],
                                     weight_info_iter->second));
  }
  return RunOnComputation(computation);
}

absl::Status HloWeightPropagation::Run(const HloComputation& computation) {
  TF_RETURN_IF_ERROR(computation.root_instruction()->Accept(this));
  for (HloInstruction* instruction : computation.instructions()) {
    if (instruction->user_count() == 0) {
      TF_RETURN_IF_ERROR(instruction->Accept(this));
    }
  }
  return absl::OkStatus();
}

absl::Status HloWeightPropagation::DefaultAction(HloInstruction* instruction) {
  return absl::OkStatus();
}

#define RETURN_IF_ALREADY_PROPAGATED(instruction) \
  if (analysis_->HasWeightInfo(instruction)) {    \
    return absl::OkStatus();                      \
  }

absl::Status HloWeightPropagation::HandleTuple(HloInstruction* tuple) {
  RETURN_IF_ALREADY_PROPAGATED(tuple);
  bool has_weight_info = false;
  ShapeTree<WeightInfo> weight_tree(tuple->shape(), WeightInfo::kUnknown);
  for (int64_t idx = 0; idx < tuple->operand_count(); ++idx) {
    const HloInstruction* operand = tuple->operand(idx);
    if (analysis_->IsInstructionWeight(operand)) {
      weight_tree.CopySubtreeFrom(*analysis_->GetWeightInfo(operand), {},
                                  {idx});
      has_weight_info = true;
    }
  }

  if (has_weight_info) {
    TF_RETURN_IF_ERROR(analysis_->SetWeightInfo(tuple, std::move(weight_tree)));
  }

  return absl::OkStatus();
}

absl::Status HloWeightPropagation::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  RETURN_IF_ALREADY_PROPAGATED(get_tuple_element);
  const HloInstruction* operand = get_tuple_element->operand(0);
  if (analysis_->IsInstructionWeight(operand)) {
    ShapeTree<WeightInfo> weight_tree(get_tuple_element->shape(),
                                      WeightInfo::kUnknown);
    weight_tree.CopySubtreeFrom(*analysis_->GetWeightInfo(operand),
                                {get_tuple_element->tuple_index()}, {});
    TF_RETURN_IF_ERROR(
        analysis_->SetWeightInfo(get_tuple_element, std::move(weight_tree)));
  }
  return absl::OkStatus();
}

absl::Status HloWeightPropagation::HandleCall(HloInstruction* call) {
  RETURN_IF_ALREADY_PROPAGATED(call);
  HloComputation* computation = call->called_computations()[0];
  TF_RETURN_IF_ERROR(
      analysis_->RunOnComputation(*computation, call->operands()));
  if (analysis_->IsInstructionWeight(computation->root_instruction())) {
    TF_RETURN_IF_ERROR(analysis_->SetWeightInfo(
        call, *analysis_->GetWeightInfo(computation->root_instruction())));
  }
  return absl::OkStatus();
}

absl::Status HloWeightPropagation::HandleWhile(HloInstruction* xla_while) {
  RETURN_IF_ALREADY_PROPAGATED(xla_while);
  TF_RETURN_IF_ERROR(analysis_->RunOnComputation(*xla_while->while_condition(),
                                                 xla_while->operands()));
  HloComputation* computation = xla_while->while_body();
  TF_RETURN_IF_ERROR(
      analysis_->RunOnComputation(*computation, xla_while->operands()));
  if (analysis_->IsInstructionWeight(computation->root_instruction())) {
    TF_RETURN_IF_ERROR(analysis_->SetWeightInfo(
        xla_while, *analysis_->GetWeightInfo(computation->root_instruction())));
  }
  return absl::OkStatus();
}

// Called for operations that operate on a single operand and do not change
// the weight "nature" of their operand.
absl::Status HloWeightPropagation::HandleSimpleOp(HloInstruction* op) {
  RETURN_IF_ALREADY_PROPAGATED(op);
  const HloInstruction* operand = op->operand(0);
  if (analysis_->IsInstructionWeight(operand)) {
    TF_RETURN_IF_ERROR(analysis_->SetInstructionAsWeight(op));
  }
  return absl::OkStatus();
}

absl::Status HloWeightPropagation::HandleDynamicSlice(
    HloInstruction* dynamic_slice) {
  return HandleSimpleOp(dynamic_slice);
}

absl::Status HloWeightPropagation::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  RETURN_IF_ALREADY_PROPAGATED(dynamic_update_slice);
  // If either the operand or the update is a weight, we consider the output to
  // be a weight.
  const HloInstruction* operand = dynamic_update_slice->operand(0);
  const HloInstruction* update = dynamic_update_slice->operand(1);
  if (analysis_->IsInstructionWeight(operand) ||
      analysis_->IsInstructionWeight(update)) {
    TF_RETURN_IF_ERROR(analysis_->SetInstructionAsWeight(dynamic_update_slice));
  }
  return absl::OkStatus();
}

absl::Status HloWeightPropagation::HandleSlice(HloInstruction* slice) {
  return HandleSimpleOp(slice);
}

absl::Status HloWeightPropagation::HandleConvert(HloInstruction* convert) {
  return HandleSimpleOp(convert);
}

absl::Status HloWeightPropagation::HandleReshape(HloInstruction* reshape) {
  return HandleSimpleOp(reshape);
}

absl::Status HloWeightPropagation::HandleBitcast(HloInstruction* bitcast) {
  return HandleSimpleOp(bitcast);
}

absl::Status HloWeightPropagation::HandleTranspose(HloInstruction* transpose) {
  return HandleSimpleOp(transpose);
}

absl::Status HloWeightPropagation::HandleCopy(HloInstruction* copy) {
  return HandleSimpleOp(copy);
}

absl::Status HloWeightPropagation::HandleBitcastConvert(
    HloInstruction* bitcast_convert) {
  return HandleSimpleOp(bitcast_convert);
}

absl::Status HloWeightPropagation::HandleOptimizationBarrier(
    HloInstruction* optimization_barrier) {
  RETURN_IF_ALREADY_PROPAGATED(optimization_barrier);
  CHECK_EQ(optimization_barrier->operand_count(), 1)
      << "Optimization barrier must have exactly one operand.";
  const HloInstruction* optimization_barrier_operand =
      optimization_barrier->operand(0);
  if (analysis_->IsInstructionWeight(optimization_barrier_operand)) {
    TF_RETURN_IF_ERROR(analysis_->SetWeightInfo(
        optimization_barrier,
        *analysis_->GetWeightInfo(optimization_barrier_operand)));
  }
  return absl::OkStatus();
}

}  // namespace xla
