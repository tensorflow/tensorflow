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
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

static void ClearDotDependent(ShapeTree<DimensionInfo>& dimension_info_tree) {
  dimension_info_tree.ForEachMutableElement(
      [&](const ShapeIndex& index, DimensionInfo* dimension_info) {
        if (dimension_info_tree.IsLeaf(index) &&
            *dimension_info == DimensionInfo::kDotDependent) {
          *dimension_info = DimensionInfo::kUnknown;
        }
      });
}

bool HloDimensionAnalysis::IsWeight(const HloInstruction* instruction) const {
  auto it = info_map_.find(instruction);
  if (it == info_map_.end()) {
    return false;
  }
  return absl::c_any_of(it->second.leaves(),
                        [](const std::pair<ShapeIndex, DimensionInfo>& leaf) {
                          return leaf.second == DimensionInfo::kWeight;
                        });
}

bool HloDimensionAnalysis::IsDotDependent(
    const HloInstruction* instruction) const {
  auto it = info_map_.find(instruction);
  if (it == info_map_.end()) {
    return false;
  }
  return absl::c_any_of(it->second.leaves(),
                        [](const std::pair<ShapeIndex, DimensionInfo>& leaf) {
                          return leaf.second == DimensionInfo::kDotDependent;
                        });
}

bool HloDimensionAnalysis::IsKnownDimensionInfo(
    const HloInstruction* instruction) const {
  auto it = info_map_.find(instruction);
  if (it == info_map_.end()) {
    return false;
  }
  return absl::c_any_of(it->second.leaves(),
                        [](const std::pair<ShapeIndex, DimensionInfo>& leaf) {
                          return leaf.second == DimensionInfo::kWeight ||
                                 leaf.second == DimensionInfo::kDotDependent;
                        });
}

std::optional<ShapeTree<DimensionInfo>> HloDimensionAnalysis::GetDimensionInfo(
    const HloInstruction* instruction) const {
  auto it = info_map_.find(instruction);
  if (it == info_map_.end()) {
    return std::nullopt;
  }
  return it->second;
}

absl::Status HloDimensionAnalysis::SetDimensionInfo(
    const HloInstruction* instruction, DimensionInfo value) {
  CHECK(value == DimensionInfo::kWeight ||
        value == DimensionInfo::kDotDependent)
      << "Unsupported dimension info: " << value;
  auto [it, success] = info_map_.emplace(
      std::piecewise_construct, std::forward_as_tuple(instruction),
      std::forward_as_tuple(instruction->shape(), DimensionInfo::kUnknown));

  if (!success) {
    return absl::InternalError(absl::StrCat("Instruction ",
                                            instruction->ToString(),
                                            " already has dimension info."));
  }

  ShapeTree<DimensionInfo>& dim_info_tree = it->second;
  dim_info_tree.ForEachMutableElement(
      [&](const ShapeIndex& index, DimensionInfo* dimension_info) {
        if (dim_info_tree.IsLeaf(index)) {
          *dimension_info = value;
          return;
        }
        *dimension_info = DimensionInfo::kUnknown;
      });
  return absl::OkStatus();
}

absl::Status HloDimensionAnalysis::SetDimensionInfo(
    const HloInstruction* target, ShapeTree<DimensionInfo> annotation) {
  auto [it, success] = info_map_.emplace(target, std::move(annotation));
  if (!success) {
    return absl::InternalError(absl::StrCat("Instruction ", target->ToString(),
                                            " already has dimension info."));
  }
  return absl::OkStatus();
}

// Annotates the entry computation parameters as weights.
absl::Status HloDimensionAnalysis::AnnotateEntryComputationParameters(
    const HloModule& module) {
  const auto& params = module.entry_computation()->parameter_instructions();
  info_map_.reserve(params.size());
  for (HloInstruction* instruction : params) {
    TF_RETURN_IF_ERROR(SetDimensionInfo(instruction, DimensionInfo::kWeight));
  }
  return absl::OkStatus();
}

bool HloDimensionAnalysis::IsDotOrHasDotDependent(
    const HloInstruction* op) const {
  if (HloPredicateIsOp<HloOpcode::kDot, HloOpcode::kConvolution,
                       HloOpcode::kRaggedDot>(op)) {
    return true;
  }
  return IsDotDependent(op);
}

absl::StatusOr<std::unique_ptr<HloDimensionAnalysis>> HloDimensionAnalysis::Run(
    const HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<HloDimensionAnalysis> analysis =
      absl::WrapUnique(new HloDimensionAnalysis(module, execution_threads));
  TF_RETURN_IF_ERROR(analysis->AnnotateEntryComputationParameters(module));
  TF_RETURN_IF_ERROR(analysis->RunOnComputation(*module.entry_computation()));
  return analysis;
}

absl::Status HloDimensionAnalysis::RunOnComputation(
    const HloComputation& computation) {
  if (HloInstruction::IsThreadIncluded(computation.execution_thread(),
                                       execution_threads_)) {
    HloDimensionInfoPropagation propagation(this);
    return propagation.Run(computation);
  }
  return absl::OkStatus();
}

absl::Status HloDimensionAnalysis::RunOnComputation(
    const HloComputation& computation,
    absl::Span<const HloInstruction* const> operands) {
  CHECK_EQ(computation.num_parameters(), operands.size());
  for (int i = 0; i < computation.num_parameters(); ++i) {
    auto dimension_info_iter = info_map_.find(operands[i]);
    if (dimension_info_iter == info_map_.end()) {
      continue;
    }
    ClearDotDependent(dimension_info_iter->second);
    TF_RETURN_IF_ERROR(SetDimensionInfo(computation.parameter_instructions()[i],
                                        dimension_info_iter->second));
  }
  return RunOnComputation(computation);
}

absl::Status HloDimensionInfoPropagation::Run(
    const HloComputation& computation) {
  TF_RETURN_IF_ERROR(computation.root_instruction()->Accept(this));
  for (HloInstruction* instruction : computation.instructions()) {
    if (instruction->user_count() == 0) {
      TF_RETURN_IF_ERROR(instruction->Accept(this));
    }
  }
  return absl::OkStatus();
}

#define RETURN_IF_ALREADY_PROPAGATED(instruction) \
  if (analysis_->HasDimensionInfo(instruction)) { \
    return absl::OkStatus();                      \
  }

absl::Status HloDimensionInfoPropagation::DefaultAction(
    HloInstruction* instruction) {
  RETURN_IF_ALREADY_PROPAGATED(instruction);
  // For non-weight, we want to find out whether the instruction has a
  // dot-dependent operand.
  for (const HloInstruction* operand : instruction->operands()) {
    if (analysis_->IsDotOrHasDotDependent(operand)) {
      TF_RETURN_IF_ERROR(analysis_->SetDimensionInfo(
          instruction, ShapeTree<DimensionInfo>(instruction->shape(),
                                                DimensionInfo::kDotDependent)));
      break;
    }
  }
  return absl::OkStatus();
}

absl::Status HloDimensionInfoPropagation::HandleTuple(HloInstruction* tuple) {
  RETURN_IF_ALREADY_PROPAGATED(tuple);
  bool has_dim_info = false;
  ShapeTree<DimensionInfo> dim_info_tree(tuple->shape(),
                                         DimensionInfo::kUnknown);
  for (int64_t idx = 0; idx < tuple->operand_count(); ++idx) {
    const HloInstruction* operand = tuple->operand(idx);
    if (analysis_->IsKnownDimensionInfo(operand)) {
      dim_info_tree.CopySubtreeFrom(*analysis_->GetDimensionInfo(operand), {},
                                    {idx});
      has_dim_info = true;
    }
  }

  if (has_dim_info) {
    TF_RETURN_IF_ERROR(
        analysis_->SetDimensionInfo(tuple, std::move(dim_info_tree)));
  }

  return absl::OkStatus();
}

absl::Status HloDimensionInfoPropagation::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  RETURN_IF_ALREADY_PROPAGATED(get_tuple_element);
  const HloInstruction* operand = get_tuple_element->operand(0);
  if (analysis_->IsKnownDimensionInfo(operand)) {
    ShapeTree<DimensionInfo> dimension_info(get_tuple_element->shape(),
                                            DimensionInfo::kUnknown);
    dimension_info.CopySubtreeFrom(*analysis_->GetDimensionInfo(operand),
                                   {get_tuple_element->tuple_index()}, {});
    TF_RETURN_IF_ERROR(analysis_->SetDimensionInfo(get_tuple_element,
                                                   std::move(dimension_info)));
  }
  return absl::OkStatus();
}

absl::Status HloDimensionInfoPropagation::HandleCall(HloInstruction* call) {
  RETURN_IF_ALREADY_PROPAGATED(call);
  HloComputation* computation = call->called_computations()[0];
  if (computation->caller_instructions().size() > 1) {
    return absl::UnimplementedError(
        "Call with multiple callers is not supported.");
  }
  TF_RETURN_IF_ERROR(
      analysis_->RunOnComputation(*computation, call->operands()));
  if (analysis_->IsWeight(computation->root_instruction())) {
    ShapeTree<DimensionInfo> dimension_info_tree =
        *analysis_->GetDimensionInfo(computation->root_instruction());
    ClearDotDependent(dimension_info_tree);
    TF_RETURN_IF_ERROR(analysis_->SetDimensionInfo(call, dimension_info_tree));
  }
  return absl::OkStatus();
}

absl::Status HloDimensionInfoPropagation::HandleWhile(
    HloInstruction* xla_while) {
  RETURN_IF_ALREADY_PROPAGATED(xla_while);
  TF_RETURN_IF_ERROR(analysis_->RunOnComputation(*xla_while->while_condition(),
                                                 xla_while->operands()));
  HloComputation* computation = xla_while->while_body();
  TF_RETURN_IF_ERROR(
      analysis_->RunOnComputation(*computation, xla_while->operands()));
  if (analysis_->IsWeight(computation->root_instruction())) {
    ShapeTree<DimensionInfo> dimension_info_tree =
        *analysis_->GetDimensionInfo(computation->root_instruction());
    ClearDotDependent(dimension_info_tree);
    TF_RETURN_IF_ERROR(
        analysis_->SetDimensionInfo(xla_while, dimension_info_tree));
  }
  return absl::OkStatus();
}

// Called for operations that operate on a single operand and do not change
// the weight "nature" of their operand.
absl::Status HloDimensionInfoPropagation::HandleSimpleOp(HloInstruction* op) {
  RETURN_IF_ALREADY_PROPAGATED(op);
  const HloInstruction* operand = op->operand(0);
  if (analysis_->IsWeight(operand)) {
    TF_RETURN_IF_ERROR(analysis_->SetDimensionInfo(op, DimensionInfo::kWeight));
  } else if (analysis_->IsDotOrHasDotDependent(operand)) {
    TF_RETURN_IF_ERROR(
        analysis_->SetDimensionInfo(op, DimensionInfo::kDotDependent));
  }
  return absl::OkStatus();
}

absl::Status HloDimensionInfoPropagation::HandleDynamicSlice(
    HloInstruction* dynamic_slice) {
  return HandleSimpleOp(dynamic_slice);
}

absl::Status HloDimensionInfoPropagation::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  RETURN_IF_ALREADY_PROPAGATED(dynamic_update_slice);
  // If either the operand or the update is a weight, we consider the output to
  // be a weight.
  const HloInstruction* operand = dynamic_update_slice->operand(0);
  const HloInstruction* update = dynamic_update_slice->operand(1);
  if (analysis_->IsWeight(operand) || analysis_->IsWeight(update)) {
    TF_RETURN_IF_ERROR(analysis_->SetDimensionInfo(dynamic_update_slice,
                                                   DimensionInfo::kWeight));
  } else if (analysis_->IsDotDependent(operand) ||
             analysis_->IsDotDependent(update)) {
    TF_RETURN_IF_ERROR(analysis_->SetDimensionInfo(
        dynamic_update_slice, DimensionInfo::kDotDependent));
  }
  return absl::OkStatus();
}

absl::Status HloDimensionInfoPropagation::HandleSlice(HloInstruction* slice) {
  return HandleSimpleOp(slice);
}

absl::Status HloDimensionInfoPropagation::HandleConvert(
    HloInstruction* convert) {
  return HandleSimpleOp(convert);
}

absl::Status HloDimensionInfoPropagation::HandleReshape(
    HloInstruction* reshape) {
  return HandleSimpleOp(reshape);
}

absl::Status HloDimensionInfoPropagation::HandleBitcast(
    HloInstruction* bitcast) {
  return HandleSimpleOp(bitcast);
}

absl::Status HloDimensionInfoPropagation::HandleTranspose(
    HloInstruction* transpose) {
  return HandleSimpleOp(transpose);
}

absl::Status HloDimensionInfoPropagation::HandleCopy(HloInstruction* copy) {
  return HandleSimpleOp(copy);
}

absl::Status HloDimensionInfoPropagation::HandleBitcastConvert(
    HloInstruction* bitcast_convert) {
  return HandleSimpleOp(bitcast_convert);
}

absl::Status HloDimensionInfoPropagation::HandleOptimizationBarrier(
    HloInstruction* optimization_barrier) {
  RETURN_IF_ALREADY_PROPAGATED(optimization_barrier);
  CHECK_EQ(optimization_barrier->operand_count(), 1)
      << "Optimization barrier must have exactly one operand.";
  const HloInstruction* optimization_barrier_operand =
      optimization_barrier->operand(0);
  if (analysis_->IsKnownDimensionInfo(optimization_barrier_operand)) {
    TF_RETURN_IF_ERROR(analysis_->SetDimensionInfo(
        optimization_barrier,
        *analysis_->GetDimensionInfo(optimization_barrier_operand)));
  }
  return absl::OkStatus();
}

absl::Status HloDimensionInfoPropagation::HandleAllGather(
    HloInstruction* all_gather) {
  return HandleSimpleOp(all_gather);
}

}  // namespace xla
