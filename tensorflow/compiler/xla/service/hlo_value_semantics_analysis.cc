/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.


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

#include "tensorflow/compiler/xla/service/hlo_value_semantics_analysis.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_value.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {

std::string HloValueSemanticLabelToString(HloValueSemanticLabel label) {
  switch (label) {
    case HloValueSemanticLabel::kStatic:
      return "Static";
    case HloValueSemanticLabel::kRandom:
      return "Random";
    case HloValueSemanticLabel::kWeight:
      return "Weight";
    case HloValueSemanticLabel::kActivation:
      return "Activation";
    case HloValueSemanticLabel::kActivationGradient:
      return "ActivationGradient";
    case HloValueSemanticLabel::kWeightGradient:
      return "WeightGradient";
    case HloValueSemanticLabel::kTupleOrToken:
      return "TupleOrToken";
  }
}

std::string HloValueSemantics::ToString() const {
  std::string content = absl::StrJoin(
      {absl::StrCat("label: ", HloValueSemanticLabelToString(label_)),
       absl::StrCat("origin: ", origin_.ToString())},
      ", ");
  return absl::StrCat("{", content, "}");
}

HloValueSemantics::HloValueSemantics(HloValueSemanticLabel label,
                                     const HloPosition& origin)
    : HloValueSemantics(0, label, origin) {}

HloValueSemantics::HloValueSemantics(Id id, HloValueSemanticLabel label,
                                     const HloPosition& origin)
    : id_(id), label_(label), origin_(origin) {}

HloValueSemanticsAnalysis::HloValueSemanticsAnalysis(const HloModule& module)
    : module_(module), next_id_(0) {}

const HloValueSemantics* HloValueSemanticsAnalysis::GetSemantics(
    const HloInstruction* instruction, const ShapeIndex& index) const {
  return GetInstructionSemantics(instruction).element(index);
}

StatusOr<std::unique_ptr<HloValueSemanticsAnalysis>>
HloValueSemanticsAnalysis::Run(const HloModule& module) {
  std::unique_ptr<HloValueSemanticsAnalysis> value_semantics_analysis =
      absl::WrapUnique(new HloValueSemanticsAnalysis(module));
  value_semantics_analysis->AnnotateWeights();
  TF_RETURN_IF_ERROR(
      value_semantics_analysis->RunOnComputation(*module.entry_computation()));
  return value_semantics_analysis;
}

HloValueSemantics::Id HloValueSemanticsAnalysis::NextId() { return next_id_++; }

const HloValueSemantics* HloValueSemanticsAnalysis::NewHloValueSemantics(
    HloValueSemanticLabel label, const HloPosition& origin) {
  HloValueSemantics::Id id = NextId();
  auto inserted = value_semantics_map_.insert(std::make_pair(
      id, std::make_unique<HloValueSemantics>(id, label, origin)));
  return inserted.first->second.get();
}

const ShapeTree<const HloValueSemantics*>&
HloValueSemanticsAnalysis::GetInstructionSemantics(
    const HloInstruction* instruction) const {
  auto semantics_iter = value_semantics_.find(instruction);
  CHECK(semantics_iter != value_semantics_.end())
      << "instruction: " << instruction->ToString();
  return semantics_iter->second;
}

void HloValueSemanticsAnalysis::DeepCopyHloValueSemantics(
    ShapeTree<const HloValueSemantics*>& copy_to,
    const ShapeTree<const HloValueSemantics*>& copy_from,
    const ShapeIndex& source_index, const ShapeIndex& destination_index) {
  copy_to.ForEachMutableElement(
      [this, &copy_from, &source_index, &destination_index](
          const ShapeIndex& index, const HloValueSemantics** semantics) {
        if (index.size() < destination_index.size()) {
          return;
        }
        bool in_subtree_to_copy = true;
        for (int i = 0; i < destination_index.size(); ++i) {
          if (index[i] != destination_index[i]) {
            in_subtree_to_copy = false;
            break;
          }
        }
        if (!in_subtree_to_copy) {
          return;
        }
        ShapeIndex full_source_index = source_index;
        for (int i = destination_index.size(); i < index.size(); ++i) {
          full_source_index.push_back(index[i]);
        }
        const HloValueSemantics* source_semantics =
            copy_from.element(full_source_index);
        *semantics = NewHloValueSemantics(source_semantics->label(),
                                          source_semantics->origin());
      });
}

void HloValueSemanticsAnalysis::DeepCopyHloValueSemantics(
    const HloInstruction* target,
    const ShapeTree<const HloValueSemantics*>& copy_from,
    const ShapeIndex& source_index) {
  auto semantics_iter = value_semantics_.find(target);
  if (semantics_iter != value_semantics_.end()) {
    DeleteHloValueSemantics(semantics_iter->second);
    DeepCopyHloValueSemantics(semantics_iter->second, copy_from, source_index,
                              {});
    return;
  }
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(target->shape(),
                                                           nullptr);
  DeepCopyHloValueSemantics(semantics_shape_tree, copy_from, source_index, {});
  value_semantics_[target] = std::move(semantics_shape_tree);
}

void HloValueSemanticsAnalysis::SetHloValueSemantics(
    const HloInstruction* target,
    const ShapeTree<const HloValueSemantics*>& semantics) {
  auto semantics_iter = value_semantics_.find(target);
  if (semantics_iter != value_semantics_.end()) {
    DeleteHloValueSemantics(semantics_iter->second);
  }
  value_semantics_[target] = semantics;
}

void HloValueSemanticsAnalysis::DeleteHloValueSemantics(
    const HloValueSemantics* to_delete) {
  value_semantics_map_.erase(to_delete->id());
}

void HloValueSemanticsAnalysis::DeleteHloValueSemantics(
    const ShapeTree<const HloValueSemantics*>& to_delete) {
  to_delete.ForEachElement(
      [this](const ShapeIndex& index, const HloValueSemantics* semantics) {
        DeleteHloValueSemantics(semantics);
      });
}

void HloValueSemanticsAnalysis::AnnotateWeights() {
  const HloComputation* entry_computation = module_.entry_computation();
  for (HloInstruction* parameter :
       entry_computation->parameter_instructions()) {
    ShapeTree<const HloValueSemantics*> semantics_shape_tree(parameter->shape(),
                                                             nullptr);
    semantics_shape_tree.ForEachMutableElement(
        [this, &semantics_shape_tree, parameter](
            const ShapeIndex& index, const HloValueSemantics** semantics) {
          if (!semantics_shape_tree.IsLeaf(index)) {
            *semantics = NewHloValueSemantics(
                HloValueSemanticLabel::kTupleOrToken, {parameter, index});
          }
          *semantics = NewHloValueSemantics(HloValueSemanticLabel::kWeight,
                                            {parameter, index});
        });
    value_semantics_[parameter] = std::move(semantics_shape_tree);
  }
}
Status HloValueSemanticsAnalysis::RunOnComputation(
    const HloComputation& computation,
    absl::Span<const HloInstruction* const> operands) {
  CHECK_EQ(computation.num_parameters(), operands.size());
  for (int i = 0; i < computation.num_parameters(); ++i) {
    auto semantics_iter = value_semantics_.find(operands[i]);
    CHECK(semantics_iter != value_semantics_.end());
    DeepCopyHloValueSemantics(computation.parameter_instructions()[i],
                              semantics_iter->second);
  }
  return RunOnComputation(computation);
}

Status HloValueSemanticsAnalysis::RunOnComputation(
    const HloComputation& computation) {
  HloValueSemanticsPropagation propagation(this);
  return propagation.Run(computation);
}

HloValueSemanticsPropagation::HloValueSemanticsPropagation(
    HloValueSemanticsAnalysis* analysis)
    : analysis_(analysis) {}

Status HloValueSemanticsPropagation::Run(const HloComputation& computation) {
  return computation.root_instruction()->Accept(this);
}

HloValueSemantics HloValueSemanticsPropagation::CopySemantics(
    const HloValueSemantics& semantics) {
  return HloValueSemantics(semantics.label(), semantics.origin());
}

HloValueSemantics HloValueSemanticsPropagation::CopySemanticsWithNewOrigin(
    const HloValueSemantics& semantics, HloInstruction* new_origin,
    const ShapeIndex& index) {
  return HloValueSemantics(semantics.label(), {new_origin, index});
}

const HloValueSemantics* HloValueSemanticsPropagation::AddSemantics(
    const HloValueSemantics& semantics) {
  return analysis_->NewHloValueSemantics(semantics.label(), semantics.origin());
}

bool HloValueSemanticsPropagation::IsActivationOriginDependentOn(
    const HloValueSemantics& activation_semantics,
    const HloPosition& origin_dependence, bool recursive) const {
  CHECK(activation_semantics.label() == HloValueSemanticLabel::kActivation);
  std::vector<HloPosition> stack;
  absl::flat_hash_set<HloPosition> visited;
  stack.push_back(activation_semantics.origin());
  while (!stack.empty()) {
    HloPosition origin = stack.back();
    stack.pop_back();
    if (visited.contains(origin)) {
      continue;
    }
    visited.insert(origin);
    absl::Span<const HloInstruction* const> operands =
        origin.instruction->operands();
    // Do not check slice indices.
    if (origin.instruction->opcode() == HloOpcode::kDynamicUpdateSlice) {
      operands = operands.subspan(0, 2);
    }
    if (origin.instruction->opcode() == HloOpcode::kDynamicSlice) {
      operands = operands.subspan(0, 1);
    }
    for (const HloInstruction* origin_operand : operands) {
      const HloValueSemantics* origin_operand_semantics =
          analysis_->GetSemantics(origin_operand);
      if (origin_operand_semantics->origin() == origin_dependence) {
        return true;
      }
      if (recursive) {
        stack.push_back(origin_operand_semantics->origin());
      }
    }
  }
  return false;
}

StatusOr<HloValueSemantics>
HloValueSemanticsPropagation::ComputeSemanticsFromStaticAndOther(
    const HloValueSemantics& static_semantics,
    const HloValueSemantics& other_semantics, HloInstruction* instruction) {
  CHECK(static_semantics.label() == HloValueSemanticLabel::kStatic)
      << __func__ << ", : " << static_semantics.ToString();
  if (other_semantics.label() == HloValueSemanticLabel::kStatic) {
    return CopySemanticsWithNewOrigin(other_semantics, instruction);
  }
  return CopySemantics(other_semantics);
}

StatusOr<HloValueSemantics>
HloValueSemanticsPropagation::ComputeSemanticsFromRandomAndOther(
    const HloValueSemantics& random_semantics,
    const HloValueSemantics& other_semantics, HloInstruction* instruction) {
  CHECK(random_semantics.label() == HloValueSemanticLabel::kRandom);
  CHECK(other_semantics.label() != HloValueSemanticLabel::kStatic);
  if (other_semantics.label() == HloValueSemanticLabel::kRandom) {
    return CopySemanticsWithNewOrigin(other_semantics, instruction);
  }
  return CopySemantics(other_semantics);
}

StatusOr<HloValueSemantics>
HloValueSemanticsPropagation::ComputeSemanticsFromWeightAndOther(
    const HloValueSemantics& weight_semantics,
    const HloValueSemantics& other_semantics, HloInstruction* instruction) {
  CHECK(weight_semantics.label() == HloValueSemanticLabel::kWeight);
  CHECK(other_semantics.label() != HloValueSemanticLabel::kStatic &&
        other_semantics.label() != HloValueSemanticLabel::kRandom);
  bool is_dot_or_convolution = instruction->opcode() == HloOpcode::kDot ||
                               instruction->opcode() == HloOpcode::kConvolution;
  if (other_semantics.label() == HloValueSemanticLabel::kWeight) {
    if (!is_dot_or_convolution) {
      return CopySemanticsWithNewOrigin(other_semantics, instruction);
    }
    return HloValueSemantics(HloValueSemanticLabel::kActivation,
                             {instruction, {}});
  }
  if (!is_dot_or_convolution) {
    return CopySemantics(other_semantics);
  }
  if (other_semantics.label() == HloValueSemanticLabel::kActivation) {
    if (IsActivationOriginDependentOn(other_semantics,
                                      weight_semantics.origin(),
                                      /*recursive=*/true)) {
      return HloValueSemantics(HloValueSemanticLabel::kActivationGradient,
                               {instruction, {}});
    }
    return CopySemanticsWithNewOrigin(other_semantics, instruction);
  }
  if (other_semantics.label() == HloValueSemanticLabel::kActivationGradient) {
    return CopySemanticsWithNewOrigin(other_semantics, instruction);
  }
  CHECK(other_semantics.label() == HloValueSemanticLabel::kWeightGradient);
  return CopySemantics(other_semantics);
}

StatusOr<HloValueSemantics>
HloValueSemanticsPropagation::ComputeSemanticsFromActivationAndOther(
    const HloValueSemantics& activation_semantics,
    const HloValueSemantics& other_semantics, HloInstruction* instruction) {
  CHECK(activation_semantics.label() == HloValueSemanticLabel::kActivation);
  CHECK(other_semantics.label() != HloValueSemanticLabel::kStatic &&
        other_semantics.label() != HloValueSemanticLabel::kRandom &&
        other_semantics.label() != HloValueSemanticLabel::kWeight);
  bool is_dot_or_convolution = instruction->opcode() == HloOpcode::kDot ||
                               instruction->opcode() == HloOpcode::kConvolution;
  if (!is_dot_or_convolution) {
    return CopySemanticsWithNewOrigin(other_semantics, instruction);
  }
  if (other_semantics.label() == HloValueSemanticLabel::kActivation) {
    return CopySemanticsWithNewOrigin(other_semantics, instruction);
  }
  if (other_semantics.label() == HloValueSemanticLabel::kActivationGradient) {
    return HloValueSemantics(HloValueSemanticLabel::kWeightGradient,
                             {instruction, {}});
  }
  CHECK(other_semantics.label() == HloValueSemanticLabel::kWeightGradient)
      << "instruction:  " << instruction->ToString()
      << ", semantics: " << other_semantics.ToString()
      << ", expected: WeightGradient.";

  return CopySemantics(other_semantics);
}

StatusOr<HloValueSemantics>
HloValueSemanticsPropagation::ComputeSemanticsFromActivationGradientAndOther(
    const HloValueSemantics& activation_gradient_semantics,
    const HloValueSemantics& other_semantics, HloInstruction* instruction) {
  CHECK(activation_gradient_semantics.label() ==
        HloValueSemanticLabel::kActivationGradient);
  CHECK(other_semantics.label() != HloValueSemanticLabel::kStatic &&
        other_semantics.label() != HloValueSemanticLabel::kRandom &&
        other_semantics.label() != HloValueSemanticLabel::kWeight &&
        other_semantics.label() != HloValueSemanticLabel::kActivation);
  if (other_semantics.label() == HloValueSemanticLabel::kActivationGradient) {
    return CopySemanticsWithNewOrigin(other_semantics, instruction);
  }

  CHECK(other_semantics.label() == HloValueSemanticLabel::kWeightGradient);
  return CopySemantics(other_semantics);
}

StatusOr<HloValueSemantics>
HloValueSemanticsPropagation::ComputeSemanticsFromWeightGradientAndOther(
    const HloValueSemantics& weight_gradient_semantics,
    const HloValueSemantics& other_semantics, HloInstruction* instruction) {
  CHECK(weight_gradient_semantics.label() ==
        HloValueSemanticLabel::kWeightGradient);
  CHECK(other_semantics.label() != HloValueSemanticLabel::kStatic &&
        other_semantics.label() != HloValueSemanticLabel::kRandom &&
        other_semantics.label() != HloValueSemanticLabel::kWeight &&
        other_semantics.label() != HloValueSemanticLabel::kActivation &&
        other_semantics.label() != HloValueSemanticLabel::kActivationGradient);
  return CopySemantics(weight_gradient_semantics);
}

StatusOr<HloValueSemantics>
HloValueSemanticsPropagation::ComputeSemanticsFromOperands(
    HloInstruction* instruction, absl::Span<const int64_t> operand_indices,
    absl::Span<const ShapeIndex> operand_shape_indices) {
  CHECK(!operand_indices.empty());
  CHECK(operand_shape_indices.empty() ||
        operand_indices.size() == operand_shape_indices.size());
  std::vector<HloValueSemantics> semantics_vec;
  for (int64_t operand_index : operand_indices) {
    const HloInstruction* operand = instruction->operand(operand_index);
    const HloValueSemantics* operand_semantics = analysis_->GetSemantics(
        operand, operand_shape_indices.empty()
                     ? ShapeIndex()
                     : operand_shape_indices[operand_index]);
    semantics_vec.push_back(*operand_semantics);
  }
  while (semantics_vec.size() >= 2) {
    absl::Span<const HloValueSemantics> operand_list =
        absl::MakeConstSpan(semantics_vec).subspan(semantics_vec.size() - 2, 2);
    auto find_operand_index_with_label =
        [&operand_list](
            HloValueSemanticLabel label) -> std::optional<int64_t> {
      auto iter = absl::c_find_if(operand_list,
                                  [label](const HloValueSemantics& operand) {
                                    return operand.label() == label;
                                  });
      return (iter != operand_list.end())
                 ? std::optional<int64_t>(
                       std::distance(operand_list.begin(), iter))
                 : std::nullopt;
    };
    auto replace_operands_semantics_with =
        [&semantics_vec](const HloValueSemantics& result_semantics) {
          semantics_vec.pop_back();
          semantics_vec.pop_back();
          semantics_vec.push_back(result_semantics);
        };
    if (auto index =
            find_operand_index_with_label(HloValueSemanticLabel::kStatic)) {
      TF_ASSIGN_OR_RETURN(
          HloValueSemantics semantics,
          ComputeSemanticsFromStaticAndOther(
              operand_list[*index], operand_list[1 - *index], instruction));
      replace_operands_semantics_with(semantics);
      continue;
    }
    if (auto index =
            find_operand_index_with_label(HloValueSemanticLabel::kRandom)) {
      TF_ASSIGN_OR_RETURN(
          HloValueSemantics semantics,
          ComputeSemanticsFromRandomAndOther(
              operand_list[*index], operand_list[1 - *index], instruction));
      replace_operands_semantics_with(semantics);
      continue;
    }
    if (auto index =
            find_operand_index_with_label(HloValueSemanticLabel::kWeight)) {
      TF_ASSIGN_OR_RETURN(
          HloValueSemantics semantics,
          ComputeSemanticsFromWeightAndOther(
              operand_list[*index], operand_list[1 - *index], instruction));
      replace_operands_semantics_with(semantics);
      continue;
    }
    if (auto index =
            find_operand_index_with_label(HloValueSemanticLabel::kActivation)) {
      TF_ASSIGN_OR_RETURN(
          HloValueSemantics semantics,
          ComputeSemanticsFromActivationAndOther(
              operand_list[*index], operand_list[1 - *index], instruction));
      replace_operands_semantics_with(semantics);
      continue;
    }
    if (auto index = find_operand_index_with_label(
            HloValueSemanticLabel::kActivationGradient)) {
      TF_ASSIGN_OR_RETURN(
          HloValueSemantics semantics,
          ComputeSemanticsFromActivationGradientAndOther(
              operand_list[*index], operand_list[1 - *index], instruction));
      replace_operands_semantics_with(semantics);
      continue;
    }
    if (auto index = find_operand_index_with_label(
            HloValueSemanticLabel::kWeightGradient)) {
      TF_ASSIGN_OR_RETURN(
          HloValueSemantics semantics,
          ComputeSemanticsFromWeightGradientAndOther(
              operand_list[*index], operand_list[1 - *index], instruction));
      replace_operands_semantics_with(semantics);
      continue;
    }
    LOG(FATAL) << "We don't expect to handle operands of label "
               << HloValueSemanticLabelToString(operand_list[0].label())
               << " and "
               << HloValueSemanticLabelToString(operand_list[1].label())
               << " in ComputeSemanticsFromOperands. Instruction: "
               << instruction->name()
               << " should be handled in its own handler instead of the "
                  "default handler.";
  }
  return semantics_vec.back();
}

Status HloValueSemanticsPropagation::DefaultAction(
    HloInstruction* instruction) {
  std::vector<int64_t> operand_indices(instruction->operand_count());
  std::iota(operand_indices.begin(), operand_indices.end(), 0);
  TF_ASSIGN_OR_RETURN(
      HloValueSemantics semantics,
      ComputeSemanticsFromOperands(instruction, operand_indices));
  const HloValueSemantics* semantics_ptr = AddSemantics(semantics);
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(instruction->shape(),
                                                           semantics_ptr);
  analysis_->SetHloValueSemantics(instruction, semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleParameter(
    HloInstruction* parameter) {
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleConstant(HloInstruction* constant) {
  const HloValueSemantics* constant_semantics = analysis_->NewHloValueSemantics(
      HloValueSemanticLabel::kStatic, {constant, {}});
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(constant->shape(),
                                                           constant_semantics);
  analysis_->SetHloValueSemantics(constant, semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleIota(HloInstruction* iota) {
  const HloValueSemantics* semantics = analysis_->NewHloValueSemantics(
      HloValueSemanticLabel::kStatic, {iota, {}});
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(iota->shape(),
                                                           semantics);
  analysis_->SetHloValueSemantics(iota, semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandlePartitionId(
    HloInstruction* partition_id) {
  const HloValueSemantics* semantics = analysis_->NewHloValueSemantics(
      HloValueSemanticLabel::kStatic, {partition_id, {}});
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(
      partition_id->shape(), semantics);
  analysis_->SetHloValueSemantics(partition_id, semantics_shape_tree);
  return OkStatus();
}
Status HloValueSemanticsPropagation::HandleReplicaId(
    HloInstruction* replica_id) {
  const HloValueSemantics* semantics = analysis_->NewHloValueSemantics(
      HloValueSemanticLabel::kStatic, {replica_id, {}});
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(replica_id->shape(),
                                                           semantics);
  analysis_->SetHloValueSemantics(replica_id, semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleClamp(HloInstruction* clamp) {
  const ShapeTree<const HloValueSemantics*>& operand_semantics =
      analysis_->GetInstructionSemantics(clamp->operand(1));
  analysis_->DeepCopyHloValueSemantics(clamp, operand_semantics);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleTuple(HloInstruction* tuple) {
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(tuple->shape(),
                                                           nullptr);
  for (int operand_index = 0; operand_index < tuple->operand_count();
       ++operand_index) {
    const HloInstruction* operand = tuple->operand(operand_index);
    const ShapeTree<const HloValueSemantics*>& operand_semantics =
        analysis_->GetInstructionSemantics(operand);
    analysis_->DeepCopyHloValueSemantics(
        semantics_shape_tree, operand_semantics, {}, {operand_index});
  }
  semantics_shape_tree.ForEachMutableElement(
      [tuple, this](const ShapeIndex& index,
                    const HloValueSemantics** semantics) {
        if (index.empty()) {
          *semantics = analysis_->NewHloValueSemantics(
              HloValueSemanticLabel::kTupleOrToken, {tuple, {}});
          return;
        }
      });
  analysis_->SetHloValueSemantics(tuple, semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  const HloInstruction* tuple = get_tuple_element->operand(0);
  int64_t tuple_index = get_tuple_element->tuple_index();
  const ShapeTree<const HloValueSemantics*>& tuple_semantics =
      analysis_->GetInstructionSemantics(tuple);
  TF_ASSIGN_OR_RETURN(
      ShapeTree<const HloValueSemantics*> tuple_element_semantics,
      tuple_semantics.SubShapeTree({tuple_index}));
  analysis_->DeepCopyHloValueSemantics(get_tuple_element,
                                       tuple_element_semantics);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleCall(HloInstruction* call) {
  HloComputation* computation = call->called_computations()[0];
  TF_RETURN_IF_ERROR(
      analysis_->RunOnComputation(*computation, call->operands()));
  const ShapeTree<const HloValueSemantics*>& root_semantics =
      analysis_->GetInstructionSemantics(computation->root_instruction());
  analysis_->DeepCopyHloValueSemantics(call, root_semantics);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleFusion(HloInstruction* fusion) {
  HloComputation* computation = fusion->called_computations()[0];
  TF_RETURN_IF_ERROR(
      analysis_->RunOnComputation(*computation, fusion->operands()));
  const ShapeTree<const HloValueSemantics*>& root_semantics =
      analysis_->GetInstructionSemantics(computation->root_instruction());
  analysis_->DeepCopyHloValueSemantics(fusion, root_semantics);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleWhile(HloInstruction* xla_while) {
  TF_RETURN_IF_ERROR(analysis_->RunOnComputation(*xla_while->while_condition(),
                                                 xla_while->operands()));
  HloComputation* computation = xla_while->while_body();
  TF_RETURN_IF_ERROR(
      analysis_->RunOnComputation(*computation, xla_while->operands()));
  const ShapeTree<const HloValueSemantics*>& root_semantics =
      analysis_->GetInstructionSemantics(computation->root_instruction());
  analysis_->DeepCopyHloValueSemantics(xla_while, root_semantics);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleCustomCall(
    HloInstruction* custom_call) {
  return Unimplemented("Unimplemented custom-call: %s",
                       custom_call->custom_call_target());
}

Status HloValueSemanticsPropagation::HandleConditional(
    HloInstruction* conditional) {
  for (int i = 0; i < conditional->called_computations().size(); ++i) {
    TF_RETURN_IF_ERROR(analysis_->RunOnComputation(
        *conditional->called_computations()[i], conditional->operands()));
  }
  HloComputation* computation = conditional->called_computations()[0];
  const ShapeTree<const HloValueSemantics*>& root_semantics =
      analysis_->GetInstructionSemantics(computation->root_instruction());
  analysis_->DeepCopyHloValueSemantics(conditional, root_semantics);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleSelect(HloInstruction* select) {
  TF_ASSIGN_OR_RETURN(HloValueSemantics semantics,
                      ComputeSemanticsFromOperands(select, {1, 2}));
  const HloValueSemantics* semantics_ptr = AddSemantics(semantics);
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(select->shape(),
                                                           semantics_ptr);
  analysis_->SetHloValueSemantics(select, semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleConcatenate(
    HloInstruction* concatenate) {
  const ShapeTree<const HloValueSemantics*>& operand_semantics =
      analysis_->GetInstructionSemantics(concatenate->operand(0));
  analysis_->DeepCopyHloValueSemantics(concatenate, operand_semantics);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleDynamicSlice(
    HloInstruction* dynamic_slice) {
  const HloInstruction* dynamic_slice_operand = dynamic_slice->operand(0);
  const HloValueSemantics* operand_semantics =
      analysis_->GetSemantics(dynamic_slice_operand);
  const HloValueSemantics* semantics = nullptr;
  if (operand_semantics->label() == HloValueSemanticLabel::kStatic ||
      operand_semantics->label() == HloValueSemanticLabel::kRandom ||
      operand_semantics->label() == HloValueSemanticLabel::kWeight) {
    semantics = analysis_->NewHloValueSemantics(operand_semantics->label(),
                                                {dynamic_slice, {}});
  } else {
    HloValueSemantics semantics_value = CopySemantics(*operand_semantics);
    semantics = AddSemantics(semantics_value);
  }
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(
      dynamic_slice->shape(), semantics);
  analysis_->SetHloValueSemantics(dynamic_slice, semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  TF_ASSIGN_OR_RETURN(
      HloValueSemantics semantics,
      ComputeSemanticsFromOperands(dynamic_update_slice, {0, 1}));
  const HloValueSemantics* semantics_ptr = AddSemantics(semantics);
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(
      dynamic_update_slice->shape(), semantics_ptr);
  analysis_->SetHloValueSemantics(dynamic_update_slice, semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleCopyStart(
    HloInstruction* copy_start) {
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(copy_start->shape());
  const ShapeTree<const HloValueSemantics*>& operand_semantics_shape_tree =
      analysis_->GetInstructionSemantics(copy_start->operand(0));
  analysis_->DeepCopyHloValueSemantics(semantics_shape_tree,
                                       operand_semantics_shape_tree, {}, {0});
  analysis_->DeepCopyHloValueSemantics(semantics_shape_tree,
                                       operand_semantics_shape_tree, {}, {1});
  semantics_shape_tree.ForEachMutableElement(
      [this, copy_start](const ShapeIndex& shape_index,
                         const HloValueSemantics** semantics) {
        if (shape_index.empty()) {
          *semantics = analysis_->NewHloValueSemantics(
              HloValueSemanticLabel::kTupleOrToken, {copy_start, shape_index});
        }
        if (shape_index == ShapeIndex{2}) {
          *semantics = analysis_->NewHloValueSemantics(
              HloValueSemanticLabel::kRandom, {copy_start, shape_index});
        }
        if (shape_index == ShapeIndex{3}) {
          *semantics = analysis_->NewHloValueSemantics(
              HloValueSemanticLabel::kRandom, {copy_start, shape_index});
        }
      });
  analysis_->SetHloValueSemantics(copy_start, semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleCopyDone(HloInstruction* copy_done) {
  const ShapeTree<const HloValueSemantics*>& operand_semantics_shape_tree =
      analysis_->GetInstructionSemantics(copy_done->operand(0));
  analysis_->DeepCopyHloValueSemantics(copy_done, operand_semantics_shape_tree,
                                       {0});
  return OkStatus();
}
Status HloValueSemanticsPropagation::HandleCollectivePermuteStart(
    HloInstruction* collective_permute_start) {
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(
      collective_permute_start->shape());
  const ShapeTree<const HloValueSemantics*>& operand_semantics_shape_tree =
      analysis_->GetInstructionSemantics(collective_permute_start->operand(0));
  analysis_->DeepCopyHloValueSemantics(semantics_shape_tree,
                                       operand_semantics_shape_tree, {}, {0});
  analysis_->DeepCopyHloValueSemantics(semantics_shape_tree,
                                       operand_semantics_shape_tree, {}, {1});
  semantics_shape_tree.ForEachMutableElement(
      [this, collective_permute_start](const ShapeIndex& shape_index,
                                       const HloValueSemantics** semantics) {
        if (shape_index.empty()) {
          *semantics = analysis_->NewHloValueSemantics(
              HloValueSemanticLabel::kTupleOrToken,
              {collective_permute_start, {}});
        }
        if (shape_index == ShapeIndex{2}) {
          *semantics = analysis_->NewHloValueSemantics(
              HloValueSemanticLabel::kRandom,
              {collective_permute_start, shape_index});
        }
        if (shape_index == ShapeIndex{3}) {
          *semantics = analysis_->NewHloValueSemantics(
              HloValueSemanticLabel::kRandom,
              {collective_permute_start, shape_index});
        }
      });
  analysis_->SetHloValueSemantics(collective_permute_start,
                                  semantics_shape_tree);
  return OkStatus();
}
Status HloValueSemanticsPropagation::HandleCollectivePermuteDone(
    HloInstruction* collective_permute_done) {
  const ShapeTree<const HloValueSemantics*>& operand_semantics_shape_tree =
      analysis_->GetInstructionSemantics(collective_permute_done->operand(0));
  analysis_->DeepCopyHloValueSemantics(collective_permute_done,
                                       operand_semantics_shape_tree, {1});
  return OkStatus();
}
Status HloValueSemanticsPropagation::HandleGather(HloInstruction* gather) {
  const ShapeTree<const HloValueSemantics*>& operand_semantics_shape_tree =
      analysis_->GetInstructionSemantics(gather->operand(0));
  analysis_->DeepCopyHloValueSemantics(gather, operand_semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleScatter(HloInstruction* scatter) {
  TF_ASSIGN_OR_RETURN(HloValueSemantics semantics,
                      ComputeSemanticsFromOperands(scatter, {0, 2}));
  const HloValueSemantics* semantics_ptr = AddSemantics(semantics);
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(scatter->shape(),
                                                           semantics_ptr);
  analysis_->SetHloValueSemantics(scatter, semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleAfterAll(HloInstruction* after_all) {
  const HloValueSemantics* semantics = analysis_->NewHloValueSemantics(
      HloValueSemanticLabel::kTupleOrToken, {after_all, {}});
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(after_all->shape(),
                                                           semantics);
  analysis_->SetHloValueSemantics(after_all, semantics_shape_tree);
  return OkStatus();
}

Status HloValueSemanticsPropagation::HandleAsyncStart(
    HloInstruction* async_start) {
  return Unimplemented("AsyncStart is not supported yet.");
}
Status HloValueSemanticsPropagation::HandleAsyncDone(
    HloInstruction* async_done) {
  return Unimplemented("AsyncDone is not supported yet.");
}

Status HloValueSemanticsPropagation::HandleInfeed(HloInstruction* infeed) {
  ShapeTree<const HloValueSemantics*> semantics_shape_tree(infeed->shape(),
                                                           nullptr);
  semantics_shape_tree.ForEachMutableElement(
      [this, &semantics_shape_tree, infeed](
          const ShapeIndex& shape_index, const HloValueSemantics** semantics) {
        if (semantics_shape_tree.IsLeaf(shape_index)) {
          *semantics = analysis_->NewHloValueSemantics(
              HloValueSemanticLabel::kWeight, {infeed, shape_index});
        } else {
          *semantics = analysis_->NewHloValueSemantics(
              HloValueSemanticLabel::kTupleOrToken, {infeed, shape_index});
        }
      });
  analysis_->SetHloValueSemantics(infeed, semantics_shape_tree);
  return OkStatus();
}

}  // namespace xla
