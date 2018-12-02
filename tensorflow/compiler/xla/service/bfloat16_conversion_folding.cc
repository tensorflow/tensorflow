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

#include "tensorflow/compiler/xla/service/bfloat16_conversion_folding.h"

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class BFloat16ConversionFoldingVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit BFloat16ConversionFoldingVisitor(
      HloComputation* computation, const BFloat16Support* bfloat16_support)
      : computation_(computation), bfloat16_support_(bfloat16_support) {}

  Status DefaultAction(HloInstruction* hlo) override;

  // Special handling for cross-replica-sum which can have a tuple output.
  Status HandleCrossReplicaSum(HloInstruction* crs) override;

  static bool Run(HloComputation* computation,
                  const BFloat16Support* bfloat16_support) {
    BFloat16ConversionFoldingVisitor visitor(computation, bfloat16_support);
    TF_CHECK_OK(computation->Accept(&visitor));
    return visitor.changed_;
  }

 private:
  // Checks if the HLO has a BF16 -> F32 conversion as input, or a F32 -> BF16
  // conversion as output, and folds them to the HLO itself if feasible.
  Status TryFoldBF16Conversions(HloInstruction* hlo);

  // Folds the F32 -> BF16 conversions from the HLO's output.
  //
  // Precondition: all of the HLO's users are F32 -> BF16 conversions.
  Status FoldOutputConversions(HloInstruction* hlo);

  // Folds the BF16 -> F32 conversion operand to the HLO.
  //
  // Precondition: the operand is a F32 -> BF16 conversion.
  Status FoldOperandConversion(HloInstruction* hlo, int64 operand_index);

  HloComputation* computation_;
  const BFloat16Support* bfloat16_support_;
  bool changed_ = false;
};

Status BFloat16ConversionFoldingVisitor::FoldOutputConversions(
    HloInstruction* hlo) {
  std::vector<HloInstruction*> materialized_users = hlo->users();
  hlo->mutable_shape()->set_element_type(BF16);
  for (auto user : materialized_users) {
    CHECK_EQ(user->opcode(), HloOpcode::kConvert);
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(hlo));
    changed_ = true;
  }
  return Status::OK();
}

Status BFloat16ConversionFoldingVisitor::FoldOperandConversion(
    HloInstruction* hlo, int64 operand_index) {
  // The operand is a convert from BF16 to F32.
  auto operand = hlo->mutable_operand(operand_index);
  CHECK_EQ(operand->opcode(), HloOpcode::kConvert);
  TF_RETURN_IF_ERROR(
      hlo->ReplaceOperandWith(operand_index, operand->mutable_operand(0)));
  changed_ = true;
  return Status::OK();
}

namespace {

// Returns whether hlo has users and all users are conversions from F32 to BF16.
bool AllUsersAreF32ToBF16Converts(const HloInstruction* hlo) {
  if (hlo->user_count() == 0 || hlo->shape().element_type() != F32) {
    return false;
  }
  for (const auto user : hlo->users()) {
    if (user->opcode() == HloOpcode::kConvert &&
        user->shape().element_type() == BF16) {
      continue;
    }
    return false;
  }
  return true;
}

}  // namespace

Status BFloat16ConversionFoldingVisitor::TryFoldBF16Conversions(
    HloInstruction* hlo) {
  std::vector<int64> bf16_to_f32_operands;
  bool has_other_f32_operands = false;
  for (int64 i = 0; i < hlo->operands().size(); ++i) {
    auto operand = hlo->operand(i);
    if (operand->shape().element_type() == F32) {
      if (operand->opcode() == HloOpcode::kConvert &&
          operand->operand(0)->shape().element_type() == BF16 &&
          bfloat16_support_->SupportsBF16Operand(*hlo, i)) {
        // Operand is a convert from BF16 to F32 and we support BF16 input
        // directly in the current HLO at the operand index.
        bf16_to_f32_operands.push_back(i);
      } else {
        has_other_f32_operands = true;
      }
      continue;
    }
  }

  const bool fold_output_conversion =
      AllUsersAreF32ToBF16Converts(hlo) &&
      bfloat16_support_->SupportsBF16Output(*hlo);

  if (!bfloat16_support_->SupportsMixedPrecisions(*hlo)) {
    if (has_other_f32_operands ||
        (!fold_output_conversion && hlo->shape().element_type() == F32)) {
      // Some of the operands/output will remain F32, but we cannot use mixed
      // precisions, so we cannot do anything here.
      return Status::OK();
    }
  }

  if (fold_output_conversion) {
    TF_RETURN_IF_ERROR(FoldOutputConversions(hlo));
  }

  for (int64 i : bf16_to_f32_operands) {
    TF_RETURN_IF_ERROR(FoldOperandConversion(hlo, i));
  }
  return Status::OK();
}

Status BFloat16ConversionFoldingVisitor::DefaultAction(HloInstruction* hlo) {
  // Do not fold BF16 conversions for instructions related to tuples, entry and
  // exit of a computation, fusion, convert, side-effecting instructions and
  // control flow.
  if (hlo->opcode() == HloOpcode::kTuple ||            //
      hlo->opcode() == HloOpcode::kGetTupleElement ||  //
      hlo->opcode() == HloOpcode::kConstant ||         //
      hlo->opcode() == HloOpcode::kParameter ||        //
      hlo->opcode() == HloOpcode::kFusion ||           //
      hlo->opcode() == HloOpcode::kConvert ||          //
      hlo->opcode() == HloOpcode::kCall ||             //
      hlo->opcode() == HloOpcode::kCustomCall ||       //
      hlo->opcode() == HloOpcode::kWhile ||            //
      hlo->opcode() == HloOpcode::kConditional ||      //
      hlo->HasSideEffectNoRecurse()) {
    return Status::OK();
  }
  if (hlo == computation_->root_instruction() &&
      !bfloat16_support_->SupportsMixedPrecisions(*hlo)) {
    // If hlo is the root instruction, we cannot change its output, so folding
    // can only happen when it supports mixed precision so that we can change
    // its operands.
    return Status::OK();
  }
  return TryFoldBF16Conversions(hlo);
}

Status BFloat16ConversionFoldingVisitor::HandleCrossReplicaSum(
    HloInstruction* crs) {
  if (crs->IsCrossModuleAllReduce()) {
    // Cross-module all-reduce has side effect.
    return Status::OK();
  }
  // First use DefaultAction() to handle the operands. It can't handle
  // tuple-shaped output.
  TF_RETURN_IF_ERROR(DefaultAction(crs));

  if (!bfloat16_support_->SupportsMixedPrecisions(*crs)) {
    return Status::OK();
  }

  // If the output is not a tuple, we don't need special handling.
  if (!ShapeUtil::IsTuple(crs->shape())) {
    return Status::OK();
  }

  // If crs is the root instruction, we should keep its original output type.
  // The root instruction implicitly has a use from being the result of the
  // computation, and the code below does not take this use into account.
  if (crs == computation_->root_instruction()) {
    return Status::OK();
  }

  // Then do per-tuple-element handling on the output.
  std::vector<std::vector<HloInstruction*>> per_tuple_element_gtes(
      crs->operand_count());
  for (auto user : crs->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return Status::OK();
    }
    per_tuple_element_gtes[user->tuple_index()].push_back(user);
  }

  for (int64 i = 0; i < crs->operand_count(); ++i) {
    // Fold conversions only when all the get-tuple-elements' users are
    // conversions from F32 to BF16.
    auto all_gte_users_are_bf16_convert = [&per_tuple_element_gtes, i]() {
      for (auto gte : per_tuple_element_gtes[i]) {
        if (!AllUsersAreF32ToBF16Converts(gte)) {
          return false;
        }
      }
      return true;
    };
    if (!all_gte_users_are_bf16_convert()) {
      continue;
    }

    ShapeUtil::GetMutableSubshape(crs->mutable_shape(), {i})
        ->set_element_type(BF16);
    for (auto gte : per_tuple_element_gtes[i]) {
      TF_RETURN_IF_ERROR(FoldOutputConversions(gte));
    }
  }

  return Status::OK();
}

StatusOr<bool> BFloat16ConversionFolding::Run(HloModule* module) {
  XLA_VLOG_LINES(
      2, "BFloat16ConversionFolding::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (BFloat16ConversionFoldingVisitor::Run(comp, bfloat16_support_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "BFloat16ConversionFolding::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
