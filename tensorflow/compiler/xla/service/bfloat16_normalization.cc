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

#include "tensorflow/compiler/xla/service/bfloat16_normalization.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class BFloat16NormalizationVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit BFloat16NormalizationVisitor(HloComputation* computation,
                                        const BFloat16Support* bfloat16_support)
      : computation_(computation), bfloat16_support_(bfloat16_support) {}

  Status DefaultAction(HloInstruction* hlo) override;

  // Special handling for cross-replica-sum which can have a tuple output.
  Status HandleCrossReplicaSum(HloInstruction* crs) override;

  static bool Run(HloComputation* computation,
                  const BFloat16Support* bfloat16_support) {
    BFloat16NormalizationVisitor visitor(computation, bfloat16_support);
    TF_CHECK_OK(computation->Accept(&visitor));
    return visitor.changed_;
  }

 private:
  // Checks if the HLO uses BF16 in an unsupported way, and if so, inserts
  // conversions between F32 and BF16 to make it supported.
  Status HandleInstruction(HloInstruction* hlo);

  // Inserts a conversion HLO that changes the given HLO's output type.
  Status InsertConvertAfterOutput(HloInstruction* hlo, PrimitiveType to,
                                  HloComputation* computation);

  // Changes the output type to the specified type, then inserts a conversion
  // to the original type.
  Status ChangeOutputTypeThenInsertConvertBack(HloInstruction* hlo,
                                               PrimitiveType to,
                                               HloComputation* computation);

  // Inserts a conversion HLO that changes the given HLO's operand type.
  Status InsertConvertBeforeOperand(HloInstruction* hlo, int64 operand_idx,
                                    PrimitiveType to,
                                    HloComputation* computation);

  // Inserts conversion HLOs to replace the called computations' BF16
  // operands/outputs to F32.
  Status ConvertCalledComputations(
      HloInstruction* hlo,
      tensorflow::gtl::ArraySlice<HloComputation*> bf16_called_comps);

  HloComputation* computation_;
  const BFloat16Support* bfloat16_support_;
  bool changed_ = false;
};

Status BFloat16NormalizationVisitor::InsertConvertAfterOutput(
    HloInstruction* hlo, PrimitiveType to, HloComputation* computation) {
  bool is_root = computation->root_instruction() == hlo;
  std::vector<HloInstruction*> materialized_users = hlo->users();
  // Use inst's shape temporarily, in order to pass checks in ReplaceUseWith.
  auto convert = computation->AddInstruction(
      HloInstruction::CreateConvert(hlo->shape(), hlo));
  for (auto* user : materialized_users) {
    TF_RETURN_IF_ERROR(hlo->ReplaceUseWith(user, convert));
  }
  if (is_root) {
    computation->set_root_instruction(convert);
  }
  convert->mutable_shape()->set_element_type(to);
  changed_ = true;
  return Status::OK();
}

Status BFloat16NormalizationVisitor::ChangeOutputTypeThenInsertConvertBack(
    HloInstruction* hlo, PrimitiveType to, HloComputation* computation) {
  auto original_type = hlo->shape().element_type();
  hlo->mutable_shape()->set_element_type(to);
  return InsertConvertAfterOutput(hlo, original_type, computation);
}

Status BFloat16NormalizationVisitor::InsertConvertBeforeOperand(
    HloInstruction* hlo, int64 operand_idx, PrimitiveType to,
    HloComputation* computation) {
  auto operand = hlo->mutable_operand(operand_idx);
  auto convert = computation->AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::ChangeElementType(operand->shape(), to), operand));
  TF_RETURN_IF_ERROR(hlo->ReplaceOperandWith(operand_idx, convert));
  changed_ = true;
  return Status::OK();
}

Status BFloat16NormalizationVisitor::ConvertCalledComputations(
    HloInstruction* hlo,
    tensorflow::gtl::ArraySlice<HloComputation*> bf16_called_comps) {
  std::map<HloComputation*, HloComputation*> cloned_computations;
  for (auto& comp : bf16_called_comps) {
    auto cloned = comp->parent()->AddEmbeddedComputation(comp->Clone());
    cloned_computations[comp] = cloned;
    changed_ = true;
  }
  hlo->ReplaceCalledComputations([&](HloComputation* comp) {
    auto it = cloned_computations.find(comp);
    if (it != cloned_computations.end()) {
      return it->second;
    }
    return comp;
  });
  for (auto& comp_pair : cloned_computations) {
    auto comp = comp_pair.second;
    if (comp->root_instruction()->shape().element_type() == BF16) {
      TF_RETURN_IF_ERROR(
          InsertConvertAfterOutput(comp->root_instruction(), F32, comp));
    }
    for (auto* param : comp->parameter_instructions()) {
      if (param->shape().element_type() == BF16) {
        // This changes the parameter to F32 then inserts a convert after it.
        TF_RETURN_IF_ERROR(
            ChangeOutputTypeThenInsertConvertBack(param, F32, comp));
      }
    }
  }
  return Status::OK();
}

Status BFloat16NormalizationVisitor::HandleCrossReplicaSum(
    HloInstruction* crs) {
  if (!ShapeUtil::IsTuple(crs->shape())) {
    return HandleInstruction(crs);
  }

  std::vector<PrimitiveType> operand_types(crs->operand_count());
  std::vector<PrimitiveType> output_types(crs->operand_count());
  bool has_f32 = false;
  bool has_bf16 = false;
  bool has_bf16_output = false;
  for (int64 i = 0; i < crs->operand_count(); ++i) {
    operand_types[i] = crs->operand(i)->shape().element_type();
    output_types[i] = ShapeUtil::GetSubshape(crs->shape(), {i}).element_type();
    if (operand_types[i] == F32 || output_types[i] == F32) {
      has_f32 = true;
    } else if (operand_types[i] == BF16) {
      has_bf16 = true;
    }
    if (output_types[i] == BF16) {
      has_bf16 = true;
      has_bf16_output = true;
    }
  }

  for (int64 i = 0; i < crs->operand_count(); ++i) {
    if (operand_types[i] != BF16) {
      continue;
    }
    if (bfloat16_support_->SupportsBF16Operand(*crs, i) &&
        (bfloat16_support_->SupportsMixedPrecisions(*crs) || !has_f32)) {
      continue;
    }
    TF_RETURN_IF_ERROR(InsertConvertBeforeOperand(crs, i, F32, computation_));
    has_f32 = true;
  }

  if (!has_bf16_output) {
    return Status::OK();
  }

  if (bfloat16_support_->SupportsBF16Output(*crs) &&
      (bfloat16_support_->SupportsMixedPrecisions(*crs) || !has_f32)) {
    return Status::OK();
  }

  std::vector<HloInstruction*> output_elements(crs->operand_count());
  auto original_shape = crs->shape();
  for (int64 i = 0; i < crs->operand_count(); ++i) {
    auto subshape = ShapeUtil::GetMutableSubshape(crs->mutable_shape(), {i});
    if (output_types[i] != BF16) {
      output_elements[i] = computation_->AddInstruction(
          HloInstruction::CreateGetTupleElement(*subshape, crs, i));
      continue;
    }
    subshape->set_element_type(F32);
    auto gte = computation_->AddInstruction(
        HloInstruction::CreateGetTupleElement(*subshape, crs, i));
    output_elements[i] =
        computation_->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(*subshape, BF16), gte));
  }
  auto tuple = computation_->AddInstruction(
      HloInstruction::CreateTuple(output_elements));

  std::vector<HloInstruction*> materialized_users = crs->users();
  // Use the crs' shape temporarily, in order to pass checks in
  // ReplaceUseWith.
  *tuple->mutable_shape() = crs->shape();
  for (auto* user : materialized_users) {
    TF_RETURN_IF_ERROR(crs->ReplaceUseWith(user, tuple));
  }
  *tuple->mutable_shape() = original_shape;
  return Status::OK();
}

Status BFloat16NormalizationVisitor::HandleInstruction(HloInstruction* hlo) {
  std::vector<int64> bf16_operands;
  std::vector<int64> f32_operands;
  bool has_f32 = false;
  bool has_bf16 = false;

  for (int64 i = 0; i < hlo->operand_count(); ++i) {
    if (hlo->operand(i)->shape().element_type() == F32) {
      f32_operands.push_back(i);
      has_f32 = true;
    } else if (hlo->operand(i)->shape().element_type() == BF16) {
      bf16_operands.push_back(i);
      has_bf16 = true;
    }
  }

  if (hlo->shape().element_type() == F32) {
    has_f32 = true;
  } else if (hlo->shape().element_type() == BF16) {
    has_bf16 = true;
  }

  std::vector<HloComputation*> bf16_called_comps;
  for (auto* comp : hlo->called_computations()) {
    bool comp_has_bf16 = false;
    if (comp->root_instruction()->shape().element_type() == F32) {
      has_f32 = true;
    } else if (comp->root_instruction()->shape().element_type() == BF16) {
      has_bf16 = true;
      comp_has_bf16 = true;
    }
    for (auto* param : comp->parameter_instructions()) {
      if (param->shape().element_type() == F32) {
        has_f32 = true;
      } else if (param->shape().element_type() == BF16) {
        has_bf16 = true;
        comp_has_bf16 = true;
      }
    }
    if (comp_has_bf16) {
      bf16_called_comps.push_back(comp);
    }
  }

  if (!bfloat16_support_->SupportsMixedPrecisions(*hlo) && has_bf16 &&
      has_f32) {
    // Resolve unsupported mixed precision.
    //
    // See if we can change everything to BF16.
    if (hlo->called_computations().empty() &&
        hlo->shape().element_type() == BF16) {
      bool can_use_bf16 = true;
      for (int i : f32_operands) {
        if (bfloat16_support_->EffectiveOperandPrecisionIsOutputPrecision(*hlo,
                                                                          i) &&
            bfloat16_support_->SupportsBF16Operand(*hlo, i)) {
          continue;
        }
        can_use_bf16 = false;
        break;
      }
      if (can_use_bf16) {
        for (int i : f32_operands) {
          TF_RETURN_IF_ERROR(
              InsertConvertBeforeOperand(hlo, i, BF16, computation_));
        }
        return Status::OK();
      }
    }
    if (hlo->shape().element_type() == BF16) {
      TF_RETURN_IF_ERROR(
          ChangeOutputTypeThenInsertConvertBack(hlo, F32, computation_));
    }
    for (int i : bf16_operands) {
      TF_RETURN_IF_ERROR(InsertConvertBeforeOperand(hlo, i, F32, computation_));
    }
    return ConvertCalledComputations(hlo, bf16_called_comps);
  }

  for (int i : bf16_operands) {
    if (!bfloat16_support_->SupportsBF16Operand(*hlo, i)) {
      TF_RETURN_IF_ERROR(InsertConvertBeforeOperand(hlo, i, F32, computation_));
    }
  }

  if (hlo->shape().element_type() == BF16 &&
      !bfloat16_support_->SupportsBF16Output(*hlo)) {
    TF_RETURN_IF_ERROR(
        ChangeOutputTypeThenInsertConvertBack(hlo, F32, computation_));
  }

  return Status::OK();
}

Status BFloat16NormalizationVisitor::DefaultAction(HloInstruction* hlo) {
  // Do not change instructions related to entry and exit of a computation,
  // tuples, fusion, convert, and control flow.
  if (hlo->opcode() == HloOpcode::kTuple ||            //
      hlo->opcode() == HloOpcode::kGetTupleElement ||  //
      hlo->opcode() == HloOpcode::kInfeed ||           //
      hlo->opcode() == HloOpcode::kOutfeed ||          //
      hlo->opcode() == HloOpcode::kConstant ||         //
      hlo->opcode() == HloOpcode::kParameter ||        //
      hlo->opcode() == HloOpcode::kFusion ||           //
      hlo->opcode() == HloOpcode::kConvert ||          //
      hlo->opcode() == HloOpcode::kCall ||             //
      hlo->opcode() == HloOpcode::kCustomCall ||       //
      hlo->opcode() == HloOpcode::kWhile ||            //
      hlo->opcode() == HloOpcode::kConditional) {
    return Status::OK();
  }
  return HandleInstruction(hlo);
}

StatusOr<bool> BFloat16Normalization::Run(HloModule* module) {
  XLA_VLOG_LINES(
      2, "BFloat16Normalization::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeComputationPostOrder()) {
    if (BFloat16NormalizationVisitor::Run(comp, bfloat16_support_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2,
                 "BFloat16Normalization::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
