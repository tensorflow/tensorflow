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

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {

namespace {

class BFloat16NormalizationVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit BFloat16NormalizationVisitor(
      const BFloat16Support* bfloat16_support,
      BFloat16Normalization* bfloat16_normalization)
      : computation_(nullptr),
        bfloat16_support_(bfloat16_support),
        bfloat16_normalization_(bfloat16_normalization) {}

  bool changed() const { return changed_; }
  Status DefaultAction(HloInstruction* hlo) override;
  Status Preprocess(HloInstruction* hlo) override;

 private:
  // Checks if the HLO uses BF16 in an unsupported way, and if so, inserts
  // conversions between F32 and BF16 to make it supported.
  Status HandleInstruction(HloInstruction* hlo);

  // Handle instructions with tuple outputs by examining each output
  // independently.
  Status HandleMultipleOutputs(HloInstruction* hlo);

  // Creates a copy of `hlo` with subshapes matching `from` type converted to
  // `to` type. If no matching subshapes are found, returns the original `hlo`.
  StatusOr<HloInstruction*> ConvertType(HloInstruction* hlo, PrimitiveType from,
                                        PrimitiveType to,
                                        HloComputation* computation);

  // Inserts a conversion HLO that changes the given HLO's output type. If the
  // output is a tuple, change all elements that match the from type.
  Status InsertConvertAfterOutput(HloInstruction* hlo, PrimitiveType from,
                                  PrimitiveType to,
                                  HloComputation* computation);

  // Changes the output type to the specified type, then inserts a conversion
  // to the original type. If the output is a tuple, change all elements that
  // match the from type.
  Status ChangeOutputTypeThenInsertConvertBack(HloInstruction* hlo,
                                               PrimitiveType from,
                                               PrimitiveType to,
                                               HloComputation* computation);

  // Inserts a conversion HLO that changes the given HLO's operand type. If the
  // operand is a tuple, change all elements that match the from type.
  Status InsertConvertBeforeOperand(HloInstruction* hlo, int64_t operand_idx,
                                    PrimitiveType from, PrimitiveType to,
                                    HloComputation* computation);

  // Inserts conversion HLOs to replace the called computations' BF16
  // operands/outputs to F32.
  Status ConvertCalledComputations(
      HloInstruction* hlo, absl::Span<HloComputation* const> bf16_called_comps);

  HloComputation* computation_;
  const BFloat16Support* bfloat16_support_;
  BFloat16Normalization* bfloat16_normalization_;
  bool changed_ = false;
};

int64_t CountSubshapesWithMatchingType(const Shape& shape, PrimitiveType type) {
  int64_t count = 0;
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.element_type() == type) {
          ++count;
        }
      });
  return count;
}

int64_t ShapeLeafCount(const Shape& shape) {
  int64_t count = 0;
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& subshape, const ShapeIndex& index) {
        if (ShapeUtil::IsLeafIndex(shape, index)) {
          ++count;
        }
      });
  return count;
}

StatusOr<HloInstruction*> BFloat16NormalizationVisitor::ConvertType(
    HloInstruction* hlo, PrimitiveType from, PrimitiveType to,
    HloComputation* computation) {
  if (CountSubshapesWithMatchingType(hlo->shape(), from) == 0) {
    return hlo;
  }
  // If `hlo` is a convert from `to` to `from`, then we can return its operand,
  // if it is a BF16->F32 convert which doesn't do rounding.
  if (hlo->opcode() == HloOpcode::kConvert &&
      hlo->operand(0)->shape().element_type() == to && to == BF16 &&
      from == F32) {
    return hlo->mutable_operand(0);
  }
  TF_ASSIGN_OR_RETURN(
      auto new_hlo,
      computation->DeepCopyInstructionWithCustomCopier(
          hlo, [&](HloInstruction* leaf, const ShapeIndex& leaf_index,
                   HloComputation* comp) {
            const auto& original_subshape =
                ShapeUtil::GetSubshape(hlo->shape(), leaf_index);
            if (original_subshape.element_type() != from) {
              return leaf;
            }
            auto new_subshape =
                ShapeUtil::ChangeElementType(original_subshape, to);
            bfloat16_normalization_->UpdateLayout(&new_subshape);
            return computation->AddInstruction(
                HloInstruction::CreateConvert(new_subshape, leaf));
          }));
  return new_hlo;
}

Status BFloat16NormalizationVisitor::InsertConvertAfterOutput(
    HloInstruction* hlo, PrimitiveType from, PrimitiveType to,
    HloComputation* computation) {
  bool is_root = computation->root_instruction() == hlo;
  std::vector<HloInstruction*> materialized_users = hlo->users();

  TF_ASSIGN_OR_RETURN(auto new_hlo, ConvertType(hlo, from, to, computation));
  if (new_hlo == hlo) {
    return OkStatus();
  }

  for (auto* user : materialized_users) {
    TF_RETURN_IF_ERROR(hlo->ReplaceUseWithDifferentShape(user, new_hlo));
  }
  if (is_root) {
    computation->set_root_instruction(new_hlo, /*accept_different_shape=*/true);
  }
  changed_ = true;
  return OkStatus();
}

Status BFloat16NormalizationVisitor::ChangeOutputTypeThenInsertConvertBack(
    HloInstruction* hlo, PrimitiveType from, PrimitiveType to,
    HloComputation* computation) {
  auto original_shape = hlo->shape();
  if (CountSubshapesWithMatchingType(original_shape, from) == 0) {
    return OkStatus();
  }
  ShapeUtil::ForEachMutableSubshape(
      hlo->mutable_shape(), [&](Shape* subshape, const xla::ShapeIndex& index) {
        if (subshape->element_type() == from) {
          subshape->set_element_type(to);
        }
      });
  bfloat16_normalization_->UpdateLayout(hlo->mutable_shape());
  bool is_root = computation->root_instruction() == hlo;
  std::vector<HloInstruction*> materialized_users = hlo->users();
  TF_ASSIGN_OR_RETURN(
      auto new_hlo,
      computation->DeepCopyInstructionWithCustomCopier(
          hlo, [&](HloInstruction* leaf, const ShapeIndex& leaf_index,
                   HloComputation* comp) {
            const auto& original_subshape =
                ShapeUtil::GetSubshape(original_shape, leaf_index);
            if (original_subshape.element_type() ==
                leaf->shape().element_type()) {
              return leaf;
            }
            return computation->AddInstruction(
                HloInstruction::CreateConvert(original_subshape, leaf));
          }));

  for (auto* user : materialized_users) {
    // If the user is a BF16 -> F32 convert, we can replace it with `hlo`, which
    // has its input changed to F32.
    if (user->opcode() == HloOpcode::kConvert &&
        user->shape().element_type() == to && to == F32 && from == BF16) {
      TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(hlo));
    } else {
      TF_RETURN_IF_ERROR(hlo->ReplaceUseWithDifferentShape(user, new_hlo));
    }
  }
  if (is_root) {
    computation->set_root_instruction(new_hlo, /*accept_different_shape=*/true);
  }
  changed_ = true;
  return OkStatus();
}

Status BFloat16NormalizationVisitor::InsertConvertBeforeOperand(
    HloInstruction* hlo, int64_t operand_idx, PrimitiveType from,
    PrimitiveType to, HloComputation* computation) {
  auto operand = hlo->mutable_operand(operand_idx);
  TF_ASSIGN_OR_RETURN(auto new_operand,
                      ConvertType(operand, from, to, computation));
  if (new_operand == operand) {
    return OkStatus();
  }
  TF_RETURN_IF_ERROR(
      hlo->ReplaceOperandWithDifferentShape(operand_idx, new_operand));
  changed_ = true;
  return OkStatus();
}

Status BFloat16NormalizationVisitor::ConvertCalledComputations(
    HloInstruction* hlo, absl::Span<HloComputation* const> bf16_called_comps) {
  absl::flat_hash_map<HloComputation*, HloComputation*> cloned_computations;
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
    TF_RETURN_IF_ERROR(
        InsertConvertAfterOutput(comp->root_instruction(), BF16, F32, comp));
    for (auto* param : comp->parameter_instructions()) {
      // This changes the parameter to F32 then inserts a convert after it.
      TF_RETURN_IF_ERROR(
          ChangeOutputTypeThenInsertConvertBack(param, BF16, F32, comp));
    }
  }
  return OkStatus();
}

Status BFloat16NormalizationVisitor::HandleMultipleOutputs(
    HloInstruction* hlo) {
  std::vector<PrimitiveType> operand_types(hlo->operand_count());
  std::vector<PrimitiveType> output_types(hlo->operand_count());
  int64_t f32_count = 0;
  int64_t bf16_count = 0;
  bool has_unsupported_bf16_operand = false;
  bool has_unsupported_bf16_output = false;
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    CHECK(hlo->operand(i)->shape().IsArray());
    CHECK(ShapeUtil::GetSubshape(hlo->shape(), {i}).IsArray());
    operand_types[i] = hlo->operand(i)->shape().element_type();
    output_types[i] = ShapeUtil::GetSubshape(hlo->shape(), {i}).element_type();
    if (operand_types[i] == F32) {
      f32_count += 1;
    } else if (operand_types[i] == BF16) {
      bf16_count += 1;
      if (!bfloat16_support_->SupportsBF16Operand(*hlo, i)) {
        has_unsupported_bf16_operand = true;
      }
    }
    if (output_types[i] == F32) {
      f32_count += 1;
    } else if (output_types[i] == BF16) {
      bf16_count += 1;
      if (!bfloat16_support_->SupportsBF16Output(*hlo)) {
        has_unsupported_bf16_output = true;
      }
    }
  }

  if (bf16_count == 0) {
    return OkStatus();
  }

  auto should_convert_operand = [&](int64_t i) {
    if (operand_types[i] != BF16) {
      return false;
    }
    if (!bfloat16_support_->SupportsBF16Operand(*hlo, i)) {
      return true;
    }
    if (bfloat16_support_->SupportsMixedPrecisions(*hlo)) {
      return false;
    }
    return has_unsupported_bf16_operand || has_unsupported_bf16_output ||
           f32_count > 0;
  };

  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    if (should_convert_operand(i)) {
      TF_RETURN_IF_ERROR(
          InsertConvertBeforeOperand(hlo, i, BF16, F32, computation_));
      f32_count += 1;
      bf16_count -= 1;
    }
  }

  if (!has_unsupported_bf16_output &&
      (bfloat16_support_->SupportsMixedPrecisions(*hlo) || f32_count == 0 ||
       bf16_count == 0)) {
    return OkStatus();
  }

  std::vector<HloComputation*> bf16_called_comps;
  for (auto* comp : hlo->called_computations()) {
    bool comp_has_bf16 = false;
    if (comp->root_instruction()->shape().element_type() == F32) {
      f32_count += 1;
    } else if (comp->root_instruction()->shape().element_type() == BF16) {
      bf16_count += 1;
      comp_has_bf16 = true;
    }
    for (auto* param : comp->parameter_instructions()) {
      if (param->shape().element_type() == F32) {
        f32_count += 1;
      } else if (param->shape().element_type() == BF16) {
        bf16_count += 1;
        comp_has_bf16 = true;
      }
    }
    if (comp_has_bf16) {
      bf16_called_comps.push_back(comp);
    }
  }

  std::vector<HloInstruction*> materialized_users = hlo->users();
  std::vector<HloInstruction*> output_elements(hlo->operand_count());
  auto original_shape = hlo->shape();
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    auto subshape = ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), {i});
    if (output_types[i] != BF16) {
      output_elements[i] = computation_->AddInstruction(
          HloInstruction::CreateGetTupleElement(*subshape, hlo, i));
      continue;
    }
    subshape->set_element_type(F32);
    bfloat16_normalization_->UpdateLayout(subshape);
    auto gte = computation_->AddInstruction(
        HloInstruction::CreateGetTupleElement(*subshape, hlo, i));
    auto shape = ShapeUtil::ChangeElementType(*subshape, BF16);
    bfloat16_normalization_->UpdateLayout(&shape);
    output_elements[i] =
        computation_->AddInstruction(HloInstruction::CreateConvert(shape, gte));
  }
  auto tuple = computation_->AddInstruction(
      HloInstruction::CreateTuple(output_elements));

  // Use the hlo' shape temporarily, in order to pass checks in
  // ReplaceUseWith.
  *tuple->mutable_shape() = hlo->shape();
  for (auto* user : materialized_users) {
    TF_RETURN_IF_ERROR(hlo->ReplaceUseWith(user, tuple));
  }
  bool is_root = computation_->root_instruction() == hlo;
  if (is_root) {
    computation_->set_root_instruction(tuple);
  }
  *tuple->mutable_shape() = original_shape;
  return ConvertCalledComputations(hlo, bf16_called_comps);
}

Status BFloat16NormalizationVisitor::HandleInstruction(HloInstruction* hlo) {
  int f32_count = 0;
  int bf16_count = 0;

  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    f32_count += CountSubshapesWithMatchingType(hlo->operand(i)->shape(), F32);
    bf16_count +=
        CountSubshapesWithMatchingType(hlo->operand(i)->shape(), BF16);
  }

  f32_count += CountSubshapesWithMatchingType(hlo->shape(), F32);
  bf16_count += CountSubshapesWithMatchingType(hlo->shape(), BF16);

  std::vector<HloComputation*> bf16_called_comps;
  for (auto* comp : hlo->called_computations()) {
    bool comp_has_bf16 = false;
    f32_count +=
        CountSubshapesWithMatchingType(comp->root_instruction()->shape(), F32);
    int64_t bf16_count_comp_root =
        CountSubshapesWithMatchingType(comp->root_instruction()->shape(), BF16);
    if (bf16_count_comp_root > 0) {
      bf16_count += bf16_count_comp_root;
      comp_has_bf16 = true;
    }
    for (auto* param : comp->parameter_instructions()) {
      f32_count += CountSubshapesWithMatchingType(param->shape(), F32);
      int64_t bf16_count_comp_param =
          CountSubshapesWithMatchingType(param->shape(), BF16);
      if (bf16_count_comp_param > 0) {
        bf16_count += bf16_count_comp_param;
        comp_has_bf16 = true;
      }
    }
    if (comp_has_bf16) {
      bf16_called_comps.push_back(comp);
    }
  }

  // Resolve unsupported BF16 operands.
  for (int i = 0; i < hlo->operand_count(); ++i) {
    int64_t bf16_count_in_operand =
        CountSubshapesWithMatchingType(hlo->operand(i)->shape(), BF16);
    if (bf16_count_in_operand > 0 &&
        !bfloat16_support_->SupportsBF16Operand(*hlo, i)) {
      TF_RETURN_IF_ERROR(
          InsertConvertBeforeOperand(hlo, i, BF16, F32, computation_));
      bf16_count -= bf16_count_in_operand;
      f32_count += bf16_count_in_operand;
    }
  }

  // Resolve unsupported BF16 output.
  if (!bfloat16_support_->SupportsBF16Output(*hlo)) {
    int64_t bf16_count_in_hlo =
        CountSubshapesWithMatchingType(hlo->shape(), BF16);
    if (bf16_count_in_hlo > 0) {
      TF_RETURN_IF_ERROR(
          ChangeOutputTypeThenInsertConvertBack(hlo, BF16, F32, computation_));
      bf16_count -= bf16_count_in_hlo;
      f32_count += bf16_count_in_hlo;
    }
  }

  // Resolve unsupported mixed precision after resolving unsupported BF16
  // operands and output, because the numbers of BF16 operands/output and F32
  // operands/output may have changed.
  if (bfloat16_support_->SupportsMixedPrecisions(*hlo) || bf16_count == 0 ||
      f32_count == 0) {
    return OkStatus();
  }
  // See if we can change everything to BF16.
  if (hlo->called_computations().empty() &&
      CountSubshapesWithMatchingType(hlo->shape(), BF16) ==
          ShapeLeafCount(hlo->shape())) {
    bool can_use_bf16 = true;
    for (int i = 0; i < hlo->operand_count(); ++i) {
      if (CountSubshapesWithMatchingType(hlo->operand(i)->shape(), BF16) ==
          ShapeLeafCount(hlo->operand(i)->shape())) {
        continue;
      }
      if ((bfloat16_support_->EffectiveOperandPrecisionIsBF16(*hlo, i) ||
           bfloat16_support_->EffectiveOperandPrecisionIsOutputPrecision(*hlo,
                                                                         i)) &&
          bfloat16_support_->SupportsBF16Operand(*hlo, i)) {
        continue;
      }
      can_use_bf16 = false;
      break;
    }
    if (can_use_bf16) {
      for (int i = 0; i < hlo->operand_count(); ++i) {
        TF_RETURN_IF_ERROR(
            InsertConvertBeforeOperand(hlo, i, F32, BF16, computation_));
      }
      return OkStatus();
    }
  }
  TF_RETURN_IF_ERROR(
      ChangeOutputTypeThenInsertConvertBack(hlo, BF16, F32, computation_));
  for (int i = 0; i < hlo->operand_count(); ++i) {
    TF_RETURN_IF_ERROR(
        InsertConvertBeforeOperand(hlo, i, BF16, F32, computation_));
  }
  return ConvertCalledComputations(hlo, bf16_called_comps);
}

Status BFloat16NormalizationVisitor::DefaultAction(HloInstruction* hlo) {
  // Do not change instructions related to entry and exit of a computation,
  // tuples, fusion, convert, side-effecting instructions, control flow, and
  // bitcast-convert.
  if (hlo->opcode() == HloOpcode::kTuple ||            //
      hlo->opcode() == HloOpcode::kGetTupleElement ||  //
      hlo->opcode() == HloOpcode::kConstant ||         //
      hlo->opcode() == HloOpcode::kDomain ||           //
      hlo->opcode() == HloOpcode::kParameter ||        //
      hlo->opcode() == HloOpcode::kFusion ||           //
      hlo->opcode() == HloOpcode::kConvert ||          //
      hlo->opcode() == HloOpcode::kCall ||             //
      hlo->opcode() == HloOpcode::kCustomCall ||       //
      hlo->opcode() == HloOpcode::kWhile ||            //
      hlo->opcode() == HloOpcode::kConditional ||      //
      hlo->opcode() == HloOpcode::kBitcastConvert ||   //
      hlo->HasSideEffectNoRecurse()) {
    return OkStatus();
  }
  // TODO(b/112040122): Correctly normalize variadic reduce.
  if ((hlo->opcode() == HloOpcode::kSort ||
       hlo->opcode() == HloOpcode::kAllReduce ||
       hlo->opcode() == HloOpcode::kReduceScatter) &&
      hlo->shape().IsTuple()) {
    return HandleMultipleOutputs(hlo);
  }
  return HandleInstruction(hlo);
}

Status BFloat16NormalizationVisitor::Preprocess(HloInstruction* hlo) {
  computation_ = hlo->parent();
  return OkStatus();
}

}  // namespace

StatusOr<bool> BFloat16Normalization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "BFloat16Normalization::Run(), before:\n" + module->ToString());
  BFloat16NormalizationVisitor visitor(bfloat16_support_, this);
  for (auto* comp : module->MakeComputationPostOrder(execution_threads)) {
    TF_RETURN_IF_ERROR(comp->Accept(&visitor));
  }
  XLA_VLOG_LINES(2,
                 "BFloat16Normalization::Run(), after:\n" + module->ToString());
  if (visitor.changed()) {
    TupleSimplifier tuple_simplifier;
    TF_RETURN_IF_ERROR(tuple_simplifier.Run(module).status());
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }
  return visitor.changed();
}

}  // namespace xla
