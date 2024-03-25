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

#include "xla/service/float_normalization.h"

#include <vector>

#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_graph.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {

namespace {

class FloatNormalizationVisitor : public DfsHloVisitorWithDefault {
 public:
  explicit FloatNormalizationVisitor(const FloatSupport* float_support,
                                     FloatNormalization* float_normalization)
      : computation_(nullptr),
        float_support_(float_support),
        float_normalization_(float_normalization) {}

  bool changed() const { return changed_; }
  Status DefaultAction(HloInstruction* hlo) override;
  Status Preprocess(HloInstruction* hlo) override;

 private:
  // Checks if the HLO uses low-precision in an unsupported way, and if so,
  // inserts conversions between the low- and high-precision types to make it
  // supported.
  Status HandleInstruction(HloInstruction* hlo);

  // Handle instructions with tuple outputs by examining each output
  // independently.
  Status HandleMultipleOutputs(HloInstruction* hlo);

  // Creates a copy of `hlo` with subshapes matching `from` type converted to
  // `to` type. If no matching subshapes are found, returns the original `hlo`.
  absl::StatusOr<HloInstruction*> ConvertType(HloInstruction* hlo,
                                              PrimitiveType from,
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

  // Inserts conversion HLOs to replace the called computations' low-precision
  // operands/outputs to high-precision.
  Status ConvertCalledComputations(
      HloInstruction* hlo,
      absl::Span<HloComputation* const> low_precision_called_comps);

  PrimitiveType LowPrecisionType() const {
    return float_support_->LowPrecisionType();
  }
  PrimitiveType HighPrecisionType() const {
    return float_support_->HighPrecisionType();
  }

  HloComputation* computation_;
  const FloatSupport* float_support_;
  FloatNormalization* float_normalization_;
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

absl::StatusOr<HloInstruction*> FloatNormalizationVisitor::ConvertType(
    HloInstruction* hlo, PrimitiveType from, PrimitiveType to,
    HloComputation* computation) {
  if (CountSubshapesWithMatchingType(hlo->shape(), from) == 0) {
    return hlo;
  }
  // If `hlo` is a convert from `to` to `from`, then we can return its operand,
  // if it is a low-precision->high-precision convert which doesn't do rounding.
  if (hlo->opcode() == HloOpcode::kConvert &&
      hlo->operand(0)->shape().element_type() == to &&
      to == LowPrecisionType() && from == HighPrecisionType()) {
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
            float_normalization_->UpdateLayout(&new_subshape);
            return computation->AddInstruction(
                HloInstruction::CreateConvert(new_subshape, leaf));
          }));
  return new_hlo;
}

Status FloatNormalizationVisitor::InsertConvertAfterOutput(
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

Status FloatNormalizationVisitor::ChangeOutputTypeThenInsertConvertBack(
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
  float_normalization_->UpdateLayout(hlo->mutable_shape());
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

  std::vector<HloInstruction*> conversions_to_simplify;
  for (auto* user : materialized_users) {
    // If the user is a low-precision -> high-precision convert, we can replace
    // it with `hlo`, which has its input changed to high-precision.
    // But we should not replace it immediately, it can lead to type mismatch in
    // the below specific case:
    //         Op
    //   bf16 /  \.
    //    Convert \ bf16
    //  fp32 |    /
    //       Tuple [fp32, bf16]
    // If we run the float normalization pass and replace `Convert` at first,
    // the result will be:
    //         Op
    //    fp32 |
    //      Convert
    //   bf16 / \ bf16
    //       Tuple [fp32, bf16]
    // So we should keep the 'Convert' and replace it after all of the other
    // users has been replaced.
    if (user->opcode() == HloOpcode::kConvert &&
        user->shape().element_type() == to && to == HighPrecisionType() &&
        from == LowPrecisionType()) {
      conversions_to_simplify.emplace_back(user);
    } else {
      TF_RETURN_IF_ERROR(hlo->ReplaceUseWithDifferentShape(user, new_hlo));
    }
  }
  for (auto* convert : conversions_to_simplify) {
    TF_RETURN_IF_ERROR(convert->ReplaceAllUsesWith(hlo));
  }
  if (is_root) {
    computation->set_root_instruction(new_hlo, /*accept_different_shape=*/true);
  }
  changed_ = true;
  return OkStatus();
}

Status FloatNormalizationVisitor::InsertConvertBeforeOperand(
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

Status FloatNormalizationVisitor::ConvertCalledComputations(
    HloInstruction* hlo,
    absl::Span<HloComputation* const> low_precision_called_comps) {
  absl::flat_hash_map<HloComputation*, HloComputation*> cloned_computations;
  for (auto& comp : low_precision_called_comps) {
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
    TF_RETURN_IF_ERROR(InsertConvertAfterOutput(comp->root_instruction(),
                                                LowPrecisionType(),
                                                HighPrecisionType(), comp));
    for (auto* param : comp->parameter_instructions()) {
      // This changes the parameter to high-precision then inserts a convert
      // after it.
      TF_RETURN_IF_ERROR(ChangeOutputTypeThenInsertConvertBack(
          param, LowPrecisionType(), HighPrecisionType(), comp));
    }
  }
  return OkStatus();
}

// Returns true if the called computations of the instruction should not
// be touched by float normalization. In particular, we must not introduce
// float conversions into collective reductions.
bool ShouldAvoidNormalizingComputationsForInstruction(HloInstruction* hlo) {
  return hlo->opcode() == HloOpcode::kAllReduce ||
         hlo->opcode() == HloOpcode::kReduceScatter;
}

Status FloatNormalizationVisitor::HandleMultipleOutputs(HloInstruction* hlo) {
  std::vector<PrimitiveType> operand_types(hlo->operand_count());
  std::vector<PrimitiveType> output_types(hlo->operand_count());
  int64_t high_prec_count = 0;
  int64_t low_prec_count = 0;
  bool has_unsupported_low_prec_operand = false;
  bool has_unsupported_low_prec_output = false;
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    CHECK(hlo->operand(i)->shape().IsArray());
    CHECK(ShapeUtil::GetSubshape(hlo->shape(), {i}).IsArray());
    operand_types[i] = hlo->operand(i)->shape().element_type();
    output_types[i] = ShapeUtil::GetSubshape(hlo->shape(), {i}).element_type();
    if (operand_types[i] == HighPrecisionType()) {
      high_prec_count += 1;
    } else if (operand_types[i] == LowPrecisionType()) {
      low_prec_count += 1;
      if (!float_support_->SupportsLowPrecisionOperand(*hlo, i)) {
        has_unsupported_low_prec_operand = true;
      }
    }
    if (output_types[i] == HighPrecisionType()) {
      high_prec_count += 1;
    } else if (output_types[i] == LowPrecisionType()) {
      low_prec_count += 1;
      if (!float_support_->SupportsLowPrecisionOutput(*hlo)) {
        has_unsupported_low_prec_output = true;
      }
    }
  }

  if (low_prec_count == 0) {
    return OkStatus();
  }

  auto should_convert_operand = [&](int64_t i) {
    if (operand_types[i] != LowPrecisionType()) {
      return false;
    }
    if (!float_support_->SupportsLowPrecisionOperand(*hlo, i)) {
      return true;
    }
    if (float_support_->SupportsMixedPrecisions(*hlo)) {
      return false;
    }
    return has_unsupported_low_prec_operand ||
           has_unsupported_low_prec_output || high_prec_count > 0;
  };

  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    if (should_convert_operand(i)) {
      TF_RETURN_IF_ERROR(InsertConvertBeforeOperand(
          hlo, i, LowPrecisionType(), HighPrecisionType(), computation_));
      high_prec_count += 1;
      low_prec_count -= 1;
    }
  }

  if (!has_unsupported_low_prec_output &&
      (float_support_->SupportsMixedPrecisions(*hlo) || high_prec_count == 0 ||
       low_prec_count == 0)) {
    return OkStatus();
  }

  std::vector<HloComputation*> low_precision_called_comps;
  for (auto* comp : hlo->called_computations()) {
    if (ShouldAvoidNormalizingComputationsForInstruction(hlo)) {
      continue;
    }
    bool comp_has_low_precision = false;
    if (comp->root_instruction()->shape().element_type() ==
        HighPrecisionType()) {
      high_prec_count += 1;
    } else if (comp->root_instruction()->shape().element_type() ==
               LowPrecisionType()) {
      low_prec_count += 1;
      comp_has_low_precision = true;
    }
    for (auto* param : comp->parameter_instructions()) {
      if (param->shape().element_type() == HighPrecisionType()) {
        high_prec_count += 1;
      } else if (param->shape().element_type() == LowPrecisionType()) {
        low_prec_count += 1;
        comp_has_low_precision = true;
      }
    }
    if (comp_has_low_precision) {
      low_precision_called_comps.push_back(comp);
    }
  }

  std::vector<HloInstruction*> materialized_users = hlo->users();
  std::vector<HloInstruction*> output_elements(hlo->operand_count());
  auto original_shape = hlo->shape();
  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    auto subshape = ShapeUtil::GetMutableSubshape(hlo->mutable_shape(), {i});
    if (output_types[i] != LowPrecisionType()) {
      output_elements[i] = computation_->AddInstruction(
          HloInstruction::CreateGetTupleElement(*subshape, hlo, i));
      continue;
    }
    subshape->set_element_type(HighPrecisionType());
    float_normalization_->UpdateLayout(subshape);
    auto gte = computation_->AddInstruction(
        HloInstruction::CreateGetTupleElement(*subshape, hlo, i));
    auto shape = ShapeUtil::ChangeElementType(*subshape, LowPrecisionType());
    float_normalization_->UpdateLayout(&shape);
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
  return ConvertCalledComputations(hlo, low_precision_called_comps);
}

Status FloatNormalizationVisitor::HandleInstruction(HloInstruction* hlo) {
  int high_prec_count = 0;
  int low_prec_count = 0;

  for (int64_t i = 0; i < hlo->operand_count(); ++i) {
    high_prec_count += CountSubshapesWithMatchingType(hlo->operand(i)->shape(),
                                                      HighPrecisionType());
    low_prec_count += CountSubshapesWithMatchingType(hlo->operand(i)->shape(),
                                                     LowPrecisionType());
  }

  high_prec_count +=
      CountSubshapesWithMatchingType(hlo->shape(), HighPrecisionType());
  low_prec_count +=
      CountSubshapesWithMatchingType(hlo->shape(), LowPrecisionType());

  std::vector<HloComputation*> low_precision_called_comps;
  for (auto* comp : hlo->called_computations()) {
    if (ShouldAvoidNormalizingComputationsForInstruction(hlo)) {
      continue;
    }
    bool comp_has_low_precision = false;
    high_prec_count += CountSubshapesWithMatchingType(
        comp->root_instruction()->shape(), HighPrecisionType());
    int64_t low_prec_count_comp_root = CountSubshapesWithMatchingType(
        comp->root_instruction()->shape(), LowPrecisionType());
    if (low_prec_count_comp_root > 0) {
      low_prec_count += low_prec_count_comp_root;
      comp_has_low_precision = true;
    }
    for (auto* param : comp->parameter_instructions()) {
      high_prec_count +=
          CountSubshapesWithMatchingType(param->shape(), HighPrecisionType());
      int64_t low_prec_count_comp_param =
          CountSubshapesWithMatchingType(param->shape(), LowPrecisionType());
      if (low_prec_count_comp_param > 0) {
        low_prec_count += low_prec_count_comp_param;
        comp_has_low_precision = true;
      }
    }
    if (comp_has_low_precision) {
      low_precision_called_comps.push_back(comp);
    }
  }

  // Resolve unsupported low-precision operands.
  for (int i = 0; i < hlo->operand_count(); ++i) {
    int64_t low_prec_count_in_operand = CountSubshapesWithMatchingType(
        hlo->operand(i)->shape(), LowPrecisionType());
    if (low_prec_count_in_operand > 0 &&
        !float_support_->SupportsLowPrecisionOperand(*hlo, i)) {
      TF_RETURN_IF_ERROR(InsertConvertBeforeOperand(
          hlo, i, LowPrecisionType(), HighPrecisionType(), computation_));
      low_prec_count -= low_prec_count_in_operand;
      high_prec_count += low_prec_count_in_operand;
    }
  }

  // Resolve unsupported low-precision output.
  if (!float_support_->SupportsLowPrecisionOutput(*hlo)) {
    int64_t low_prec_count_in_hlo =
        CountSubshapesWithMatchingType(hlo->shape(), LowPrecisionType());
    if (low_prec_count_in_hlo > 0) {
      TF_RETURN_IF_ERROR(ChangeOutputTypeThenInsertConvertBack(
          hlo, LowPrecisionType(), HighPrecisionType(), computation_));
      low_prec_count -= low_prec_count_in_hlo;
      high_prec_count += low_prec_count_in_hlo;
    }
  }

  // Resolve unsupported mixed precision after resolving unsupported
  // low-precision operands and output, because the numbers of low-precision
  // operands/output and high-precision operands/output may have changed.
  if (float_support_->SupportsMixedPrecisions(*hlo) || low_prec_count == 0 ||
      high_prec_count == 0) {
    return OkStatus();
  }
  // See if we can change everything to low-precision.
  if (hlo->called_computations().empty() &&
      CountSubshapesWithMatchingType(hlo->shape(), LowPrecisionType()) ==
          ShapeLeafCount(hlo->shape())) {
    bool can_use_low_prec = true;
    for (int i = 0; i < hlo->operand_count(); ++i) {
      if (CountSubshapesWithMatchingType(hlo->operand(i)->shape(),
                                         LowPrecisionType()) ==
          ShapeLeafCount(hlo->operand(i)->shape())) {
        continue;
      }
      if ((float_support_->EffectiveOperandPrecisionIsLowPrecision(*hlo, i) ||
           float_support_->EffectiveOperandPrecisionIsOutputPrecision(*hlo,
                                                                      i)) &&
          float_support_->SupportsLowPrecisionOperand(*hlo, i)) {
        continue;
      }
      can_use_low_prec = false;
      break;
    }
    if (can_use_low_prec) {
      for (int i = 0; i < hlo->operand_count(); ++i) {
        TF_RETURN_IF_ERROR(InsertConvertBeforeOperand(
            hlo, i, HighPrecisionType(), LowPrecisionType(), computation_));
      }
      return OkStatus();
    }
  }
  TF_RETURN_IF_ERROR(ChangeOutputTypeThenInsertConvertBack(
      hlo, LowPrecisionType(), HighPrecisionType(), computation_));
  for (int i = 0; i < hlo->operand_count(); ++i) {
    TF_RETURN_IF_ERROR(InsertConvertBeforeOperand(
        hlo, i, LowPrecisionType(), HighPrecisionType(), computation_));
  }
  return ConvertCalledComputations(hlo, low_precision_called_comps);
}

Status FloatNormalizationVisitor::DefaultAction(HloInstruction* hlo) {
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

Status FloatNormalizationVisitor::Preprocess(HloInstruction* hlo) {
  computation_ = hlo->parent();
  return OkStatus();
}

// We must avoid normalizing computations that have non-normalizing users
// (e.g., all-reduce's computations should not be normalized). If a
// computation is shared between normalizing and non-normalizing users, we will
// clone the computation for the non-normalizing users so that it can be left
// unchanged. This function clones the shared computations and returns the set
// of non-normalizing computations that must be skipped by the visitor.
absl::flat_hash_set<HloComputation*>
CloneComputationsForNonNormalizingInstructions(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::unique_ptr<CallGraph> call_graph =
      CallGraph::Build(module, execution_threads);

  absl::flat_hash_set<HloComputation*> computations_to_skip;
  for (const CallGraphNode& node : call_graph->nodes()) {
    bool has_normalizing_users = false;
    bool has_users_to_skip_normalization = false;
    for (const CallSite& site : node.caller_callsites()) {
      if (ShouldAvoidNormalizingComputationsForInstruction(
              site.instruction())) {
        has_users_to_skip_normalization = true;
      } else {
        has_normalizing_users = true;
      }
    }
    // If the computation is only used by normalizing users or only by
    // non-normalizing users, then we do not clone.
    if (!has_users_to_skip_normalization) {
      continue;
    }
    if (!has_normalizing_users) {
      computations_to_skip.insert(node.computation());
      continue;
    }
    // Otherwise, we create a clone and replace the normalizing instructions'
    // computations with the clone.
    HloComputation* clone = module->DeepCloneComputation(node.computation());
    for (const CallSite& site : node.caller_callsites()) {
      if (ShouldAvoidNormalizingComputationsForInstruction(
              site.instruction())) {
        site.instruction()->ReplaceCalledComputations(
            [&](HloComputation* called) {
              return called == node.computation() ? clone : called;
            });
      }
    }
    computations_to_skip.insert(clone);
  }
  return computations_to_skip;
}
}  // namespace

absl::StatusOr<bool> FloatNormalization::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2, "FloatNormalization::Run() for " +
                        primitive_util::LowercasePrimitiveTypeName(
                            float_support_->LowPrecisionType()) +
                        ", before:\n" + module->ToString());
  auto computations_to_visit =
      module->MakeComputationPostOrder(execution_threads);
  auto computations_to_skip =
      CloneComputationsForNonNormalizingInstructions(module, execution_threads);

  FloatNormalizationVisitor visitor(float_support_, this);
  for (auto* comp : computations_to_visit) {
    if (computations_to_skip.contains(comp)) continue;
    TF_RETURN_IF_ERROR(comp->Accept(&visitor));
  }
  XLA_VLOG_LINES(2, "FloatNormalization::Run() for " +
                        primitive_util::LowercasePrimitiveTypeName(
                            float_support_->LowPrecisionType()) +
                        ", after:\n" + module->ToString());
  if (visitor.changed()) {
    TupleSimplifier tuple_simplifier;
    TF_RETURN_IF_ERROR(tuple_simplifier.Run(module).status());
    HloDCE dce;
    TF_RETURN_IF_ERROR(dce.Run(module).status());
  }
  return visitor.changed();
}

}  // namespace xla
