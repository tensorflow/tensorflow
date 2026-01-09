/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/dot_decomposer.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

bool IsCanonical(Shape operand_shape,
                 absl::Span<const int64_t> contracting_dimensions,
                 absl::Span<const int64_t> batch_dimensions) {
  // A dot is not canonical if there is more than one contracting dimension.
  if (contracting_dimensions.size() != 1) {
    return false;
  }
  // A dot is not canonical if it has more than one non-contracting
  // dimension.
  if (batch_dimensions.size() + 2 < operand_shape.dimensions().size()) {
    return false;
  }
  if (batch_dimensions.empty() && contracting_dimensions.empty()) {
    return false;
  }
  // Check that batch dims, if present, are canonical.
  std::vector<int64_t> canonical_batch_dims(batch_dimensions.size());
  absl::c_iota(canonical_batch_dims, 0);
  if (!absl::c_equal(batch_dimensions, canonical_batch_dims)) {
    return false;
  }
  return true;
}

HloInstruction* CanonicalizeOperand(
    HloInstruction* operand, absl::Span<const int64_t> original_batch_dims,
    absl::Span<const int64_t> original_contracting_dims,
    absl::Span<const int64_t> canonical_batch_dims,
    const std::vector<bool>& canonical_batch_dynamic_dims,
    tsl::protobuf::RepeatedField<int64_t>* canonical_contracting_dims,
    bool contracting_dim_as_most_minor) {
  const Shape& operand_shape = operand->shape();
  const int64_t rank = operand_shape.dimensions().size();
  const int64_t num_non_contracting_dims =
      rank - original_batch_dims.size() - original_contracting_dims.size();
  std::vector<int64_t> non_contracting_dims;
  non_contracting_dims.reserve(num_non_contracting_dims);
  int64_t contracting_size = 1;
  bool contracting_dynamic = false;
  int64_t non_contracting_size = 1;
  bool non_contracting_dynamic = false;
  for (int64_t i = 0; i < rank; ++i) {
    if (absl::c_linear_search(original_contracting_dims, i)) {
      contracting_size *= operand_shape.dimensions(i);
      contracting_dynamic |= operand_shape.is_dynamic_dimension(i);
    } else if (!absl::c_linear_search(original_batch_dims, i)) {
      non_contracting_dims.push_back(i);
      non_contracting_size *= operand_shape.dimensions(i);
      non_contracting_dynamic |= operand_shape.is_dynamic_dimension(i);
    }
  }
  std::vector<int64_t> permutation;
  permutation.reserve(rank);
  permutation.insert(permutation.end(), original_batch_dims.begin(),
                     original_batch_dims.end());
  if (contracting_dim_as_most_minor) {
    permutation.insert(permutation.end(), non_contracting_dims.begin(),
                       non_contracting_dims.end());
    permutation.insert(permutation.end(), original_contracting_dims.begin(),
                       original_contracting_dims.end());
  } else {
    permutation.insert(permutation.end(), original_contracting_dims.begin(),
                       original_contracting_dims.end());
    permutation.insert(permutation.end(), non_contracting_dims.begin(),
                       non_contracting_dims.end());
  }
  HloComputation* computation = operand->parent();
  HloInstruction* transpose_op = computation->AddInstruction(
      HloInstruction::CreateTranspose(
          ShapeUtil::PermuteDimensionsIgnoringLayout(permutation,
                                                     operand_shape),
          operand, permutation),
      &operand->metadata());

  std::vector<int64_t> reshape_dims(canonical_batch_dims.begin(),
                                    canonical_batch_dims.end());
  std::vector<bool> reshape_dynamic_dims(canonical_batch_dynamic_dims.begin(),
                                         canonical_batch_dynamic_dims.end());
  if (contracting_dim_as_most_minor) {
    canonical_contracting_dims->Add(canonical_batch_dims.size() +
                                    (non_contracting_size != 1 ? 1 : 0));
    if (non_contracting_size != 1) {
      reshape_dims.push_back(non_contracting_size);
      reshape_dynamic_dims.push_back(non_contracting_dynamic);
    }
    reshape_dims.push_back(contracting_size);
    reshape_dynamic_dims.push_back(contracting_dynamic);
  } else {
    canonical_contracting_dims->Add(canonical_batch_dims.size());
    reshape_dims.push_back(contracting_size);
    reshape_dynamic_dims.push_back(contracting_dynamic);
    if (non_contracting_size != 1) {
      reshape_dims.push_back(non_contracting_size);
      reshape_dynamic_dims.push_back(non_contracting_dynamic);
    }
  }
  // Reshape the contracting and non-contracting dimensions together.
  return computation->AddInstruction(
      HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(operand_shape.element_type(), reshape_dims,
                               reshape_dynamic_dims),
          transpose_op),
      &transpose_op->metadata());
}

// Convert a dot into a canonical form;
// * Non-contracting dimensions are reshaped together,
// * Contracting dimensions are reshaped together,
// * Batch dimensions are the most major dimensions.
// This requires transposing and reshaping of the lhs and rhs, and reshaping the
// output batch to the original shape.
absl::Status CanonicalizeDot(HloDotInstruction* original_dot) {
  auto computation = original_dot->parent();
  const auto& original_dnums = original_dot->dot_dimension_numbers();
  const int64_t num_batch_dims = original_dnums.lhs_batch_dimensions_size();
  std::vector<int64_t> canonical_batch_dims;
  canonical_batch_dims.reserve(num_batch_dims);
  std::vector<bool> canonical_batch_dynamic_dims;
  canonical_batch_dynamic_dims.reserve(num_batch_dims);

  HloInstruction* lhs_operand = original_dot->mutable_operand(0);
  HloInstruction* rhs_operand = original_dot->mutable_operand(1);
  const auto& lhs_shape = lhs_operand->shape();
  const int64_t lhs_rank = lhs_shape.dimensions().size();
  for (int64_t i = 0; i < lhs_rank; ++i) {
    if (absl::c_linear_search(original_dnums.lhs_batch_dimensions(), i)) {
      canonical_batch_dims.push_back(lhs_shape.dimensions(i));
      canonical_batch_dynamic_dims.push_back(lhs_shape.is_dynamic_dimension(i));
    }
  }
  DotDimensionNumbers canonical_dnums;
  std::vector<int64_t> canonical_dot_dims = canonical_batch_dims;
  std::vector<bool> canonical_dot_dynamic_dims = canonical_batch_dynamic_dims;
  for (int64_t i = 0; i < num_batch_dims; ++i) {
    canonical_dnums.add_lhs_batch_dimensions(i);
    canonical_dnums.add_rhs_batch_dimensions(i);
  }

  // The canonical form of the lhs is
  // [BatchDims, NonContractingDimsProduct, ContractingsDimsProduct]
  // However, [ContractingDim, NonContractingDim] is considered canonical too.
  // If NonContractingDimsProduct is 1, it is omitted.

  HloInstruction* reshaped_lhs =
      CanonicalizeOperand(lhs_operand, original_dnums.lhs_batch_dimensions(),
                          original_dnums.lhs_contracting_dimensions(),
                          canonical_batch_dims, canonical_batch_dynamic_dims,
                          canonical_dnums.mutable_lhs_contracting_dimensions(),
                          /*contracting_dim_as_most_minor=*/true);
  const auto& canonical_lhs_shape = reshaped_lhs->shape();
  const auto& canonical_lhs_non_contracting_dims =
      GetNonContractingDims(canonical_lhs_shape.dimensions().size(),
                            canonical_dnums.lhs_batch_dimensions(),
                            canonical_dnums.lhs_contracting_dimensions());
  // At this point of canonicalization, there are 0 or 1 non-contracting dims.
  if (canonical_lhs_non_contracting_dims.size() == 1) {
    int64_t lhs_non_contracting_dim = canonical_lhs_non_contracting_dims.at(0);
    canonical_dot_dims.push_back(
        canonical_lhs_shape.dimensions(lhs_non_contracting_dim));
    canonical_dot_dynamic_dims.push_back(
        canonical_lhs_shape.is_dynamic_dimension(lhs_non_contracting_dim));
  }

  // The canonical form of the rhs is
  // [BatchDims, ContractingsDimsProduct, NonContractingDimsProduct]
  // However, [NonContractingDim, ContractingDim] is considered canonical too.
  // If NonContractingDimsProduct is 1, it is omitted.
  HloInstruction* reshaped_rhs =
      CanonicalizeOperand(rhs_operand, original_dnums.rhs_batch_dimensions(),
                          original_dnums.rhs_contracting_dimensions(),
                          canonical_batch_dims, canonical_batch_dynamic_dims,
                          canonical_dnums.mutable_rhs_contracting_dimensions(),
                          /*contracting_dim_as_most_minor=*/false);

  const auto& canonical_rhs_shape = reshaped_rhs->shape();
  const auto& canonical_rhs_non_contracting_dims =
      GetNonContractingDims(canonical_rhs_shape.dimensions().size(),
                            canonical_dnums.rhs_batch_dimensions(),
                            canonical_dnums.rhs_contracting_dimensions());
  // At this point of canonicalization, there are 0 or 1 non-contracting dims.
  if (canonical_rhs_non_contracting_dims.size() == 1) {
    int64_t rhs_non_contracting_dim = canonical_rhs_non_contracting_dims.at(0);
    canonical_dot_dims.push_back(
        canonical_rhs_shape.dimensions(rhs_non_contracting_dim));
    canonical_dot_dynamic_dims.push_back(
        canonical_rhs_shape.is_dynamic_dimension(rhs_non_contracting_dim));
  }

  HloInstruction* dot = computation->AddInstruction(HloInstruction::CreateDot(
      ShapeUtil::MakeShape(original_dot->shape().element_type(),
                           canonical_dot_dims, canonical_dot_dynamic_dims),
      reshaped_lhs, reshaped_rhs, canonical_dnums,
      original_dot->precision_config()));
  original_dot->SetupDerivedInstruction(dot);

  std::unique_ptr<HloInstruction> replacement =
      HloInstruction::CreateReshape(original_dot->shape(), dot);
  VLOG(3) << "Canonicalizing dot:\n"
          << "\t old: " << original_dot->ToString() << "\n"
          << "\t new: " << dot->ToString() << "\n"
          << "\t   -> " << replacement->ToString();
  return computation->ReplaceWithNewInstruction(original_dot,
                                                std::move(replacement));
}

}  // namespace

absl::StatusOr<bool> DotDecomposer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Gather all Non-canonical Dot operations.
  std::vector<HloInstruction*> non_canonical_dots;
  for (auto* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kDot) {
        continue;
      }
      const DotDimensionNumbers& dnums = instruction->dot_dimension_numbers();
      if (!IsCanonical(instruction->operand(0)->shape(),
                       dnums.lhs_contracting_dimensions(),
                       dnums.lhs_batch_dimensions()) ||
          !IsCanonical(instruction->operand(1)->shape(),
                       dnums.rhs_contracting_dimensions(),
                       dnums.rhs_batch_dimensions())) {
        non_canonical_dots.push_back(instruction);
      }
    }
  }
  bool changed = false;
  for (auto* dot : non_canonical_dots) {
    TF_RETURN_IF_ERROR(CanonicalizeDot(Cast<HloDotInstruction>(dot)));
    changed = true;
  }
  return changed;
}

}  // namespace xla
