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

#include <cstddef>
#include <cstdint>
#include <optional>
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
#include "xla/permutation_util.h"
#include "xla/service/gpu/matmul_indexing_utils.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {
using DotOperandDims = gpu::DotOperandDims;

absl::StatusOr<std::optional<std::vector<int64_t>>>
ComputeContractingDimPermutation(HloDotInstruction* dot) {
  std::optional<std::vector<int64_t>> result;

  for (const SparsityDescriptor& descriptor : dot->sparsity()) {
    int operand = descriptor.index();
    int dim = descriptor.dimension();
    TF_ASSIGN_OR_RETURN(auto dims, DotOperandDims::FromDot(dot, operand));
    TF_ASSIGN_OR_RETURN(int64_t local_dim,
                        dims.LocalIndex(DotOperandDims::kContracting, dim));
    std::vector<int64_t> permutation;
    for (size_t i = 0; i < dims.DimensionCount(DotOperandDims::kContracting);
         ++i) {
      if (i != local_dim) {
        permutation.push_back(i);
      }
    }
    permutation.push_back(local_dim);
    TF_RET_CHECK(!result || absl::c_equal(*result, permutation));
    result = permutation;
  }

  return result;
}

struct DotOperand {
  HloInstruction* operand = nullptr;
  HloInstruction* sparsity_meta = nullptr;
  std::optional<SparsityDescriptor> sparsity_descriptor;
  DotOperandDims dims;
};

int64_t SuggestDimensionIndex(
    const DotOperandDims& dims,
    absl::Span<const DotOperandDims::Category> categories_order,
    DotOperandDims::Category category_to_insert) {
  // Assume that the dimensions are sorted in the categories_order.
  int64_t index = 0;
  for (auto cat : categories_order) {
    if (cat == category_to_insert) {
      return index;
    }
    index += dims.DimensionCount(cat);
  }
  return index;
}

absl::StatusOr<DotOperand> CanonicalizeDotOperand(
    HloDotInstruction* dot, int operand_idx,
    std::optional<std::vector<int64_t>> contracting_dim_permutation,
    absl::Span<const DotOperandDims::Category> categories_order) {
  auto computation = dot->parent();
  HloInstruction* operand = dot->mutable_operand(operand_idx);
  TF_ASSIGN_OR_RETURN(auto operand_dims,
                      DotOperandDims::FromDot(dot, operand_idx));

  // Build a permutation. The order of the categories comes from the parameter,
  // and potentially we apply the permutation to the contracting dimensions.
  std::vector<int64_t> transpose;
  transpose.reserve(operand_dims.shape().dimensions().size());
  for (auto cat : categories_order) {
    if (cat == DotOperandDims::kContracting && contracting_dim_permutation) {
      // If the contracting dimension order is specified, permute the
      // contracting dimensions.
      const auto& dims =
          Permute(operand_dims.Indices(cat), *contracting_dim_permutation);
      transpose.insert(transpose.end(), dims.begin(), dims.end());
    } else {
      transpose.insert(transpose.end(), operand_dims.Indices(cat).begin(),
                       operand_dims.Indices(cat).end());
    }
  }
  operand_dims.Permute(transpose);
  HloInstruction* transposed =
      operand_dims.shape() == operand->shape()
          ? operand
          : computation->AddInstruction(
                HloInstruction::CreateTranspose(operand_dims.shape(), operand,
                                                transpose),
                &operand->metadata());

  // * Batch dimensions are not collapsed.
  // * Non-contracting dimensions are collapsed and removed if empty.
  // * Contracting dimensions are collapsed but not removed if empty.
  TF_RETURN_IF_ERROR(operand_dims.Collapse(DotOperandDims::kNonContracting,
                                           /*remove_if_empty=*/true));
  TF_RETURN_IF_ERROR(operand_dims.Collapse(DotOperandDims::kContracting,
                                           /*remove_if_empty=*/false));
  if (operand_dims.DimensionCount(DotOperandDims::kContracting) == 0) {
    // If there are no contracting dimensions, insert a dimension of size 1.
    const int64_t new_dim_idx = SuggestDimensionIndex(
        operand_dims, categories_order, DotOperandDims::kContracting);
    TF_RETURN_IF_ERROR(operand_dims.InsertDimension(
        DotOperandDims::kContracting, new_dim_idx, 1));
  }
  HloInstruction* reshape =
      operand_dims.shape() == transposed->shape()
          ? transposed
          : computation->AddInstruction(
                HloInstruction::CreateReshape(operand_dims.shape(), transposed),
                &transposed->metadata());
  DotOperand result;
  result.operand = reshape;

  // Now also do the same transformations for the sparsity metadata.
  const auto& sparsity = dot->sparsity();
  if (auto iter = absl::c_find_if(sparsity,
                                  [&](const SparsityDescriptor& descriptor) {
                                    return descriptor.index() == operand_idx;
                                  });
      iter != sparsity.end()) {
    SparsityDescriptor descriptor = *iter;
    descriptor.set_dimension(
        operand_dims.Indices(DotOperandDims::kContracting).back());
    result.sparsity_descriptor = descriptor;
    HloInstruction* meta = dot->mutable_operand(HloDotInstruction::kOperands +
                                                (iter - sparsity.begin()));
    HloInstruction* meta_transpose = computation->AddInstruction(
        HloInstruction::CreateTranspose(
            ShapeUtil::PermuteDimensions(transpose, meta->shape()), meta,
            transpose),
        &meta->metadata());
    TF_ASSIGN_OR_RETURN(
        Shape result_shape,
        ShapeInference::InferSparseDotMetadataShape(
            result.operand->shape(),
            operand_dims.Indices(DotOperandDims::kContracting), descriptor));
    result.sparsity_meta = computation->AddInstruction(
        HloInstruction::CreateReshape(result_shape, meta_transpose),
        &meta->metadata());
  }
  result.dims = operand_dims;
  return result;
}

// Convert a dot into a canonical form;
// * Non-contracting dimensions are reshaped together,
// * Contracting dimensions are reshaped together,
// * Batch dimensions are the most major dimensions.
// This requires transposing and reshaping of the lhs and rhs, and reshaping the
// output non-contracting dimensions to the original shape.
absl::Status CanonicalizeDot(HloDotInstruction* original_dot) {
  auto computation = original_dot->parent();
  // Check whether there is special requirements for the contracting dimension
  // order (e.g. for sparse operands, the sparse dimension should be the last).
  TF_ASSIGN_OR_RETURN(
      std::optional<std::vector<int64_t>> contracting_dim_permutation,
      ComputeContractingDimPermutation(original_dot));

  TF_ASSIGN_OR_RETURN(
      DotOperand lhs,
      CanonicalizeDotOperand(
          original_dot, 0, contracting_dim_permutation,
          {DotOperandDims::kBatch, DotOperandDims::kNonContracting,
           DotOperandDims::kContracting}));
  TF_ASSIGN_OR_RETURN(DotOperand rhs,
                      CanonicalizeDotOperand(
                          original_dot, 1, contracting_dim_permutation,
                          {DotOperandDims::kBatch, DotOperandDims::kContracting,
                           DotOperandDims::kNonContracting}));

  TF_ASSIGN_OR_RETURN(Shape out_shape, DotOperandDims::IntoOutputShape(
                                           original_dot->shape().element_type(),
                                           lhs.dims, rhs.dims));
  TF_ASSIGN_OR_RETURN(
      DotDimensionNumbers dnums,
      DotOperandDims::IntoDotDimensionNumbers(lhs.dims, rhs.dims));

  std::vector<SparsityDescriptor> sparsity;
  std::vector<HloInstruction*> sparse_meta;
  for (auto& operand : {lhs, rhs}) {
    if (operand.sparsity_descriptor.has_value()) {
      sparsity.push_back(operand.sparsity_descriptor.value());
    }
    if (operand.sparsity_meta != nullptr) {
      sparse_meta.push_back(operand.sparsity_meta);
    }
  }

  HloInstruction* dot = computation->AddInstruction(HloInstruction::CreateDot(
      out_shape, lhs.operand, rhs.operand, dnums,
      original_dot->precision_config(), sparsity, sparse_meta));
  original_dot->SetupDerivedInstruction(dot);

  HloInstruction* reshape =
      original_dot->shape() == dot->shape()
          ? dot
          : computation->AddInstruction(
                HloInstruction::CreateReshape(original_dot->shape(), dot));
  VLOG(3) << "Canonicalizing dot:\n"
          << "\t old: " << original_dot->ToString() << "\n"
          << "\t new: " << dot->ToString() << "\n"
          << "\t   -> " << reshape->ToString();
  return computation->ReplaceInstruction(original_dot, reshape);
}

}  // namespace

absl::StatusOr<bool> DotDecomposer::Run(
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
      // A dot it not canonical if there is more than one contracting dimension.
      if (dnums.lhs_contracting_dimensions_size() != 1) {
        non_canonical_dots.push_back(instruction);
        continue;
      }
      // A dot is not canonical if it has more than one non-contracting
      // dimension.
      if (dnums.lhs_batch_dimensions_size() + 2 <
              instruction->operand(0)->shape().dimensions().size() ||
          dnums.rhs_batch_dimensions_size() + 2 <
              instruction->operand(1)->shape().dimensions().size()) {
        non_canonical_dots.push_back(instruction);
        continue;
      }
      if (dnums.lhs_batch_dimensions().empty() &&
          dnums.lhs_contracting_dimensions().empty()) {
        non_canonical_dots.push_back(instruction);
        continue;
      }
      // Check that batch dims, if present, are canonical.
      std::vector<int64_t> canonical_batch_dims(
          dnums.lhs_batch_dimensions_size());
      absl::c_iota(canonical_batch_dims, 0);
      if (!absl::c_equal(dnums.lhs_batch_dimensions(), canonical_batch_dims) ||
          !absl::c_equal(dnums.rhs_batch_dimensions(), canonical_batch_dims)) {
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
