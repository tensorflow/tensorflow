/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/gpu/matmul_indexing_utils.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/permutation_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<std::vector<int64_t>> GetNonContractingDims(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contracting_dims) {
  auto nc = ::xla::GetNonContractingDims(shape.dimensions().size(),
                                         contracting_dims, batch_dims);

  TF_RET_CHECK(batch_dims.size() + contracting_dims.size() + nc.size() ==
               shape.dimensions().size());
  return std::vector<int64_t>(nc.begin(), nc.end());
}

const tsl::protobuf::RepeatedField<int64_t>& BatchDimensionsForOperand(
    const HloInstruction& dot, const int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    return dimension_numbers.lhs_batch_dimensions();
  }
  return dimension_numbers.rhs_batch_dimensions();
}

const tsl::protobuf::RepeatedField<int64_t>& ContractingDimensionsForOperand(
    const HloInstruction& dot, const int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  return operand_number == 0 ? dimension_numbers.lhs_contracting_dimensions()
                             : dimension_numbers.rhs_contracting_dimensions();
}

absl::StatusOr<int64_t> ContractingDimensionIndex(const HloInstruction& dot,
                                                  const int operand_number) {
  const DotDimensionNumbers& dimension_numbers = dot.dot_dimension_numbers();
  if (operand_number == 0) {
    TF_RET_CHECK(dimension_numbers.lhs_contracting_dimensions().size() == 1);
    return dimension_numbers.lhs_contracting_dimensions(0);
  }
  TF_RET_CHECK(dimension_numbers.rhs_contracting_dimensions().size() == 1);
  return dimension_numbers.rhs_contracting_dimensions(0);
}

absl::StatusOr<int64_t> NonContractingDimensionIndex(const HloInstruction& dot,
                                                     const int operand_number) {
  TF_ASSIGN_OR_RETURN(int64_t contracting_dim,
                      ContractingDimensionIndex(dot, operand_number));
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> non_contracting_dims,
      GetNonContractingDims(dot.operand(operand_number)->shape(),
                            BatchDimensionsForOperand(dot, operand_number),
                            {contracting_dim}));
  TF_RET_CHECK(non_contracting_dims.size() == 1);
  return non_contracting_dims.front();
}

DotOperandDims::DotOperandDims(Shape shape,
                               absl::Span<const int64_t> batch_dims,
                               absl::Span<const int64_t> non_contracting_dims,
                               absl::Span<const int64_t> contracting_dims)
    : shape_(shape) {
  dim_numbers_[kBatch].assign(batch_dims.begin(), batch_dims.end());
  dim_numbers_[kNonContracting].assign(non_contracting_dims.begin(),
                                       non_contracting_dims.end());
  dim_numbers_[kContracting].assign(contracting_dims.begin(),
                                    contracting_dims.end());
}

absl::StatusOr<DotOperandDims> DotOperandDims::FromDot(
    const HloInstruction* dot, int operand_idx) {
  TF_RET_CHECK(operand_idx == 0 || operand_idx == 1);
  const Shape& shape = dot->operand(operand_idx)->shape();
  const auto& batch_dims = BatchDimensionsForOperand(*dot, operand_idx);
  const auto& contracting_dims =
      ContractingDimensionsForOperand(*dot, operand_idx);
  TF_ASSIGN_OR_RETURN(
      std::vector<int64_t> non_contracting_dims,
      GetNonContractingDims(shape, batch_dims, contracting_dims));
  return DotOperandDims(shape, batch_dims, non_contracting_dims,
                        contracting_dims);
}

absl::StatusOr<DotDimensionNumbers> DotOperandDims::IntoDotDimensionNumbers(
    const DotOperandDims& lhs_dims, const DotOperandDims& rhs_dims) {
  DotDimensionNumbers dot_dim_numbers;
  TF_RET_CHECK(lhs_dims.Indices(kBatch).size() ==
               rhs_dims.Indices(kBatch).size());
  TF_RET_CHECK(lhs_dims.Indices(kContracting).size() ==
               rhs_dims.Indices(kContracting).size());
  dot_dim_numbers.mutable_lhs_batch_dimensions()->Assign(
      lhs_dims.Indices(kBatch).begin(), lhs_dims.Indices(kBatch).end());
  dot_dim_numbers.mutable_rhs_batch_dimensions()->Assign(
      rhs_dims.Indices(kBatch).begin(), rhs_dims.Indices(kBatch).end());
  dot_dim_numbers.mutable_lhs_contracting_dimensions()->Assign(
      lhs_dims.Indices(kContracting).begin(),
      lhs_dims.Indices(kContracting).end());
  dot_dim_numbers.mutable_rhs_contracting_dimensions()->Assign(
      rhs_dims.Indices(kContracting).begin(),
      rhs_dims.Indices(kContracting).end());
  return dot_dim_numbers;
}

// Computes the output shape of the dot instruction.
absl::StatusOr<Shape> DotOperandDims::IntoOutputShape(
    PrimitiveType element_type, const DotOperandDims& lhs_dims,
    const DotOperandDims& rhs_dims) {
  TF_RET_CHECK(lhs_dims.Indices(kBatch).size() ==
               rhs_dims.Indices(kBatch).size());
  TF_RET_CHECK(lhs_dims.Indices(kContracting).size() ==
               rhs_dims.Indices(kContracting).size());
  std::vector<int64_t> output_dimensions;
  std::vector<bool> output_dynamic_dimensions;

  for (int64_t i = 0; i < lhs_dims.Indices(kBatch).size(); ++i) {
    int64_t lhs_batch_dim = lhs_dims.Indices(kBatch)[i];
    int64_t rhs_batch_dim = rhs_dims.Indices(kBatch)[i];
    TF_RET_CHECK(lhs_dims.shape_.dimensions(lhs_batch_dim) ==
                 rhs_dims.shape_.dimensions(rhs_batch_dim));
    output_dimensions.push_back(lhs_dims.shape_.dimensions(lhs_batch_dim));
    TF_RET_CHECK(lhs_dims.shape_.is_dynamic_dimension(lhs_batch_dim) ==
                 rhs_dims.shape_.is_dynamic_dimension(rhs_batch_dim));
    output_dynamic_dimensions.push_back(
        lhs_dims.shape_.is_dynamic_dimension(lhs_batch_dim));
  }
  for (auto& operand : {lhs_dims, rhs_dims}) {
    for (int64_t nc_dim : operand.Indices(kNonContracting)) {
      output_dimensions.push_back(operand.shape_.dimensions(nc_dim));
      output_dynamic_dimensions.push_back(
          operand.shape_.is_dynamic_dimension(nc_dim));
    }
  }
  TF_ASSIGN_OR_RETURN(Shape output_shape, ShapeUtil::MakeValidatedShape(
                                              element_type, output_dimensions));
  for (int64_t i = 0; i < output_dynamic_dimensions.size(); ++i) {
    output_shape.set_dynamic_dimension(i, output_dynamic_dimensions[i]);
  }
  return output_shape;
}

std::vector<int64_t> DotOperandDims::DimensionSizes(Category category) const {
  std::vector<int64_t> dim_sizes;
  dim_sizes.reserve(dim_numbers_[category].size());
  absl::c_transform(dim_numbers_[category], std::back_inserter(dim_sizes),
                    [this](int64_t dim) { return shape_.dimensions(dim); });
  return dim_sizes;
}

void DotOperandDims::Permute(absl::Span<const int64_t> permutation) {
  auto inversed_permutation = InversePermutation(permutation);
  shape_ = ShapeUtil::PermuteDimensions(permutation, shape_);
  for (auto& dim_numbers : dim_numbers_) {
    for (auto& dim : dim_numbers) {
      dim = inversed_permutation[dim];
    }
  }
}

absl::Status DotOperandDims::Collapse(Category category, bool remove_if_empty) {
  const auto& dims = dim_numbers_[category];
  if (dims.empty()) {
    return absl::OkStatus();
  }
  int64_t min_dim = *absl::c_min_element(dims);
  int64_t max_dim = *absl::c_max_element(dims);
  if (max_dim - min_dim + 1 != dims.size()) {
    return absl::InvalidArgumentError(
        "Attempting to collapse non-consecutive dimensions");
  }
  const int64_t total_size = absl::c_accumulate(
      dims, int64_t{1},
      [&](int64_t size, int64_t idx) { return size * shape_.dimensions(idx); });
  if (total_size == 1 && remove_if_empty) {
    return EraseDimensions(min_dim, max_dim + 1);
  }
  bool is_dynamic = absl::c_any_of(
      dims, [this](int64_t dim) { return shape_.is_dynamic_dimension(dim); });
  shape_.set_dimensions(min_dim, total_size, is_dynamic);
  return EraseDimensions(min_dim + 1, max_dim + 1);
}

absl::Status DotOperandDims::EraseDimensions(int64_t start, int64_t end) {
  TF_RET_CHECK(start >= 0);
  TF_RET_CHECK(start <= end);
  TF_RET_CHECK(end <= shape_.dimensions().size());
  for (auto category : {kBatch, kNonContracting, kContracting}) {
    auto& dims = dim_numbers_[category];
    auto write_iter = dims.begin();
    for (int64_t dim_val : dims) {
      if (dim_val < start) {
        *write_iter++ = dim_val;
      } else if (dim_val >= end) {
        *write_iter++ = dim_val - (end - start);
      }
    }
    dims.erase(write_iter, dims.end());
  }
  shape_ = ShapeUtil::FilterDimensions(
      [&](int64_t dim) { return dim < start || dim >= end; }, shape_);
  return absl::OkStatus();
}

absl::StatusOr<int64_t> DotOperandDims::LocalIndex(
    Category category, int64_t global_dim_idx) const {
  const auto& dims = dim_numbers_[category];
  auto iter =
      absl::c_find_if(dims, [&](int64_t d) { return d == global_dim_idx; });
  TF_RET_CHECK(iter != dims.end());
  return iter - dims.begin();
}

}  // namespace gpu
}  // namespace xla
