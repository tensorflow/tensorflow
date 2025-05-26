/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_MATMUL_INDEXING_UTILS_H_
#define XLA_SERVICE_GPU_MATMUL_INDEXING_UTILS_H_

#include <array>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// Ordered non-contracting dimensions for a dot instruction operand.
absl::StatusOr<std::vector<int64_t>> GetNonContractingDims(
    const Shape& shape, absl::Span<const int64_t> batch_dims,
    absl::Span<const int64_t> contracting_dims);

// Batch dimensions of an operand of a dot instruction.
// Just an unified accessor to lhs_batch_dimensions and rhs_batch_dimensions.
const tsl::protobuf::RepeatedField<int64_t>& BatchDimensionsForOperand(
    const HloInstruction& dot, int operand_number);

// Contracting dimensions of an operand of a dot instruction.
// Just an unified accessor to lhs_contracting_dimensions and
// rhs_contracting_dimensions.
const tsl::protobuf::RepeatedField<int64_t>& ContractingDimensionsForOperand(
    const HloInstruction& dot, int operand_number);

// Index of the only contracting dimension of dot instruction operand.
absl::StatusOr<int64_t> ContractingDimensionIndex(const HloInstruction& dot,
                                                  int operand_number);

// Index of the only non-contracting dimension of dot instruction operand.
absl::StatusOr<int64_t> NonContractingDimensionIndex(const HloInstruction& dot,
                                                     int operand_number);

// A class to handle the dimensions of an operand of a dot instruction.
class DotOperandDims {
 public:
  DotOperandDims() = default;
  DotOperandDims(Shape shape, absl::Span<const int64_t> batch_dims,
                 absl::Span<const int64_t> non_contracting_dims,
                 absl::Span<const int64_t> contracting_dims);

  enum Category { kBatch, kNonContracting, kContracting };
  // Creates a DotOperandDims from a dot instruction and operand index (0 or 1).
  static absl::StatusOr<DotOperandDims> FromDot(HloInstruction* dot,
                                                int operand_idx);
  // Converts two DotOperandDims to a DotDimensionNumbers.
  static absl::StatusOr<DotDimensionNumbers> IntoDotDimensionNumbers(
      const DotOperandDims& lhs_dims, const DotOperandDims& rhs_dims);
  // Computes the output shape of the dot instruction.
  static absl::StatusOr<Shape> IntoOutputShape(PrimitiveType element_type,
                                               const DotOperandDims& lhs_dims,
                                               const DotOperandDims& rhs_dims);
  // Returns the indices of the dimensions of the category.
  absl::Span<const int64_t> Indices(Category category) const {
    return dim_numbers_[category];
  }
  // Returns the category size (number of dimensions).
  int64_t DimensionCount(Category category) const {
    return dim_numbers_[category].size();
  }
  // Returns the dimension sizes of the category.
  std::vector<int64_t> DimensionSizes(Category category) const;
  // Permute the dimensions of the category.
  // The permutation is in the same format as you'd pass to the transpose
  // instruction. The corresponding dimension numbers are updated.
  void Permute(absl::Span<const int64_t> permutation);
  // Collapses the dimensions of the category. Returns error if the dimensions
  // are not consecutive (but can be permuted).
  // If the dimensions are empty (i.e. the product of sizes is 1), then all
  // dimensions are removed if remove_if_empty; otherwise one dimension is kept
  // (if there was any).
  absl::Status Collapse(Category category, bool remove_if_empty);
  // Removes the dimensions in the range [start, end).
  absl::Status EraseDimensions(int64_t start, int64_t end);
  // Returns the shape of the operand.
  const Shape& shape() const { return shape_; }
  // Converts the shape dimension index to the category dimension index.
  absl::StatusOr<int64_t> LocalIndex(Category category,
                                     int64_t global_dim_idx) const;

 private:
  Shape shape_;
  std::array<std::vector<int64_t>, 3> dim_numbers_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MATMUL_INDEXING_UTILS_H_
