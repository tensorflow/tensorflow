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

#ifndef XLA_SERVICE_MATMUL_INDEXING_UTILS_H_
#define XLA_SERVICE_MATMUL_INDEXING_UTILS_H_

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/shape_tracker.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {

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
  enum Category { kBatch, kNonContracting, kContracting };

  DotOperandDims() = default;
  DotOperandDims(Shape shape, absl::Span<const int64_t> batch_dims,
                 absl::Span<const int64_t> non_contracting_dims,
                 absl::Span<const int64_t> contracting_dims);

  // Creates a DotOperandDims from a dot instruction.
  static absl::StatusOr<std::array<DotOperandDims, 2>> FromDot(
      const HloInstruction* dot);

  static absl::StatusOr<std::array<DotOperandDims, 4>> FromScaledDot(
      const HloInstruction* scaled_dot);

  // --- Factories ---

  // Creates a DotOperandDims from a dot instruction and operand index (0 or 1).
  static absl::StatusOr<DotOperandDims> FromDotOperand(
      const HloInstruction* dot, int operand_number);

  // Converts two DotOperandDims to a DotDimensionNumbers.
  static absl::StatusOr<DotDimensionNumbers> CreateDotDimensionNumbers(
      const DotOperandDims& lhs_dims, const DotOperandDims& rhs_dims);

  // Converts a span of two DotOperandDims to a DotDimensionNumbers.
  static absl::StatusOr<DotDimensionNumbers> CreateDotDimensionNumbers(
      absl::Span<const DotOperandDims> operands_dims);

  // Computes the output shape of the dot instruction.
  static absl::StatusOr<Shape> ComputeOutputShape(
      PrimitiveType element_type, const DotOperandDims& lhs_dims,
      const DotOperandDims& rhs_dims);

  // --- Category scoped functions ---

  // Returns the indices of the dimensions of the category.
  absl::Span<const int64_t> Indices(Category category) const {
    return dim_numbers_[category];
  }

  // Returns the category size (number of dimensions).
  int64_t Rank(Category category) const {
    return dim_numbers_[category].size();
  }

  // Returns the dimension sizes of the category.
  std::vector<int64_t> Sizes(Category category) const;

  // Returns the total size (product of dimensions) of the category.
  int64_t TotalSize(Category category) const;

  // Collapses the dimensions of the category. Returns error if the dimensions
  // are not sorted and consecutive.
  // If the dimensions are empty (i.e. the product of sizes is 1), then all
  // dimensions are removed if remove_if_empty; otherwise one dimension is kept
  // (if there was any).
  absl::Status CollapseCategory(Category category, bool remove_if_empty);

  // Returns true if the dimensions of the given category are sorted and
  // consecutive.
  bool IsConsecutive(Category category) const;

  // --- Global dimension functions ---

  // Permute the dimensions of the category.
  // The permutation is in the same format as you'd pass to the transpose
  // instruction. The corresponding dimension numbers are updated.
  void ApplyPermutation(absl::Span<const int64_t> permutation);

  // Functional version of ApplyPermutation. Returns a new DotOperandDims.
  DotOperandDims GetPermuted(absl::Span<const int64_t> permutation) const;

  // Converts the shape dimension index to the category dimension index.
  absl::StatusOr<int64_t> IndexWithinCategory(Category category,
                                              int64_t global_dim_idx) const;

  // Removes all degenerate (size=1) dimensions.
  absl::Status RemoveDegenerateDimensions();

  // Merges consecutive dimensions of the same category.
  absl::Status MergeAdjacentDimensions();

  // Removes the dimensions in the range [start, end).
  absl::Status EraseDimensions(int64_t start, int64_t end);

  // Inserts a dimension at the given index. The dimension is assigned the given
  // category. Within the category, the dimension is inserted before the first
  // dimension with index >= dim_idx (to keep sorted order).
  absl::StatusOr<int64_t> InsertDimension(
      Category category, int64_t dim_idx, int64_t dim_size,
      std::optional<int64_t> insert_at_idx = std::nullopt);

  // Returns the shape of the operand.
  const Shape& shape() const { return shape_; }

  // Overwrites the operand's shape with `new_shape` without re-indexing
  // dimension categories. The rank of `new_shape` must match the current shape.
  // This is typically used to update layout, element type, or bounds of
  // existing dimensions without logical restructuring.
  absl::Status SetShape(const Shape& new_shape);

  // Maps the current DotOperandDims to `target_shape` by tracking dimension
  // categories. We match dimensions between the two shapes using common prime
  // factors of the dimension sizes. Returns std::nullopt if the reshape would
  // require merging dimensions of different categories.
  absl::StatusOr<std::optional<DotOperandDims>> Reshape(
      const Shape& target_shape) const;

  // Maps the current DotOperandDims (assumed to be the output of the
  // instruction) to the operand of the instruction. Supports kTranspose and
  // kReshape. Returns std::nullopt if category mixing occurs during reshape.
  absl::StatusOr<std::optional<DotOperandDims>> MapBackward(
      const HloInstruction* inst) const;

  // Maps the current DotOperandDims (assumed to be the input of the
  // instruction) to the output of the instruction. Supports kTranspose and
  // kReshape. Returns std::nullopt if category mixing occurs during reshape.
  absl::StatusOr<std::optional<DotOperandDims>> MapForward(
      const HloInstruction* inst) const;

  // Creates a ShapeTracker to convert the shape of the current DotOperandDims
  // to the shape of another DotOperandDims, assuming they have the same total
  // number of elements in each category.
  absl::StatusOr<ShapeTracker> CreateShapeTrackerTo(
      const DotOperandDims& dst) const;

  // Applies the same transformations (for batch and contracting dims) as
  // observed between src_before and src_after to the current DotOperandDims.
  // Returns the transformed DotOperandDims.
  absl::StatusOr<DotOperandDims> ApplyTransformationsFrom(
      const DotOperandDims& src_before, const DotOperandDims& src_after) const;

  // Returns a debug string representation of the DotOperandDims.
  std::string ToString() const;

 private:
  Shape shape_;
  std::array<std::vector<int64_t>, 3> dim_numbers_;
};

}  // namespace xla

#endif  // XLA_SERVICE_MATMUL_INDEXING_UTILS_H_
