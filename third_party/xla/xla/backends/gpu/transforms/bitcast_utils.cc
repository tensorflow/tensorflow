/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/bitcast_utils.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/layout.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Copies the element type and size from `source` to `destination`.
void CopyElementType(const Shape& source, Shape* destination) {
  destination->set_element_type(source.element_type());
  destination->mutable_layout()->set_element_size_in_bits(
      source.layout().element_size_in_bits());
}

llvm::SmallVector<int64_t> GetInversePermutation(
    absl::Span<const int64_t> permutation) {
  llvm::SmallVector<int64_t> result(permutation.size());
  for (int64_t i = 0; i < permutation.size(); ++i) {
    result[permutation[i]] = i;
  }
  return result;
}

// Applies the backward-mapping 'permutation' to 'values'.
llvm::SmallVector<int64_t> ApplyPermutation(
    absl::Span<const int64_t> values, absl::Span<const int64_t> permutation) {
  llvm::SmallVector<int64_t> result;
  result.reserve(permutation.size());
  for (int64_t index : permutation) {
    result.push_back(values[index]);
  }
  return result;
}

// Returns the dimensions of 'shape' in minor-to-major order.
llvm::SmallVector<int64_t> GetPhysicalDimensions(const Shape& shape) {
  return ApplyPermutation(shape.dimensions(), shape.layout().minor_to_major());
}

// Returns parameters to rewrite a broadcast + bitcast as bitcast + broadcast.
//
// Example:
//
// broadcast = broadcast(operand)
// result = result_shape bitcast(broadcast)
//
// to
//
// bitcast = new_shape bitcast(operand)
// result = broadcast(bitcast), dimensions={new_dims}.
//
// Assumes that:
// - broadcast does not transpose dimensions (checked by hlo_verifier);
// - bitcast does not mix operand and broadcast dimensions (checks);
absl::StatusOr<BitcastParams> CalculateBitcastOfBroadcast(
    const HloBroadcastInstruction* broadcast, const Shape& result_shape) {
  const Shape& broadcast_shape = broadcast->shape();

  // Maps broadcast dimension index to whether it's an operand dimension.
  llvm::SmallVector<bool> is_operand_dim(broadcast_shape.dimensions().size());
  for (const int64_t index : broadcast->dimensions()) {
    is_operand_dim[index] = true;
  }

  // Dimensions of the new broadcast.
  llvm::SmallVector<int64_t> new_dims;
  llvm::SmallVector<int64_t> broadcast_physical_dims =
      GetPhysicalDimensions(broadcast_shape);
  auto factors = CommonFactors(GetPhysicalDimensions(result_shape),
                               broadcast_physical_dims);
  for (int64_t i = 1; i < factors.size(); ++i) {
    auto [result_from, broadcast_from] = factors[i - 1];
    auto [result_to, broadcast_to] = factors[i];

    bool all_operands = true, any_operands = false;
    for (int64_t j = broadcast_from; j < broadcast_to; ++j) {
      if (broadcast_physical_dims[j] == 1) {
        // If dimension size is 1 then we can ignore it: it's either immediately
        // dropped by old reshape or it's coming from the operand and then the
        // new reshape will handle it.
        continue;
      }
      bool value = is_operand_dim[broadcast_shape.layout().minor_to_major(j)];
      all_operands &= value;
      any_operands |= value;
    }
    if (!any_operands) {
      continue;  // All dimensions in this group are broadcast dimensions.
    }
    if (!all_operands) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot hoist bitcast across ", broadcast->ToString(),
                       " as it mixes operand and broadcast dimensions."));
    }

    for (int64_t j = result_from; j < result_to; ++j) {
      new_dims.push_back(result_shape.layout().minor_to_major(j));
    }
  }
  absl::c_sort(new_dims);  // Sort into logical order.

  BitcastParams result;
  CopyElementType(result_shape, &result.new_shape);
  for (int64_t index : new_dims) {
    result.new_shape.add_dimensions(result_shape.dimensions(index));
  }
  auto* new_layout =
      result.new_shape.mutable_layout()->mutable_minor_to_major();
  new_layout->reserve(new_dims.size());
  for (int64_t index : result_shape.layout().minor_to_major()) {
    if (auto it = absl::c_lower_bound(new_dims, index);
        it != new_dims.end() && *it == index) {
      new_layout->push_back(it - new_dims.begin());
    }
  }
  result.new_dims = std::move(new_dims);

  VLOG(3) << "CalculateBitcastOfBroadcast:";
  VLOG(3) << "  broadcast = " << broadcast_shape.ToString(true) << " broadcast("
          << broadcast->operand(0)->shape().ToString(true)
          << " operand), dimensions="
          << absl::StrJoin(broadcast->dimensions(), ",");
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " bitcast("
          << broadcast_shape.ToString(true) << " broadcast)";
  VLOG(3) << "--------------------------------";
  VLOG(3) << "  bitcast   = " << result.new_shape.ToString(true) << " bitcast("
          << broadcast->operand(0)->shape().ToString(true) << " operand)";
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " broadcast("
          << result.new_shape.ToString(true)
          << " bitcast), dimensions=" << absl::StrJoin(result.new_dims, ",");

  return result;
}

// Returns parameters to rewrite a bitcast + broadcast as broadcast + bitcast.
//
// Example:
//
// bitcast = bitcast(operand_shape operand)
// result = broadcast(bitcast)
//
// to
//
// broadcast = new_shape broadcast(operand), dimensions={new_dims}.
// result = bitcast(broadcast)
//
// Assumes that:
// - broadcast does not transpose dimensions (checked by hlo_verifier);
// - bitcast does not mix operand and broadcast dimensions (checks);
absl::StatusOr<BitcastParams> CalculateBroadcastOfBitcast(
    const HloBroadcastInstruction* broadcast, const Shape& operand_shape) {
  const Shape& bitcast_shape = broadcast->operand(0)->shape();
  const Shape& result_shape = broadcast->shape();

  // Maps logical result dimension index to a range of physical operand
  // dimensions, or nullopt if the dimension is broadcasted.
  llvm::SmallVector<std::optional<std::pair<int64_t, int64_t>>>
      result_to_operand_range(result_shape.dimensions().size());
  auto result_inv_layout =
      GetInversePermutation(result_shape.layout().minor_to_major());
  auto factors = CommonFactors(GetPhysicalDimensions(bitcast_shape),
                               GetPhysicalDimensions(operand_shape));
  for (int64_t i = 1; i < factors.size(); ++i) {
    auto [bitcast_from, operand_from] = factors[i - 1];
    auto [bitcast_to, operand_to] = factors[i];

    llvm::SmallVector<int64_t> indices;
    indices.reserve(bitcast_to - bitcast_from);
    for (int64_t j = bitcast_from; j < bitcast_to; ++j) {
      int64_t index =
          broadcast->dimensions()[bitcast_shape.layout().minor_to_major(j)];

      // Store the entire operand dimension range in the minor-most dimension
      // index and an empty range in all others.
      result_to_operand_range[index].emplace(operand_from, operand_to);
      operand_from = operand_to;

      // Check that the physical result indices form a contiguous range.
      indices.push_back(result_inv_layout[index]);
    };

    if (indices.back() - indices.front() >= bitcast_to - bitcast_from ||
        !absl::c_is_sorted(indices)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot hoist bitcast across ", broadcast->ToString(),
                       " because result dimensions are not contiguous."));
    }
  }

  BitcastParams result;
  CopyElementType(operand_shape, &result.new_shape);
  result.new_dims.resize(operand_shape.dimensions().size());
  auto* new_layout =
      result.new_shape.mutable_layout()->mutable_minor_to_major();
  int64_t new_rank = operand_shape.dimensions().size() +
                     result_shape.dimensions().size() -
                     bitcast_shape.dimensions().size();
  new_layout->reserve(new_rank);
  llvm::SmallVector<int64_t> new_shape_dims(new_rank);

  // We are free to insert the broadcast dimensions in any order. Insert them
  // at the end of the the logical dimension order.
  int64_t broadcast_index = operand_shape.dimensions().size();

  // Iterate through the logical result dimension indices in physical order.
  for (int64_t result_index : result_shape.layout().minor_to_major()) {
    if (auto range = result_to_operand_range[result_index]) {
      // This result dimension corresponds to a group of operand dimensions.
      // Iterate through the range of physical operand dimension indices.
      for (int64_t i = range->first; i < range->second; ++i) {
        int64_t operand_index = operand_shape.layout().minor_to_major(i);
        int64_t new_index = operand_index;
        new_shape_dims[new_index] = operand_shape.dimensions(operand_index);
        new_layout->push_back(new_index);
        result.new_dims[operand_index] = new_index;
      }
    } else {
      // This is a new dimension introduced by the original broadcast.
      int64_t new_index = broadcast_index++;
      new_shape_dims[new_index] = result_shape.dimensions(result_index);
      new_layout->push_back(new_index);
    }
  }
  absl::c_sort(result.new_dims);  // Sort into logical order.
  for (int64_t dimension : new_shape_dims) {
    result.new_shape.add_dimensions(dimension);
  }

  VLOG(3) << "CalculateBroadcastOfBitcast:";
  VLOG(3) << "  bitcast   = " << bitcast_shape.ToString(true) << " bitcast("
          << operand_shape.ToString(true) << " operand)";
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " broadcast("
          << bitcast_shape.ToString(true) << " bitcast), dimensions="
          << absl::StrJoin(broadcast->dimensions(), ",");
  VLOG(3) << "--------------------------------";
  VLOG(3) << "  broadcast = " << result.new_shape.ToString(true)
          << " broadcast(" << operand_shape.ToString(true)
          << " operand), dimensions=" << absl::StrJoin(result.new_dims, ",");
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " bitcast("
          << result.new_shape.ToString(true) << " broadcast)";

  return result;
}

// Implements CalculateBitcastOfTranspose(), except that result.new_dims is
// the inverse permutation, mapping the input dimensions to the output
// dimensions.
absl::StatusOr<BitcastParams> CalculateBitcastOfTransposeImpl(
    const HloTransposeInstruction* transpose, const Shape& result_shape,
    const Shape& transpose_shape, const Shape& operand_shape,
    absl::Span<const int64_t> transpose_dims) {
  if (transpose->shape().layout() != transpose->operand(0)->shape().layout()) {
    return absl::InternalError(
        absl::StrCat("Expected input and output layouts to be the same for ",
                     transpose->ToString()));
  }

  // Maps physical operand dimension index to a range of physical result
  // dimensions.
  llvm::SmallVector<std::pair<int64_t, int64_t>> operand_to_result_range(
      operand_shape.dimensions().size());
  // Maps logical operand dimension index to the physical dimension index.
  llvm::SmallVector<int64_t> operand_inv_layout =
      GetInversePermutation(operand_shape.layout().minor_to_major());

  const absl::InlinedVector<std::pair<int64_t, int64_t>, 8> factors =
      ::xla::gpu::detail::CommonFactorsMergingTrivialRanges(
          GetPhysicalDimensions(result_shape),
          GetPhysicalDimensions(transpose_shape));
  for (int64_t i = 1; i < factors.size(); ++i) {
    auto [result_from, transpose_from] = factors[i - 1];
    auto [result_to, transpose_to] = factors[i];

    llvm::SmallVector<int64_t> indices;
    indices.reserve(transpose_to - transpose_from);
    for (int64_t j = transpose_from; j < transpose_to; ++j) {
      int64_t dim_index = transpose_shape.layout().minor_to_major(j);
      int64_t index = operand_inv_layout[transpose_dims[dim_index]];

      if (transpose_shape.dimensions(dim_index) == 1) {
        // Size-1 dimensions do not affect the physical layout, so we can ignore
        // them for the purpose of checking contiguity. We mark them with an
        // empty range in the operand_to_result_range map, so that they are
        // dropped from the new bitcast/transpose shape.
        operand_to_result_range[index] = {result_from, result_from};
        continue;
      }

      // Store the entire result dimension range in the minor-most dimension
      // index and an empty range in all others.
      operand_to_result_range[index] = {result_from, result_to};
      result_from = result_to;

      // Check that the physical operand indices form a contiguous range.
      indices.push_back(index);
    };

    if (indices.empty()) {
      // If all dimensions are size 1, we can just drop them.
      continue;
    }
    if (indices.back() - indices.front() >= indices.size() ||
        !absl::c_is_sorted(indices)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot hoist bitcast across ", transpose->ToString(),
                       " because result dimensions are not contiguous."));
    }
  }

  BitcastParams result;
  CopyElementType(result_shape, &result.new_shape);
  // Just like the old transpose, the new transpose does not change the
  // layout.
  *result.new_shape.mutable_layout() = result_shape.layout();
  result.new_dims.resize(result_shape.dimensions().size());
  llvm::SmallVector<int64_t> new_shape_dims(result_shape.dimensions().size());
  // Iterate through the physical operand and new_shape dimension indices.
  for (int64_t i = 0, j = 0; i < operand_shape.dimensions().size(); ++i) {
    auto range = operand_to_result_range[i];
    // Iterate through corresponding range of physical result dimension
    // indices.
    for (int64_t k = range.first; k < range.second; ++k) {
      int64_t new_index = result_shape.layout().minor_to_major(j++);
      int64_t result_index = result_shape.layout().minor_to_major(k);
      new_shape_dims[new_index] = result_shape.dimensions(result_index);
      result.new_dims[new_index] = result_index;
    }
  }
  for (int64_t dimension : new_shape_dims) {
    result.new_shape.add_dimensions(dimension);
  }

  VLOG(3) << "CalculateBitcastOfTransposeImpl:";
  VLOG(3) << "  transpose = " << transpose_shape.ToString(true) << " transpose("
          << operand_shape.ToString(true)
          << " operand), dimensions=" << absl::StrJoin(transpose_dims, ",");
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " bitcast("
          << transpose_shape.ToString(true) << " transpose)";
  VLOG(3) << "--------------------------------";
  VLOG(3) << "  bitcast   = " << result.new_shape.ToString(true) << " bitcast("
          << operand_shape.ToString(true) << " operand)";
  VLOG(3) << "  result    = " << result_shape.ToString(true) << " transpose("
          << result.new_shape.ToString(true) << " bitcast), dimensions="
          << absl::StrJoin(GetInversePermutation(result.new_dims), ",");

  return result;
}

// Returns parameters to rewrite a transpose + bitcast as bitcast + transpose.
//
// Example:
//
// transpose = transpose(operand)
// result = result_shape bitcast(transpose)
//
// to
//
// bitcast = new_shape bitcast(operand)
// result = transpose(bitcast), dimensions={new_dims}.
//
// Assumes that:
// - bitcast only mixes contiguous dimensions (checks);
// - transpose does not change layout (checks);
absl::StatusOr<BitcastParams> CalculateBitcastOfTranspose(
    const HloTransposeInstruction* transpose, const Shape& result_shape) {
  ASSIGN_OR_RETURN(
      BitcastParams result,
      CalculateBitcastOfTransposeImpl(
          transpose, result_shape, transpose->shape(),
          transpose->operand(0)->shape(), transpose->dimensions()));
  result.new_dims = GetInversePermutation(result.new_dims);
  return result;
}

// Returns parameters to rewrite a bitcast + transpose as transpose + bitcast.
//
// Example:
//
// bitcast = bitcast(operand_shape operand)
// result = transpose(bitcast)
//
// to
//
// transpose = new_shape transpose(operand), dimensions={new_dims}.
// result = bitcast(transpose)
//
// Assumes that:
// - bitcast only mixes contiguous dimensions (checks);
// - transpose does not change layout (checks);
absl::StatusOr<BitcastParams> CalculateTransposeOfBitcast(
    const HloTransposeInstruction* transpose, const Shape& operand_shape) {
  return CalculateBitcastOfTransposeImpl(
      transpose, operand_shape, transpose->operand(0)->shape(),
      transpose->shape(), GetInversePermutation(transpose->dimensions()));
}

namespace detail {

absl::InlinedVector<std::pair<int64_t, int64_t>, 8>
CommonFactorsMergingTrivialRanges(absl::Span<const int64_t> a,
                                  absl::Span<const int64_t> b) {
  // CommonFactors does what we need but it also creates empty groups with
  // product of 1, e.g. `[1] -> []` or `[] -> [1]`. We remove the bounds of
  // such ranges to merge them with neighbors. There are many different ways
  // to do this, here we continously append ranges to the start of the next
  // group unless it is the very last range.
  absl::InlinedVector<std::pair<int64_t, int64_t>, 8> bounds =
      CommonFactors(a, b);
  for (size_t i = 0; i + 1 < bounds.size() && bounds.size() > 2;) {
    auto [a_start, b_start] = bounds[i];
    auto [a_end, b_end] = bounds[i + 1];
    if (a_start != a_end && b_start != b_end) {
      i++;
      continue;
    }
    if (i + 2 == bounds.size()) {
      // Very last range - append it to the previous one.
      bounds.erase(bounds.begin() + i);
    } else {
      bounds.erase(bounds.begin() + i + 1);
    }
  }
  return bounds;
}

}  // namespace detail
}  // namespace xla::gpu
