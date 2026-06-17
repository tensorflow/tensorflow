/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_UTILS_H_

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {

using llvm::ArrayRef;
using mlir::Operation;
using mlir::ShapedType;
using mlir::Value;

// Returns true if the value is the min float value.
inline bool IsNegInfiniteValue(APFloat value) {
  if (!value.isNegative()) return false;
  return value.isInfinity();
}

// Returns true if the value is the max float value.
inline bool IsPosInfiniteValue(APFloat value) {
  if (value.isNegative()) return false;
  return value.isInfinity();
}

// Returns 1D 32-bit dense elements attribute with the given values.
inline DenseIntElementsAttr GetI32ElementsAttr(ArrayRef<int32_t> values,
                                               Builder* builder) {
  RankedTensorType ty = mlir::RankedTensorType::get(
      {static_cast<int32_t>(values.size())}, builder->getIntegerType(32));
  return DenseIntElementsAttr::get(ty, values);
}

inline DenseIntElementsAttr GetI64ElementsAttr(ArrayRef<int64_t> values,
                                               Builder* builder) {
  RankedTensorType ty = RankedTensorType::get(
      {static_cast<int64_t>(values.size())}, builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

// Returns true if all tensor value in `values` has static shape and same shape.
inline bool OpHasSameStaticShapes(Operation* op) {
  auto values = op->getOperands();
  int operand_num = 0;
  ArrayRef<int64_t> shape;
  for (Value value : values) {
    auto shaped_type = mlir::dyn_cast<ShapedType>(value.getType());
    if (!shaped_type || !shaped_type.hasStaticShape()) {
      return false;
    }
    if (operand_num == 0) {
      shape = shaped_type.getShape();
    } else {
      if (shape != shaped_type.getShape()) {
        return false;
      }
    }
    ++operand_num;
  }
  return true;
}

// Utility function to map final permutation to initial permutation
// initial -> permutation1 -> permutation2 -> final
inline DenseElementsAttr RemapPermutation(Value permutation1,
                                          DenseElementsAttr perm2_const) {
  SmallVector<int32_t> initial_permutation;
  DenseElementsAttr perm1_const;

  SmallVector<int32_t> new_permutation;
  if (matchPattern(permutation1, m_Constant(&perm1_const))) {
    for (int32_t idx = 0; idx < perm1_const.getNumElements(); ++idx) {
      initial_permutation.push_back(idx);
    }
    for (auto perm : perm2_const.getValues<APInt>()) {
      new_permutation.push_back(
          initial_permutation[perm1_const
                                  .getValues<APInt>()[perm.getSExtValue()]
                                  .getSExtValue()]);
    }
  }

  return mlir::DenseElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int>(new_permutation.size())},
          mlir::IntegerType::get(permutation1.getContext(), 32)),
      llvm::ArrayRef(new_permutation));
}

// Utility function to map final permutation to initial permutation
// initial -> permutation1 -> permutation2 -> final
inline DenseElementsAttr RemapPermutation(Value permutation1,
                                          Value permutation2) {
  DenseElementsAttr perm2_const;
  (void)matchPattern(permutation2, m_Constant(&perm2_const));

  return RemapPermutation(permutation1, perm2_const);
}

inline bool IsTransposeNoop(Value permutation) {
  DenseElementsAttr perm_values_attr;
  if (!matchPattern(permutation, m_Constant(&perm_values_attr))) return false;

  for (const auto& [idx, perm_value] :
       llvm::enumerate(perm_values_attr.getValues<APInt>())) {
    if (perm_value.getSExtValue() != idx) {
      return false;
    }
  }
  return true;
}

// Returns true if the transpose op is trivial. Trivial means that
// the permutation is a cyclic permutation of the original shape with only the
// identity dimensions permuted.
inline bool IsTransposeTrivial(llvm::ArrayRef<int64_t> input_shape,
                               Value perm) {
  DenseElementsAttr perm_values_attr;
  if (!matchPattern(perm, m_Constant(&perm_values_attr))) return false;

  SmallVector<int64_t, 8> perm_values;
  for (const auto& dim : perm_values_attr.getValues<APInt>()) {
    // Valid range is [-input_shape.size(), input_shape.size()).
    int64_t p = dim.getSExtValue();
    if (p < 0) {
      p += input_shape.size();
    }
    if (p < 0 || p >= input_shape.size()) {
      return false;
    }
    perm_values.push_back(p);
  }

  // This should never happen unless the input graph is malformed.
  if (input_shape.size() != perm_values.size()) {
    return false;
  }

  SmallVector<int, 8> old_major_index_ordering;
  SmallVector<int, 8> new_major_index_ordering;
  for (int i = 0, end = input_shape.size(); i < end; i++) {
    if (input_shape[i] != 1) {
      old_major_index_ordering.push_back(i);
    }

    if (input_shape[perm_values[i]] != 1) {
      new_major_index_ordering.push_back(perm_values[i]);
    }
  }
  return (old_major_index_ordering == new_major_index_ordering);
}

// Returns the permutation that maps the input shape to the output shape.
// This is only valid for trivial reshape ops.
inline DenseElementsAttr GetPermutationFromTrivialReshape(
    mlir::ShapedType input_type, mlir::ShapedType output_type) {
  ArrayRef<int64_t> in_shape = input_type.getShape();
  ArrayRef<int64_t> out_shape = output_type.getShape();

  // Get the indexes of the non-identity dimensions and the identity dimensions
  // in the input shape.
  SmallVector<int32_t> input_nonidentity_dims_index_array;
  SmallVector<int32_t> input_identity_dims_index_array;

  // Since the reshape is trivial, the input and output shapes should have the
  // same number of dimensions. And the non-identity dimensions must be in the
  // same cyclic order.
  for (size_t idx = 0; idx < in_shape.size(); ++idx) {
    if (in_shape[idx] != 1) {
      input_nonidentity_dims_index_array.push_back(idx);
    } else {
      input_identity_dims_index_array.push_back(idx);
    }
  }

  // Get the permutation that maps the input shape to the output shape.
  SmallVector<int32_t> permutation;
  size_t nonidentity_dims_index_poiter = 0;
  size_t identity_dims_index_pointer = 0;
  for (auto out_dim : out_shape) {
    if (out_dim != 1) {
      permutation.push_back(
          input_nonidentity_dims_index_array[nonidentity_dims_index_poiter++]);
    } else {
      permutation.push_back(
          input_identity_dims_index_array[identity_dims_index_pointer++]);
    }
  }

  return mlir::DenseElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int>(permutation.size())},
          mlir::IntegerType::get(input_type.getContext(), 32)),
      llvm::ArrayRef(permutation));
}

// Returns true if the reshape op is equivalent to a transpose op.
// This is true if the reshape op is a trivial reshape op, meaning no change in
// the order of non-identity dimensions.
inline bool IsReshapeEquivalentToTranspose(mlir::ShapedType input_type,
                                           mlir::ShapedType output_type) {
  std::vector<int64_t> in_shape{input_type.getShape().vec()};
  std::vector<int64_t> out_shape{output_type.getShape().vec()};

  // If the reshape changes the number of dimensions so it cannot be interpreted
  // as a transpose.
  if (in_shape.size() != out_shape.size()) {
    return false;
  }

  in_shape.erase(std::remove(in_shape.begin(), in_shape.end(), 1),
                 in_shape.end());
  out_shape.erase(std::remove(out_shape.begin(), out_shape.end(), 1),
                  out_shape.end());
  return in_shape == out_shape;
}

// Checks if all elements in the constant attribute value are 1.
inline bool IsAllOnesConstant(Attribute value) {
  auto values = mlir::cast<DenseElementsAttr>(value).getValues<int32_t>();
  return !std::any_of(values.begin(), values.end(),
                      [](int32_t element_value) { return element_value != 1; });
}

// Checks if all elements in the constant attribute value are non-negative.
inline bool HasNonNegativeValues(Attribute value) {
  auto values = mlir::cast<DenseElementsAttr>(value).getValues<APInt>();
  return !std::any_of(
      values.begin(), values.end(),
      [](const APInt& element_value) { return element_value.isNegative(); });
}

// Utility function to get the offset between two dense attribute values.
inline TypedAttr GetOffSet(Attribute begin, Attribute end) {
  auto begin_values = mlir::cast<DenseElementsAttr>(begin).getValues<int32_t>();
  auto end_values = mlir::cast<DenseElementsAttr>(end).getValues<int32_t>();

  SmallVector<int32_t> offsets;
  if (begin_values.size() == end_values.size()) {
    for (size_t i = 0; i < begin_values.size(); ++i) {
      offsets.push_back(end_values[i] - begin_values[i]);
    }
  }

  return mlir::DenseElementsAttr::get(
      RankedTensorType::get({static_cast<int>(offsets.size())},
                            mlir::IntegerType::get(begin.getContext(), 32)),
      llvm::ArrayRef(offsets));
}

// Check if the offset between two dense attribute values is non-negative.
inline bool HasNonNegativeOffset(Attribute begin, Attribute end) {
  return HasNonNegativeValues(GetOffSet(begin, end));
}

// Return true if the permutation value only swaps the last two dimensions
inline bool AreLastTwoDimsTransposed(Value permutation) {
  if (!permutation) return false;
  DenseElementsAttr perm_values_attr;

  if (!matchPattern(permutation, m_Constant(&perm_values_attr))) return false;
  auto perm_values = perm_values_attr.getValues<APInt>();
  size_t idx = 0;
  for (; idx < perm_values_attr.size() - 2; ++idx) {
    if (perm_values[idx].getSExtValue() != idx) return false;
  }

  return (perm_values[idx].getSExtValue() == perm_values_attr.size() - 1) &&
         (perm_values[idx + 1].getSExtValue() == idx);
}

// Gets the new type after transposing the last 2 dimensions.
inline Type TransposeLastTwoDims(Type type) {
  auto shaped_type = mlir::dyn_cast<ShapedType>(type);
  if (!shaped_type.hasStaticShape() || shaped_type.getRank() < 2) {
    return nullptr;
  }
  int rank = shaped_type.getRank();
  if (rank < 2) {
    return nullptr;
  }
  SmallVector<int64_t> new_shape(shaped_type.getShape().begin(),
                                 shaped_type.getShape().end());
  std::swap(new_shape[rank - 1], new_shape[rank - 2]);
  return shaped_type.clone(new_shape);
}

// Returns a ShapedType for a permutation and the shape of input after
// applying the permutation to the given shape through a transpose.
inline mlir::ShapedType GetTransposedType(
    Value input, llvm::ArrayRef<int64_t> permutation_array) {
  auto input_type = mlir::cast<ShapedType>(input.getType());
  if (permutation_array.size() != input_type.getRank()) {
    return nullptr;
  }
  llvm::SmallVector<int64_t> transposed_shape(permutation_array.size());
  for (int64_t i = 0; i < permutation_array.size(); ++i) {
    transposed_shape[i] = input_type.getDimSize(permutation_array[i]);
  }
  auto transposed_type =
      RankedTensorType::get(transposed_shape, input_type.getElementType());
  return transposed_type;
}

// Return the resultant shape if the shape of the supplied attribute/value is
// expanded by n leading 1s'.
inline SmallVector<int32_t> GetExpandedShape(Value input_val, int n) {
  auto input_shape = mlir::cast<ShapedType>(input_val.getType()).getShape();
  SmallVector<int32_t> expanded_shape;
  expanded_shape.reserve(input_shape.size() + n);
  for (int i = 0; i < n; ++i) {
    expanded_shape.push_back(1);
  }
  expanded_shape.insert(expanded_shape.end(), input_shape.begin(),
                        input_shape.end());
  return expanded_shape;
}

// Return the resultant shape as a DenseElementsAttr if the shape of the
// supplied attribute/value is expanded by n leading 1s'.
inline DenseElementsAttr GetExpandedShapeAttr(Value input_val, int n) {
  auto expanded_shape = GetExpandedShape(input_val, n);

  return mlir::DenseElementsAttr::get(
      RankedTensorType::get({static_cast<int>(expanded_shape.size())},
                            mlir::IntegerType::get(input_val.getContext(), 32)),
      llvm::ArrayRef(expanded_shape));
}

// Return the resultant shape type if the shape of the supplied attribute/value
// is expanded by n leading 1s'.
inline mlir::ShapedType GetExpandedShapeType(Value input_val, int n) {
  auto expanded_shape = GetExpandedShape(input_val, n);
  return RankedTensorType::get(
      SmallVector<int64_t>{expanded_shape.begin(), expanded_shape.end()},
      mlir::cast<ShapedType>(input_val.getType()).getElementType());
}

// Returns shape of a ranked tensor as a SmallVector.
// Precondition: input_value's is ranked tensor.
// Returns a squeezed shape when `squeeze_leading_ones` is set to true.
inline SmallVector<int32_t> GetShape(Value input_value,
                                     bool squeeze_leading_ones = false) {
  auto output_shape =
      mlir::dyn_cast<ShapedType>(input_value.getType()).getShape();

  SmallVector<int32_t> shape;
  shape.reserve(output_shape.size());

  bool can_squeeze = true;
  for (size_t dim_idx = 0; dim_idx < output_shape.size(); ++dim_idx) {
    int64_t dim = output_shape[dim_idx];
    if (squeeze_leading_ones && can_squeeze && dim == 1) {
      continue;
    } else if (can_squeeze && dim != 1) {
      can_squeeze = false;
    }
    shape.push_back(ShapedType::isDynamic(dim) ? -1
                                               : static_cast<int32_t>(dim));
  }
  return shape;
}

// Returns shape of a ranked tensor as a DenseElementsAttr.
// Precondition: input_value's is ranked tensor.
// Returns a squeezed shape when `squeeze_leading_ones` is set to true.
inline DenseElementsAttr GetShapeAttr(Value input_value,
                                      bool squeeze_leading_ones = false) {
  SmallVector<int32_t> shape = GetShape(input_value, squeeze_leading_ones);

  return mlir::DenseElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int>(shape.size())},
          mlir::IntegerType::get(input_value.getContext(), 32)),
      llvm::ArrayRef(shape));
}

// Returns the value of a constant attribute as an int array, if the value is
// not a constant, returns an error status.
inline absl::StatusOr<SmallVector<int32_t>> GetValueAsIntArray(Value value) {
  DenseElementsAttr values_const_attr;
  if (!matchPattern(value, m_Constant(&values_const_attr))) {
    return absl::InvalidArgumentError("Value is not a constant.");
  }

  SmallVector<int32_t> values;
  for (const auto& value : values_const_attr.getValues<APInt>()) {
    values.push_back(value.getSExtValue());
  }
  return values;
}

////////////////////////////////////////////////////////////////////////////////
///////////////// OP BROADCASTING UTILITIES ////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

// Returns whether the resultant type of any broadcastable operation with
// operands `a` and `b` matches `expected_output`. Returns false if `a` is not
// broadcast-compatible with `b`.
inline bool OperandsBroadcastToOutputType(Type a, Type b,
                                          Type expected_output) {
  Type output_element_type =
      mlir::cast<ShapedType>(expected_output).getElementType();
  Type broadcasted_type =
      OpTrait::util::getBroadcastedType(a, b, output_element_type);
  return broadcasted_type != Type() && broadcasted_type == expected_output;
}

// Returns int, float or complex DenseElementsAttr with scalar shape with the
// given element type and the integer value.
template <typename T>
DenseElementsAttr GetScalarOfType(Type ty, T raw_value) {
  RankedTensorType scalar_ty = RankedTensorType::get({}, ty);
  if (auto float_ty = mlir::dyn_cast<FloatType>(ty)) {
    FloatAttr attr = FloatAttr::get(float_ty, raw_value);
    return DenseElementsAttr::get(scalar_ty, attr);
  } else if (auto int_ty = mlir::dyn_cast<IntegerType>(ty)) {
    IntegerAttr attr = IntegerAttr::get(int_ty, raw_value);
    return DenseElementsAttr::get(scalar_ty, attr);
  } else if (auto complex_ty = mlir::dyn_cast<ComplexType>(ty)) {
    Type complex_element_ty = complex_ty.getElementType();
    if (complex_element_ty.isF32()) {
      return DenseElementsAttr::get(
          scalar_ty, static_cast<std::complex<float>>(raw_value));
    } else if (complex_element_ty.isF64()) {
      return DenseElementsAttr::get(
          scalar_ty, static_cast<std::complex<double>>(raw_value));
    }
  }
  llvm_unreachable("unsupported type");
}

// Checks if reduction axes and broadcast axes are disjoint.
// Broadcast axes are derived by comparing the shape of `input_val` to the shape
// represented by `target_shape_attr` according to standard broadcasting rules.
// Returns true if the sets of axes are disjoint, false otherwise or on error.
inline bool AreBroadcastAndReductionAxesIndependent(
    mlir::Value input_val, const mlir::Attribute& indices_attr,
    const mlir::Attribute& target_shape_attr) {
  // 1. Get input type and shape.
  // Use llvm::dyn_cast for safer casting.
  auto ranked_input_type =
      llvm::dyn_cast<mlir::RankedTensorType>(input_val.getType());
  if (!ranked_input_type) {
    // Consider logging or error emission if builder context is
    // available/needed.
    return false;  // Expect ranked type.
  }
  llvm::ArrayRef<int64_t> input_shape = ranked_input_type.getShape();
  const int64_t input_rank = ranked_input_type.getRank();

  // 2. Validate and extract reduction axes.
  // Use llvm::dyn_cast for safer casting.
  auto indices = llvm::dyn_cast<mlir::DenseElementsAttr>(indices_attr);
  if (!indices || !indices.getElementType().isIntOrIndex()) {
    return false;  // Invalid indices attribute.
  }

  // Use std::set for efficient storage and lookup of axes.
  std::set<int64_t> reduction_axes_set;
  if (!indices.empty()) {  // Only process if there are reduction axes.
    if (input_rank == 0) {
      // It's invalid to specify reduction axes for a scalar (rank 0) input.
      return false;
    }

    // Iterate using range-based for loop and structured binding (if applicable)
    // or direct value access.
    for (const mlir::APInt& axis_val : indices.getValues<mlir::APInt>()) {
      int64_t axis =
          axis_val.getSExtValue();  // Use sign extension for neg axes.

      // Normalize axis and check bounds.
      if (axis < -input_rank || axis >= input_rank) {
        return false;  // Axis out of bounds.
      }
      if (axis < 0) {
        axis += input_rank;  // Convert negative axis to positive.
      }
      reduction_axes_set.insert(axis);
    }
  }

  // If there are no reduction axes, they are trivially independent of any
  // broadcast axes.
  if (reduction_axes_set.empty()) {
    return true;
  }

  // 3. Validate and extract target shape for broadcast.
  // Use llvm::dyn_cast for safer casting.
  auto target_shape_value_attr =
      llvm::dyn_cast<mlir::DenseElementsAttr>(target_shape_attr);
  if (!target_shape_value_attr ||
      !target_shape_value_attr.getElementType().isIntOrIndex()) {
    return false;  // Invalid target shape attribute.
  }

  // Use llvm::SmallVector for efficient shape storage.
  llvm::SmallVector<int64_t, 4> target_shape_vec;
  target_shape_vec.reserve(
      target_shape_value_attr.getNumElements());  // Pre-allocate
  for (const mlir::APInt& shape_val :
       target_shape_value_attr.getValues<mlir::APInt>()) {
    // Assuming shape dimensions should be non-negative, consider getZExtValue.
    // However, getSExtValue is safe if intermediate calculations handle signs.
    target_shape_vec.push_back(shape_val.getSExtValue());
  }
  // Use llvm::ArrayRef for safe, non-owning view of the shape vector.
  llvm::ArrayRef<int64_t> target_shape = target_shape_vec;
  const int64_t target_rank = target_shape.size();

  // 4. Determine broadcast axes based on standard broadcasting rules.
  std::set<int64_t> broadcast_axes_set;
  const int64_t max_rank = std::max(input_rank, target_rank);

  // Iterate through dimensions, aligning from the right (trailing dimensions).
  for (int64_t i = 0; i < max_rank; ++i) {
    // Calculate indices relative to the end of the shape arrays.
    const int64_t input_dim_idx = input_rank - 1 - i;
    const int64_t target_dim_idx = target_rank - 1 - i;

    // Treat dimensions missing due to lower rank as having size 1.
    const int64_t input_dim =
        (input_dim_idx >= 0) ? input_shape[input_dim_idx] : 1;
    const int64_t target_dim =
        (target_dim_idx >= 0) ? target_shape[target_dim_idx] : 1;

    // Check for incompatible shapes (dimensions differ and neither is 1).
    // This indicates an invalid broadcast according to NumPy rules.
    if (input_dim != target_dim && input_dim != 1 && target_dim != 1) {
      // Consider if the specific broadcast op allows other behaviors (e.g.,
      // -1). For standard rules, this is an incompatibility.
      return false;
    }

    // An axis in the *input* tensor is involved in broadcasting if its size is
    // 1 and the corresponding target dimension size is greater than 1.
    if (input_dim == 1 && target_dim > 1) {
      // Ensure the axis index is valid for the input tensor's rank.
      if (input_dim_idx >= 0) {
        broadcast_axes_set.insert(input_dim_idx);
      }
      // Note: If input_dim_idx < 0, broadcasting occurs due to rank difference,
      // but it doesn't correspond to an axis *within* the original input
      // tensor.
    }
  }

  // 5. Check for intersection between the set of reduction axes and the set of
  //    broadcast axes derived above.
  for (int64_t reduction_axis : reduction_axes_set) {
    if (broadcast_axes_set.count(reduction_axis)) {
      // Found an axis that is present in both sets.
      return false;
    }
  }

  // 6. No overlapping axes were found.
  return true;
}

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_UTILS_H_
