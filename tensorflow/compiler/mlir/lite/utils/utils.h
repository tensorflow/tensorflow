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
#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
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

// Returns true if all tensor value in `values` has static shape and same shape.
inline bool OpHasSameStaticShapes(Operation* op) {
  auto values = op->getOperands();
  int operand_num = 0;
  ArrayRef<int64_t> shape;
  for (Value value : values) {
    auto shaped_type = value.getType().dyn_cast<ShapedType>();
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
  for (const auto& dim : perm_values_attr.getValues<APInt>())
    perm_values.push_back(dim.getSExtValue());

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
    ShapedType input_type, ShapedType output_type) {
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
inline bool IsReshapeEquivalentToTranspose(ShapedType input_type,
                                           ShapedType output_type) {
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
  auto values = value.cast<DenseElementsAttr>().getValues<int32_t>();
  return !std::any_of(values.begin(), values.end(),
                      [](int32_t element_value) { return element_value != 1; });
}

// Checks if all elements in the constant attribute value are non-negative.
inline bool HasNonNegativeValues(Attribute value) {
  auto values = value.cast<DenseElementsAttr>().getValues<APInt>();
  return !std::any_of(
      values.begin(), values.end(),
      [](const APInt& element_value) { return element_value.isNegative(); });
}

// Utility function to get the offset between two dense attribute values.
inline TypedAttr GetOffSet(Attribute begin, Attribute end) {
  auto begin_values = begin.cast<DenseElementsAttr>().getValues<int32_t>();
  auto end_values = end.cast<DenseElementsAttr>().getValues<int32_t>();

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
  auto shaped_type = type.dyn_cast<ShapedType>();
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
inline ShapedType GetTransposedType(Value input,
                                    llvm::ArrayRef<int64_t> permutation_array) {
  auto input_type = input.getType().cast<ShapedType>();
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
inline ShapedType GetExpandedShapeType(Value input_val, int n) {
  auto expanded_shape = GetExpandedShape(input_val, n);
  return RankedTensorType::get(
      SmallVector<int64_t>{expanded_shape.begin(), expanded_shape.end()},
      mlir::cast<ShapedType>(input_val.getType()).getElementType());
}

// Returns shape of a ranked tensor.
// Precondition: output_val's is ranked tensor.
// Returns a truncated shape when `truncate` is set to true.
inline DenseElementsAttr GetShape(Value output_val, bool truncate = false) {
  auto output_shape = output_val.getType().dyn_cast<ShapedType>().getShape();

  SmallVector<int32_t> shape;
  shape.reserve(output_shape.size());

  bool needs_truncation = true;
  for (size_t dim_idx = 0; dim_idx < output_shape.size(); ++dim_idx) {
    int64_t dim = output_shape[dim_idx];
    if (truncate && needs_truncation && dim == 1) {
      continue;
    } else if (needs_truncation && dim != 1) {
      needs_truncation = false;
    }
    shape.push_back(ShapedType::isDynamic(dim) ? -1
                                               : static_cast<int32_t>(dim));
  }

  return mlir::DenseElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int>(shape.size())},
          mlir::IntegerType::get(output_val.getContext(), 32)),
      llvm::ArrayRef(shape));
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

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_UTILS_H_
