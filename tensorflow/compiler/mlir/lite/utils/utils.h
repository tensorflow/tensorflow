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
#include <cstddef>
#include <cstdint>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
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

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_UTILS_H_
