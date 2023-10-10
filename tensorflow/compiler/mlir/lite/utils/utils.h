/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
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

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_UTILS_H_
