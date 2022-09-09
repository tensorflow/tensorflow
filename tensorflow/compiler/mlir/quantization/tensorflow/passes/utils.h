/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_UTILS_H_

#include <type_traits>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

// TODO(b/238829558): Populate quantization config based on the
// QuantizationOptions proto. We might want to clean QuantizationMethod as well
// as this can be inferred from the proto.
using OpSet = tensorflow::quantization::OpSet;

enum class QuantizationMethod {
  kQuantizationAwareTraining,
  kPostTrainingQuantization,
  kDynamicRangeQuantization
};

// Returns true if the value has static shape.
bool HasStaticShape(Value value);

// Returns true if the value has static shape at given dims.
bool HasStaticShapeAtDims(Value value, llvm::ArrayRef<int> dims);

// Returns true if the op has any quantized tensors as input or output.
bool HasQuantizedTensors(Operation *op);

// Creates a new type that has the shape from the `old_type` and the element
// type from the `element_type`.
Type CloneTypeWithNewElementType(Type old_type, Type element_type);

// Creates an array with integer/float type.
template <typename T>
Value CreateConstValue(OpBuilder &builder, Location loc,
                       const llvm::SmallVector<int64_t> &shape,
                       const llvm::SmallVector<T> &values) {
  static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
  if (std::is_integral_v<T>) {
    auto shape_type =
        RankedTensorType::get(shape, builder.getIntegerType(sizeof(T) * 8));

    DenseIntElementsAttr attr = DenseIntElementsAttr::get(shape_type, values);
    return builder.create<TF::ConstOp>(loc, attr);
  }

  auto type = RankedTensorType::get(shape, builder.getF32Type());
  auto value_attr = DenseFPElementsAttr::get(type, values);
  return builder.create<TF::ConstOp>(loc, value_attr);
}

// Creates a 1D array with integer/float type.
template <typename T>
Value Create1DConstValue(OpBuilder &builder, Location loc,
                         const llvm::SmallVector<T> &values) {
  return CreateConstValue<T>(builder, loc,
                             {static_cast<int64_t>(values.size())}, values);
}

// Creates a scalar with integer/float type.
template <typename T>
Value CreateScalarConstValue(OpBuilder &builder, Location loc, T value) {
  return CreateConstValue<T>(builder, loc, {}, {value});
}

// Checks if the value is a constant and return its splat value.
template <typename T>
bool GetSplatValue(Value value, T &splat_value) {
  static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
  if (std::is_integral_v<T>) {
    DenseIntElementsAttr value_attr;
    if (!matchPattern(value, m_Constant(&value_attr)) ||
        !value_attr.isSplat()) {
      return false;
    }
    splat_value = value_attr.getSplatValue<T>();
    return true;
  }

  DenseFPElementsAttr value_attr;
  if (!matchPattern(value, m_Constant(&value_attr)) || !value_attr.isSplat()) {
    return false;
  }
  splat_value = value_attr.getSplatValue<T>();

  return true;
}

// Checks if the value is a constant and its splat value is equal to x.
template <typename T>
bool IsSplatValueEqual(Value value, T x) {
  T splat_value;
  if (!GetSplatValue(value, splat_value)) return false;

  return splat_value == x;
}

// Checks if two values are constants and their splat values are equal.
template <typename T>
bool AreSplatValuesEqual(Value x, Value y) {
  T splat_x, splat_y;
  if (!GetSplatValue(x, splat_x) || !GetSplatValue(y, splat_y)) {
    return false;
  }

  return splat_x == splat_y;
}

// TODO(b/241488936): Remove this function after adding a new constant folding
// pass to TensorFlow.
// Applies constant folding to the operation if possible and return the folded
// results.
llvm::SmallVector<Value> ConstantFoldOpIfPossible(Operation *op);

}  // namespace quant
}  // namespace mlir
#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_TENSORFLOW_PASSES_UTILS_H_
