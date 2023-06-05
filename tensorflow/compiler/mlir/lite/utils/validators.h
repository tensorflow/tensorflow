/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This header file defines common validators used by TFLite transformation
// passes to validate op attributes or values.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_VALIDATORS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_VALIDATORS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project

namespace mlir {
namespace TFL {

// TODO(jpienaar): Change these to being one of these variants and/or generate
// these predicates.

// Returns true if the given TensorFlow op does not have a `data_format`
// attribute (then default to "NHWC"), or its `data_format` attribute is "NHWC".
inline bool TFDataFormatIsNHWC(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("data_format");
  return !attr || attr.getValue() == "NHWC";
}

// Returns true if the given TensorFlow op does not have a `data_format`
// attribute (then default to "NDHWC"), or its `data_format` attribute is
// "NDHWC".
inline bool TFDataFormatIsNDHWC(Operation *op) {
  auto attr = op->getAttrOfType<StringAttr>("data_format");
  return !attr || attr.getValue() == "NDHWC";
}

// Returns true if the given `op`
//   * has an attribute with the given `name`,
//   * and the attribute is an integer list of the form [1, X, Y, 1],
// and writes X, Y as 32-bit integer attribute to `x`, `y`.
bool TFIntListIs1XY1(Operation *op, StringRef name, IntegerAttr *x,
                     IntegerAttr *y);

// Returns true if the attribute is an integer list of the form [1, X, Y, 1],
bool TFIntListIs1XY1(Attribute attr);

// Returns true if the given `op`
//   * has an attribute with the given `name`,
//   * and the attribute is an integer list of the form [1, X, Y, Z, 1],
// and writes X, Y as 32-bit integer attribute to `x`, `y`, z.
bool TFIntListIs1XYZ1(Operation *op, StringRef name, IntegerAttr *x,
                      IntegerAttr *y, IntegerAttr *z);

// Returns true if every element of the attribute is 1. All elements of `attr`
// must be `IntegerAttr`.
bool TFIntListIsAllOnes(Attribute attr);

// Returns true iff the given value is a float32 tensor.
// is "DT_FLOAT".
inline bool TFTypeIsFloat32Tensor(Value value) {
  auto tensorType = value.getType().dyn_cast<TensorType>();
  if (!tensorType) return false;
  return tensorType.getElementType().isF32();
}

// Returns true iff the given value is a bf16 tensor.
inline bool TFTypeIsBFloat16Tensor(Value value) {
  auto tensorType = value.getType().dyn_cast<TensorType>();
  if (!tensorType) return false;
  return tensorType.getElementType().isBF16();
}

// Returns true iff the given value is a f16 tensor.
inline bool TFTypeIsHalfTensor(Value value) {
  auto tensorType = value.getType().dyn_cast<TensorType>();
  if (!tensorType) return false;
  return tensorType.getElementType().isF16();
}

// Returns true iff the given value is a f16 or bf16 tensor.
inline bool TFTypeIsBFloat16OrHalfTensor(Value value) {
  return TFTypeIsBFloat16Tensor(value) || TFTypeIsHalfTensor(value);
}

// Returns true iff the given TensorFlow op has a `padding` attribute whose
// value is "SAME" or "VALID", and writes the attribute to `padding`.
inline bool TFPaddingIsSameOrValid(Operation *op, StringAttr *padding) {
  auto padding_attr = op->getAttrOfType<StringAttr>("padding");
  if (padding_attr.getValue() != "SAME" && padding_attr.getValue() != "VALID")
    return false;
  *padding = padding_attr;
  return true;
}

/// Returns whether the given `a` and `b` have broadcast-compatible
/// types.
bool IsBroadcastableElementsAttrs(mlir::TypedAttr a, mlir::TypedAttr b);
// Returns true if every dimension of the attribute is 1 except the last one.
bool IsDimensionsDegenerateExceptLastOne(mlir::TypedAttr val);
// Returns true if every element is 1 except the last one.
bool IsDimensionsDegenerateExceptLastOne(ArrayRef<int64_t> elements_shape);

}  // end namespace TFL
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_VALIDATORS_H_
