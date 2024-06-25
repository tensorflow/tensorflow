/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_REWRITE_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_REWRITE_UTIL_H_

#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TF {

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

// Returns true if `value` is compile-time constant and its splat value equals
// to `raw_value`.
template <typename T>
bool IsConstantValueOf(Value value, T raw_value) {
  auto element_type = mlir::cast<ShapedType>(value.getType()).getElementType();
  if (mlir::isa<FloatType>(element_type)) {
    DenseFPElementsAttr float_attr;
    if (matchPattern(value, m_Constant(&float_attr)) && float_attr.isSplat() &&
        float_attr.getSplatValue<APFloat>().isExactlyValue(raw_value))
      return true;
  } else if (mlir::isa<IntegerType>(element_type)) {
    DenseIntElementsAttr int_attr;
    if (matchPattern(value, m_Constant(&int_attr)) && int_attr.isSplat() &&
        int_attr.getSplatValue<APInt>() == raw_value)
      return true;
  }

  return false;
}

// Returns true if `op` is placed on GPU device, and false if it's on other
// devices or the device is not specified.
bool IsOnGpuDevice(mlir::Operation *op);

// Wrappers for CopyDeviceAndUnderscoredAttributes
void CopyDeviceAndUnderscoredAttributesAdaptor(mlir::OpResult src,
                                               mlir::OpResult dest);
void CopyDeviceAndUnderscoredAttributesAdaptor(mlir::Operation *src,
                                               mlir::OpResult dest);
void CopyDeviceAndUnderscoredAttributesAdaptor(mlir::Operation *src,
                                               mlir::Operation *dest);

// Wrappers for CopyXlaOutsideCompilationAttributes
void CopyXlaOutsideCompilationAttributesAdaptor(mlir::OpResult src,
                                                mlir::OpResult dest);
void CopyXlaOutsideCompilationAttributesAdaptor(mlir::Operation *src,
                                                mlir::OpResult dest);
void CopyXlaOutsideCompilationAttributesAdaptor(mlir::Operation *src,
                                                mlir::Operation *dest);

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_TRANSFORMS_REWRITE_UTIL_H_
