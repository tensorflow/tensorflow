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

// This file defines helpers useful when creating or manipulating lhlo/hlo.

#include "tensorflow/compiler/mlir/xla/convert_op_folder.h"

#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project

namespace mlir {
namespace xla {

mlir::ElementsAttr ConvertElementsAttr(const mlir::ElementsAttr& elements,
                                       mlir::Type new_type) {
  auto old_type = getElementTypeOrSelf(elements);
  size_t bit_width = new_type.isBF16() ? 64 : new_type.getIntOrFloatBitWidth();

  if (old_type.isa<mlir::FloatType>()) {
    // mapValues always takes a function returning APInt, even when the output
    // is actually float.
    using func_type = mlir::APInt(const llvm::APFloat&);
    if (auto newFloatType = new_type.dyn_cast<mlir::FloatType>()) {
      // Float -> Float
      return elements.mapValues(
          new_type, llvm::function_ref<func_type>(
                        [&newFloatType](const llvm::APFloat& floatVal) {
                          llvm::APFloat newDouble(
                              mlir::FloatAttr::getValueAsDouble(floatVal));
                          bool loses_info = false;
                          newDouble.convert(newFloatType.getFloatSemantics(),
                                            llvm::APFloat::rmNearestTiesToEven,
                                            &loses_info);
                          return newDouble.bitcastToAPInt();
                        }));
    }
    // Float -> Int
    return elements.mapValues(
        new_type, llvm::function_ref<func_type>(
                      [&bit_width](const llvm::APFloat& floatVal) {
                        return llvm::APInt(
                            bit_width,
                            mlir::FloatAttr::getValueAsDouble(floatVal));
                      }));
  }

  // old_type is Integer
  // mapValues always takes a function returning APInt, even when the output
  // is actually float.
  using func_type = llvm::APInt(const llvm::APInt&);
  if (auto newFloatType = new_type.dyn_cast<mlir::FloatType>()) {
    // Int -> Float
    return elements.mapValues(
        new_type, llvm::function_ref<func_type>([&newFloatType](
                                                    const llvm::APInt& intVal) {
          llvm::APFloat newDouble(static_cast<double>(intVal.getSExtValue()));
          bool loses_info = false;
          newDouble.convert(newFloatType.getFloatSemantics(),
                            llvm::APFloat::rmNearestTiesToEven, &loses_info);
          return newDouble.bitcastToAPInt();
        }));
  }
  // new_type is Integer
  // Int -> Int
  return elements.mapValues(
      new_type,
      llvm::function_ref<func_type>([&bit_width](const llvm::APInt& intVal) {
        return llvm::APInt(bit_width, intVal.getSExtValue());
      }));
}

}  // namespace xla
}  // namespace mlir
