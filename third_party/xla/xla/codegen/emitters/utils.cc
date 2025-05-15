/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/emitters/utils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

namespace xla::emitters {

using mlir::DenseElementsAttr;
using mlir::ShapedType;

DenseElementsAttr GetZeroDenseElementsAttr(ShapedType shaped_type) {
  auto elem_type = shaped_type.getElementType();
  if (auto float_type = mlir::dyn_cast<mlir::FloatType>(elem_type)) {
    mlir::SmallVector<llvm::APFloat, 4> values(
        shaped_type.getNumElements(),
        mlir::APFloat::getZero(float_type.getFloatSemantics()));
    return DenseElementsAttr::get(shaped_type, values);
  }
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(elem_type)) {
    mlir::SmallVector<llvm::APInt, 4> values(
        shaped_type.getNumElements(),
        mlir::APInt::getZero(int_type.getIntOrFloatBitWidth()));
    return DenseElementsAttr::get(shaped_type, values);
  }
  llvm_unreachable("Unsupported element type");
}

}  // namespace xla::emitters
