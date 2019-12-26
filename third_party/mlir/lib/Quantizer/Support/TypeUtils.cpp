//===- TypeUtils.cpp - Helper function for manipulating types -------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Quantizer/Support/TypeUtils.h"

#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::quantizer;

Type mlir::quantizer::getElementOrPrimitiveType(Type t) {
  if (auto sType = t.dyn_cast<ShapedType>()) {
    return sType.getElementType();
  } else {
    return t;
  }
}
