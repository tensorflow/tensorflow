//===- QuantOps.cpp - Quantization Type and Ops Implementation --*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/QuantOps/QuantOps.h"
#include "TypeDetail.h"

#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

using namespace mlir;
using namespace mlir::quant;
using namespace mlir::quant::detail;

QuantizationDialect::QuantizationDialect(MLIRContext *context)
    : Dialect(/*name=*/"quant", context) {
  addTypes<AnyQuantizedType, UniformQuantizedType,
           UniformQuantizedPerAxisType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/QuantOps/QuantOps.cpp.inc"
      >();
}

OpFoldResult StorageCastOp::fold(ArrayRef<Attribute> operands) {
  /// Matches x -> [scast -> scast] -> y, replacing the second scast with the
  /// value of x if the casts invert each other.
  auto srcScastOp = dyn_cast_or_null<StorageCastOp>(arg()->getDefiningOp());
  if (!srcScastOp || srcScastOp.arg()->getType() != getType())
    return OpFoldResult();
  return srcScastOp.arg();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/QuantOps/QuantOps.cpp.inc"
