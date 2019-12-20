//===- QuantOps.cpp - Quantization Type and Ops Implementation --*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

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
