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

using namespace mlir;
using namespace mlir::quant;
using namespace mlir::quant::detail;

#define GET_OP_CLASSES
#include "mlir/Dialect/QuantOps/QuantOps.cpp.inc"

namespace {

/// Matches x -> [scast -> scast] -> y, replacing the second scast with the
/// value of x if the casts invert each other.
class RemoveRedundantStorageCastsRewrite : public RewritePattern {
public:
  RemoveRedundantStorageCastsRewrite(MLIRContext *context)
      : RewritePattern(StorageCastOp::getOperationName(), 1, context) {}

  PatternMatchResult match(Operation *op) const override {
    auto scastOp = cast<StorageCastOp>(op);
    if (matchPattern(scastOp.arg(), m_Op<StorageCastOp>())) {
      auto srcScastOp = cast<StorageCastOp>(scastOp.arg()->getDefiningOp());
      if (srcScastOp.arg()->getType() == scastOp.getResult()->getType()) {
        return matchSuccess();
      }
    }
    return matchFailure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto scastOp = cast<StorageCastOp>(op);
    auto srcScastOp = cast<StorageCastOp>(scastOp.arg()->getDefiningOp());
    rewriter.replaceOp(op, srcScastOp.arg());
  }
};

} // end anonymous namespace

void StorageCastOp::getCanonicalizationPatterns(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.push_back(
      llvm::make_unique<RemoveRedundantStorageCastsRewrite>(context));
}

QuantizationDialect::QuantizationDialect(MLIRContext *context)
    : Dialect(/*name=*/"quant", context) {
  addTypes<AnyQuantizedType, UniformQuantizedType,
           UniformQuantizedPerAxisType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/QuantOps/QuantOps.cpp.inc"
      >();
}
