//=======- EarlyLowering.cpp - Toy Lowering to Linear Algebra Dialect -=======//
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
//
// This file implements early lowering of Toy IR to Linalg Dialect: we only
// lower the computationally intensive part of the program (matmul...) to a
// dialect specialized for optimizations.
//
// This is intended to showcase how multiple dialects can cohabit in the same
// function. After this lowering, you would still have toy.print in the IR for
// example.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include "linalg1/Dialect.h"
#include "linalg1/Intrinsics.h"
#include "linalg1/ViewOp.h"
#include "linalg3/TensorOps.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"

#include <algorithm>

using namespace mlir;

namespace {
/// Utility function for type casting: this is making the type checker happy,
/// while delaying the actual work involved to convert the type. Most of the
/// time both side of the cast (producer and consumer) will be lowered to a
/// dialect like LLVM and end up with the same LLVM representation, at which
/// point this becomes a no-op and is eliminated.
Value *typeCast(PatternRewriter &builder, Value *val, Type destTy) {
  if (val->getType() == destTy)
    return val;
  return builder.create<toy::TypeCastOp>(val->getLoc(), val, destTy)
      .getResult();
}

/// Create a type cast to turn a toy.array into a memref. The Toy Array will be
/// lowered to a memref during buffer allocation, at which point the type cast
/// becomes useless.
Value *memRefTypeCast(PatternRewriter &builder, Value *val) {
  if (val->getType().isa<MemRefType>())
    return val;
  auto toyArrayTy = val->getType().dyn_cast<toy::ToyArrayType>();
  if (!toyArrayTy)
    return val;
  return typeCast(builder, val, toyArrayTy.toMemref());
}

/// Lower toy.mul to Linalg `matmul`.
///
/// This class inherit from `ConversionPattern` and override `rewrite`,
/// similarly to the PatternRewriter introduced in the previous chapter.
/// It will be called by the DialectConversion framework (see `LateLowering`
/// class below).
class MulOpConversion : public ConversionPattern {
public:
  explicit MulOpConversion(MLIRContext *context)
      : ConversionPattern(toy::MulOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                                     PatternRewriter &rewriter) const override {
    using namespace edsc;
    using intrinsics::constant_index;
    using linalg::intrinsics::range;
    using linalg::intrinsics::view;
    toy::MulOp mul = cast<toy::MulOp>(op);
    auto loc = mul.getLoc();
    Value *result = memRefTypeCast(
        rewriter, rewriter.create<toy::AllocOp>(loc, mul.getResult()->getType())
                      .getResult());
    Value *lhs = memRefTypeCast(rewriter, operands[0]);
    auto memrefLHSTy = lhs->getType().cast<MemRefType>();
    Value *rhs = memRefTypeCast(rewriter, operands[1]);
    auto memrefRHSTy = rhs->getType().cast<MemRefType>();
    mlir::edsc::ScopedContext scope(rewriter, loc);
    edsc::ValueHandle r0 =
        range(constant_index(0), constant_index(memrefLHSTy.getDimSize(0)),
              constant_index(1));
    edsc::ValueHandle r1 =
        range(constant_index(0), constant_index(memrefLHSTy.getDimSize(1)),
              constant_index(1));
    edsc::ValueHandle r2 =
        range(constant_index(0), constant_index(memrefRHSTy.getDimSize(1)),
              constant_index(1));
    auto lhsView = view(lhs, {r0, r1});
    auto rhsView = view(rhs, {r1, r2});
    auto resultView = view(result, {r0, r2});
    rewriter.create<linalg::MatmulOp>(loc, lhsView, rhsView, resultView);
    rewriter.replaceOp(op, {typeCast(rewriter, result, mul.getType())});
    return matchSuccess();
  }
};

/// This is lowering to Linalg the parts that are computationally intensive
/// (like matmul for example...) while keeping the rest of the code in the Toy
/// dialect.
struct EarlyLoweringPass : public FunctionPass<EarlyLoweringPass> {
  void runOnFunction() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();
    target.addLegalOp<toy::AllocOp, toy::TypeCastOp>();

    OwningRewritePatternList patterns;
    RewriteListBuilder<MulOpConversion>::build(patterns, &getContext());
    if (failed(applyPartialConversion(getFunction(), target,
                                      std::move(patterns)))) {
      emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering Toy\n");
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

namespace toy {
Pass *createEarlyLoweringPass() { return new EarlyLoweringPass(); }
} // namespace toy
