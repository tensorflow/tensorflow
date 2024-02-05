/* Copyright 2019 The OpenXLA Authors.

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

// Thsi file implements passes to convert complex operations to equivalent real
// value operations. This does not include removing complex values from function
// argument or return types.

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "utils/hlo_utils.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LOWERCOMPLEXPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {
class LowerComplexPass : public impl::LowerComplexPassBase<LowerComplexPass> {
 public:
  /// Performs the lowering to MHLO dialect.
  void runOnOperation() override;
};

#include "lower_complex/generated_lower_complex.inc"

// Lowers the complex operations that can be represented using other operations.
void LowerComplexPass::runOnOperation() {
  // Add lowering patterns to the list.
  RewritePatternSet patterns(&getContext());
  mlir::mhlo::populateComplexLoweringPatterns(&getContext(), &patterns);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

class ConvertComplexDot : public OpRewritePattern<DotOp> {
 public:
  using OpRewritePattern<DotOp>::OpRewritePattern;

  // Will decompose mlir::DotOp with complex parameters down to
  // four Dot operations in the following fashion:
  //   result.real = lhs.real <DOT> rhs.real - lhs.imag <DOT> rhs.imag
  //   result.imag = lhs.imag <DOT> rhs.real + lhs.real <DOT> rhs.imag
  //   result = complex(result.real, result.imag)
  LogicalResult matchAndRewrite(DotOp dot,
                                PatternRewriter &rewriter) const override {
    auto precision = dot.getPrecisionConfigAttr();
    auto lhs = dot.getLhs();
    auto rhs = dot.getRhs();
    ShapedType lhsType = lhs.getType();
    ShapedType rhsType = rhs.getType();
    if (!isa<ComplexType>(lhsType.getElementType()) ||
        !isa<ComplexType>(rhsType.getElementType())) {
      return rewriter.notifyMatchFailure(dot, "lhs/rhs types are not complex");
    }

    Location loc = dot.getLoc();
    Value lhsReal = rewriter.createOrFold<mhlo::RealOp>(loc, lhs);
    Value lhsImag = rewriter.createOrFold<mhlo::ImagOp>(loc, lhs);
    Value rhsReal = rewriter.createOrFold<mhlo::RealOp>(loc, rhs);
    Value rhsImag = rewriter.createOrFold<mhlo::ImagOp>(loc, rhs);
    auto resultType = dot.getType();
    Type newType = hlo::createRealType(resultType);

    Value realComponent = rewriter.create<mhlo::SubtractOp>(
        loc,
        rewriter.create<mhlo::DotOp>(loc, newType, lhsReal, rhsReal, precision),
        rewriter.create<mhlo::DotOp>(loc, newType, lhsImag, rhsImag,
                                     precision));
    Value imagComponent = rewriter.create<mhlo::AddOp>(
        loc,
        rewriter.create<mhlo::DotOp>(loc, newType, lhsReal, rhsImag, precision),
        rewriter.create<mhlo::DotOp>(loc, newType, lhsImag, rhsReal,
                                     precision));
    Value result = rewriter.create<mhlo::ComplexOp>(
        loc, resultType, realComponent, imagComponent);
    rewriter.replaceOp(dot, result);
    return success();
  }
};

}  // end anonymous namespace
}  // end namespace mhlo
}  // end namespace mlir

void mlir::mhlo::populateComplexLoweringPatterns(MLIRContext *context,
                                                 RewritePatternSet *patterns) {
  populateWithGenerated(*patterns);
  patterns->insert<mlir::mhlo::ConvertComplexDot>(context);
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
mlir::mhlo::createLowerComplexPass() {
  return std::make_unique<mlir::mhlo::LowerComplexPass>();
}
