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

#include <memory>
#include <utility>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace mt = ::mlir::triton;

#define GEN_PASS_DEF_ROUNDF32TOTF32FORTF32DOTREWRITEPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

class Tf32DotPattern : public OpRewritePattern<mt::DotOp> {
 public:
  explicit Tf32DotPattern(MLIRContext *context)
      : OpRewritePattern<mt::DotOp>(context) {}

  using OpRewritePattern<mt::DotOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mt::DotOp op, PatternRewriter &rewriter) const override {
    constexpr auto tf32_args_rounded = "tf32_arguments_rounded";
    if (op.getInputPrecision() != mt::InputPrecision::TF32) return failure();
    if (!op.getA().getType().getElementType().isF32()) return failure();
    if (!op.getB().getType().getElementType().isF32()) return failure();
    if (op->hasAttr(tf32_args_rounded)) return failure();

    auto f32ToTF32 = [&](Value value) -> Value {
      return rewriter
          .create<ElementwiseInlineAsmOp>(
              op.getLoc(), value.getType(), "cvt.rna.tf32.f32 $0, $1;", "=r,r",
              /*isPure=*/true, /*pack=*/1, ArrayRef<Value>{value})
          ->getResult(0);
    };
    auto lhs = f32ToTF32(op.getA());
    auto rhs = f32ToTF32(op.getB());
    auto dot = rewriter.replaceOpWithNewOp<mt::DotOp>(
        op, op.getC().getType(), lhs, rhs, op.getC(), mt::InputPrecision::TF32,
        /*maxNumImpreciseAcc=*/0);
    dot->setAttr(tf32_args_rounded, rewriter.getUnitAttr());

    return success();
  }
};

struct RoundF32ToTF32ForTf32DotRewritePass
    : public impl::RoundF32ToTF32ForTf32DotRewritePassBase<
          RoundF32ToTF32ForTf32DotRewritePass> {
  void runOnOperation() override {
    auto module = getOperation();
    RewritePatternSet patterns(&getContext(),
                               std::make_unique<Tf32DotPattern>(&getContext()));
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateRoundF32ToTF32ForTf32DotRewritePass() {
  return std::make_unique<RoundF32ToTF32ForTf32DotRewritePass>();
}

}  // namespace mlir::triton::xla
