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

#include <cassert>
#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace xla::cpu {

#define GEN_PASS_DECL_ADDREDUCTIONFASTMATHFLAGSPASS
#define GEN_PASS_DEF_ADDREDUCTIONFASTMATHFLAGSPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;

struct RewriteCallPattern
    : public mlir::OpInterfaceRewritePattern<mlir::CallOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::CallOpInterface call_op,
      mlir::PatternRewriter& rewriter) const override {
    if (!call_op->hasAttr("xla.is_reduction")) {
      return rewriter.notifyMatchFailure(call_op, "Call is not a reduction.");
    }

    mlir::func::FuncOp callee =
        mlir::dyn_cast<mlir::func::FuncOp>(call_op.resolveCallable());
    if (!callee) {
      // Could be a call to an external function.
      return rewriter.notifyMatchFailure(call_op, "Could not resolve callee.");
    }

    callee->walk([&rewriter](mlir::Operation* op) {
      if (auto fm_op =
              mlir::dyn_cast_or_null<mlir::arith::ArithFastMathInterface>(op)) {
        ma::FastMathFlagsAttr current_fm_attr = fm_op.getFastMathFlagsAttr();
        ma::FastMathFlags current_fm_flags = current_fm_attr
                                                 ? current_fm_attr.getValue()
                                                 : ma::FastMathFlags::none;
        ma::FastMathFlagsAttr new_fm_flags =
            rewriter.getAttr<ma::FastMathFlagsAttr>(current_fm_flags |
                                                    ma::FastMathFlags::reassoc);
        op->setAttr(fm_op.getFastMathAttrName(), new_fm_flags);
      }
    });

    // Remove the attribute to avoid an infinite loop.
    call_op->removeAttr("xla.is_reduction");

    return mlir::success();
  }
};

class AddReductionFastMathFlagsPass
    : public impl::AddReductionFastMathFlagsPassBase<
          AddReductionFastMathFlagsPass> {
 public:
  using AddReductionFastMathFlagsPassBase::AddReductionFastMathFlagsPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteCallPattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateAddReductionFastMathFlagsPass() {
  return std::make_unique<AddReductionFastMathFlagsPass>();
}

}  // namespace xla::cpu
