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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/transforms/passes.h"

namespace xla {
namespace emitters {
namespace {

#define GEN_PASS_DEF_LOWERXLAMATHLIBPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

struct LowerXlaMathLibPattern
    : public mlir::OpRewritePattern<mlir::math::ExpOp> {
  using OpRewritePattern::OpRewritePattern;
  explicit LowerXlaMathLibPattern(mlir::MLIRContext* context)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ExpOp op, mlir::PatternRewriter& rewriter) const override {
    // Only convert F64 (or vectorized F64) exp operations
    auto op_type = op.getOperand().getType();
    bool op_is_f64 = mlir::isa<mlir::FloatType>(op_type) &&
                     mlir::dyn_cast<mlir::FloatType>(op_type).isF64();
    bool op_is_f64_vector =
        mlir::isa<mlir::VectorType>(op_type) &&
        mlir::dyn_cast<mlir::VectorType>(op_type).getElementType().isF64();
    if (!(op_is_f64 || op_is_f64_vector)) {
      return rewriter.notifyMatchFailure(op, "not an f64 exp operation");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    // Get or declare the xla.exp.f64 function
    auto func_type = builder.getFunctionType({op_type}, {op_type});
    auto module_op = op->getParentOfType<mlir::ModuleOp>();
    mlir::func::FuncOp xla_exp_func =
        module_op.lookupSymbol<mlir::func::FuncOp>("xla.exp.f64");

    if (!xla_exp_func) {
      // Insert function declaration
      auto insertion_point = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(module_op.getBody());
      xla_exp_func =
          builder.create<mlir::func::FuncOp>("xla.exp.f64", func_type);
      xla_exp_func.setPrivate();
      builder.restoreInsertionPoint(insertion_point);
    }

    // Replace math.exp with call to xla.exp.f64
    auto call_op = builder.create<mlir::func::CallOp>(
        "xla.exp.f64", mlir::TypeRange{op_type},
        mlir::ValueRange{op.getOperand()});

    rewriter.replaceOp(op, call_op.getResult(0));
    return mlir::success();
  }
};

class LowerXlaMathLibPass
    : public impl::LowerXlaMathLibPassBase<LowerXlaMathLibPass> {
 public:
  LowerXlaMathLibPass()
      : impl::LowerXlaMathLibPassBase<LowerXlaMathLibPass>() {}

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerXlaMathLibPattern>(&getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerXlaMathLibPass() {
  return std::make_unique<LowerXlaMathLibPass>();
}

}  // namespace emitters
}  // namespace xla
