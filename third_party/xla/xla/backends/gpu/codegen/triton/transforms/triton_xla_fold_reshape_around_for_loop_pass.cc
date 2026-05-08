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

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {

namespace ttir = ::mlir::triton;

#define GEN_PASS_DEF_TRITONXLAFOLDRESHAPEAROUNDFORLOOPPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

class FoldReshapeAroundForLoop : public mlir::OpRewritePattern<scf::ForOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      scf::ForOp for_op, mlir::PatternRewriter& rewriter) const override {
    auto yield_op = mlir::cast<scf::YieldOp>(for_op.getBody()->getTerminator());
    ValueRange yield_operands = yield_op.getOperands();

    for (auto it :
         llvm::zip(for_op.getInitArgsMutable(), for_op.getResults())) {
      OpOperand& iter_op_operand = std::get<0>(it);
      unsigned i =
          iter_op_operand.getOperandNumber() - for_op.getNumControlOperands();

      ttir::ReshapeOp reshape_op =
          yield_operands[i].getDefiningOp<ttir::ReshapeOp>();
      if (!reshape_op ||
          (reshape_op.getOperand().getType() == yield_operands[i].getType())) {
        continue;
      }

      Value inner_yield_val = reshape_op.getOperand();
      const Location op_loc = for_op.getLoc();

      // Sink rank reduction for initialization.
      Value new_init = ttir::ReshapeOp::create(
          rewriter, op_loc, inner_yield_val.getType(), iter_op_operand.get());

      // Update the yield of the original loop to provide the un-reshaped value.
      // This prevents 'replaceAndCastForOpIterArg' from cloning the original
      // reshape into the new loop body, which would otherwise cause the
      // rewriter to recursively trigger on the same pattern in the new loop.
      rewriter.modifyOpInPlace(
          yield_op, [&]() { yield_op->setOperand(i, inner_yield_val); });

      // Use the SCF utility to handle structural rewrite and cast injection.
      SmallVector<Value> new_results = mlir::scf::replaceAndCastForOpIterArg(
          rewriter, for_op, iter_op_operand, new_init,
          [](OpBuilder& b, Location loc, Type type, Value val) -> Value {
            return ttir::ReshapeOp::create(b, loc, type, val);
          });

      rewriter.replaceOp(for_op, new_results);
      return mlir::success();
    }
    return mlir::failure();
  }
};

class TritonXLAFoldReshapeAroundForLoopPass
    : public impl::TritonXLAFoldReshapeAroundForLoopPassBase<
          TritonXLAFoldReshapeAroundForLoopPass> {
 public:
  using TritonXLAFoldReshapeAroundForLoopPassBase::
      TritonXLAFoldReshapeAroundForLoopPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* mlir_context = &getContext();
    mlir::RewritePatternSet patterns(mlir_context);
    patterns.add<FoldReshapeAroundForLoop>(mlir_context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateTritonXLAFoldReshapeAroundForLoopPass() {
  return std::make_unique<TritonXLAFoldReshapeAroundForLoopPass>();
}

}  // namespace mlir::triton::xla
