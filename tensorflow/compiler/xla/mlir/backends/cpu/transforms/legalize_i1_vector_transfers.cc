/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <optional>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/xla_cpu/ir/xla_cpu.h"

namespace xla {
namespace cpu {
namespace {

#define GEN_PASS_DEF_LEGALIZEI1VECTORTRANSFEROPSPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

class LegalizeI1VectorTransferOpsPass
    : public impl::LegalizeI1VectorTransferOpsPassBase<
          LegalizeI1VectorTransferOpsPass> {
  void runOnOperation() override;
};

Value CastToI8(Value in, ImplicitLocOpBuilder& b, bool optional = false) {
  auto ty = in.getType();
  assert(optional || getElementTypeOrSelf(ty).isInteger(1));
  if (!getElementTypeOrSelf(ty).isInteger(1)) {
    return {};
  }

  if (auto vec_ty = ty.dyn_cast<VectorType>()) {
    return b.create<arith::ExtUIOp>(
        vec_ty.cloneWith(std::nullopt, b.getI8Type()), in);
  }
  if (auto memref_ty = ty.dyn_cast<MemRefType>()) {
    auto cast_ty = memref_ty.cloneWith(std::nullopt, b.getI8Type());
    return b.create<xla_cpu::MemRefElementCastOp>(cast_ty, in);
  }
  if (ty == b.getI1Type()) {
    return b.create<arith::ExtUIOp>(b.getI8Type(), in);
  }
  return {};
}

class I1TransferReadLowering : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(op);
    Value cast_src = CastToI8(op.getSource(), b, /*optional=*/true);
    if (!cast_src) {
      return failure();
    }

    auto cast_result_ty =
        op.getVector().getType().cloneWith(std::nullopt, b.getI8Type());
    TypedValue<VectorType> new_read =
        b.create<vector::TransferReadOp>(
             TypeRange{cast_result_ty}, cast_src, op.getIndices(),
             op.getPermutationMap(), CastToI8(op.getPadding(), b), op.getMask(),
             op.getInBoundsAttr())
            .getResult();
    Value zero = b.create<arith::ConstantOp>(
        DenseElementsAttr::get(new_read.getType(), b.getI8IntegerAttr(0)));
    auto result =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::ne, new_read, zero);
    rewriter.replaceOp(op, {result});
    return success();
  };
};

class I1TransferWriteLowering
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    b.setInsertionPoint(op);
    // Confusingly, the destination is called 'source'.
    auto cast_dst = CastToI8(op.getSource(), b, /*optional=*/true);
    if (!cast_dst) {
      return failure();
    }

    op.getVectorMutable().assign(CastToI8(op.getVector(), b));
    op.getSourceMutable().assign(cast_dst);
    return success();
  };
};

void LegalizeI1VectorTransferOpsPass::runOnOperation() {
  func::FuncOp func = getOperation();
  MLIRContext* ctx = func.getContext();

  RewritePatternSet patterns(ctx);
  patterns.insert<I1TransferReadLowering, I1TransferWriteLowering>(ctx);
  // TODO(jreiffers): Handle other transfer ops if we need them (load,
  // maskedload, etc.).

  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeI1VectorTransferOpsPass() {
  return std::make_unique<LegalizeI1VectorTransferOpsPass>();
}

}  // namespace cpu
}  // namespace xla
