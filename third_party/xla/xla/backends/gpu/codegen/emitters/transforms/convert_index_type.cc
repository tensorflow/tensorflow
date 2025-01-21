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
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"

namespace xla {
namespace gpu {
namespace {

using mlir::ImplicitLocOpBuilder;
using mlir::IndexType;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::Type;

namespace arith = mlir::arith;

#define GEN_PASS_DEF_CONVERTINDEXTYPEPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

// Rewrites a binary elementwise op on 'index' types to a binary elementwise
// op on integers with the specified bit width.
template <typename BinaryElementwiseOp>
class RewriteIndexBinaryElementwiseOp
    : public OpRewritePattern<BinaryElementwiseOp> {
 public:
  using OpRewritePattern<BinaryElementwiseOp>::OpRewritePattern;
  RewriteIndexBinaryElementwiseOp(MLIRContext* context, uint64_t index_bitwidth)
      : OpRewritePattern<BinaryElementwiseOp>(context, 2),
        index_bitwidth_(index_bitwidth) {}

  LogicalResult matchAndRewrite(BinaryElementwiseOp op,
                                PatternRewriter& rewriter) const override {
    CHECK_EQ(op->getNumOperands(), 2);
    CHECK_EQ(op->getNumResults(), 1);

    auto is_index = [](Type type) { return llvm::isa<IndexType>(type); };

    bool operands_are_indices = absl::c_all_of(op->getOperandTypes(), is_index);
    bool result_is_index = is_index(op->getResultTypes().front());

    if (!operands_are_indices || !result_is_index) {
      return rewriter.notifyMatchFailure(
          op, "operation has non-index operands or results");
    }

    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    b.setInsertionPoint(op);

    Type index_type = IndexType::get(op->getContext());
    Type dst_type = b.getIntegerType(index_bitwidth_);
    auto lhs = b.create<arith::IndexCastUIOp>(dst_type, op->getOperand(0));
    auto rhs = b.create<arith::IndexCastUIOp>(dst_type, op->getOperand(1));
    auto new_op = b.create<BinaryElementwiseOp>(lhs, rhs);

    rewriter.replaceAllUsesWith(
        op.getResult(),
        b.create<arith::IndexCastUIOp>(index_type, new_op.getResult()));

    return mlir::success();
  }

 private:
  uint64_t index_bitwidth_;
};

struct ConvertIndexTypePass
    : public impl::ConvertIndexTypePassBase<ConvertIndexTypePass> {
 public:
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    mlir::DataLayout layout(getOperation());
    std::optional<uint64_t> index_bitwidth =
        layout.getTypeIndexBitwidth(IndexType::get(ctx));
    CHECK(index_bitwidth.has_value());
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewriteIndexBinaryElementwiseOp<arith::AddIOp>,
                 RewriteIndexBinaryElementwiseOp<arith::DivUIOp>,
                 RewriteIndexBinaryElementwiseOp<arith::MulIOp>,
                 RewriteIndexBinaryElementwiseOp<arith::RemUIOp>,
                 RewriteIndexBinaryElementwiseOp<arith::SubIOp>>(
        ctx, *index_bitwidth);
    arith::IndexCastUIOp::getCanonicalizationPatterns(patterns, ctx);

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateConvertIndexTypePass() {
  return std::make_unique<ConvertIndexTypePass>();
}

}  // namespace gpu
}  // namespace xla
