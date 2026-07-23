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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"

namespace xla::cpu {

#define GEN_PASS_DECL_TENSOROPSTOBUFFERIZABLEPASS
#define GEN_PASS_DEF_TENSOROPSTOBUFFERIZABLEPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

struct TensorToArithBitcast : mlir::OpRewritePattern<mlir::tensor::BitcastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::BitcastOp op,
      mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::BitcastOp>(op, op.getType(),
                                                        op.getOperand());
    return mlir::success();
  }
};

struct LowerXTileMask : mlir::OpRewritePattern<xtile::MaskOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::MaskOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::RankedTensorType tensor_type =
        mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!tensor_type) return mlir::failure();

    llvm::SmallVector<int64_t> masked_dims = op.getMaskedDimensions();
    if (masked_dims.empty()) {
      rewriter.replaceOp(op, op.getSource());
      return mlir::success();
    }

    mlir::Location loc = op.getLoc();
    mlir::Value mask = nullptr;

    for (int64_t d : masked_dims) {
      int64_t dim_bound = op.getBounds()[d];
      int64_t dim_size = tensor_type.getDimSize(d);

      auto iota_type =
          mlir::RankedTensorType::get({dim_size}, rewriter.getI32Type());
      auto range = mlir::stablehlo::IotaOp::create(rewriter, loc, iota_type, 0);

      auto bcast_type = mlir::RankedTensorType::get(tensor_type.getShape(),
                                                    rewriter.getI32Type());
      auto bcast = mlir::stablehlo::BroadcastInDimOp::create(
          rewriter, loc, bcast_type, range, llvm::ArrayRef<int64_t>{d});

      auto bound_attr = mlir::DenseElementsAttr::get(
          bcast_type, rewriter.getI32IntegerAttr(dim_bound));
      auto bound_const =
          mlir::arith::ConstantOp::create(rewriter, loc, bound_attr);

      auto mask_type = mlir::RankedTensorType::get(tensor_type.getShape(),
                                                   rewriter.getI1Type());
      mlir::Value dim_mask = mlir::arith::CmpIOp::create(
          rewriter, loc, mask_type, mlir::arith::CmpIPredicate::slt, bcast,
          bound_const);

      if (!mask) {
        mask = dim_mask;
      } else {
        mask = mlir::arith::AndIOp::create(rewriter, loc, mask, dim_mask);
      }
    }

    auto scalar_value = op.getValue();
    auto scalar_tensor_type =
        mlir::RankedTensorType::get({}, scalar_value.getType());
    auto value_tensor = mlir::tensor::FromElementsOp::create(
        rewriter, loc, scalar_tensor_type, scalar_value);
    auto neutral = mlir::stablehlo::BroadcastInDimOp::create(
        rewriter, loc, tensor_type, value_tensor, llvm::ArrayRef<int64_t>{});

    rewriter.replaceOp(
        op, mlir::arith::SelectOp::create(rewriter, loc, tensor_type, mask,
                                          op.getSource(), neutral));
    return mlir::success();
  }
};

class TensorOpsToBufferizablePass
    : public impl::TensorOpsToBufferizablePassBase<
          TensorOpsToBufferizablePass> {
 public:
  using TensorOpsToBufferizablePassBase::TensorOpsToBufferizablePassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<TensorToArithBitcast, LowerXTileMask>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace
}  // namespace xla::cpu
