/* Copyright 2024 The OpenXLA Authors.

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

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Complex/IR/Complex.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_LOWERXLAGPUTOSCFPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

using mlir::success;

struct RewritePredicatedInsert : mlir::OpRewritePattern<PredicatedInsertOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      PredicatedInsertOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(
        op, op.getCondition(),
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(
              loc, b.create<mlir::tensor::InsertOp>(
                        loc, op.getValue(), op.getDest(), op.getIndices())
                       .getResult());
        },
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, op.getDest());
        });
    return success();
  }
};

struct RewritePredicatedExtract : mlir::OpRewritePattern<PredicatedExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      PredicatedExtractOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(
        op, op.getCondition(),
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(
              loc, b.create<mlir::tensor::ExtractOp>(loc, op.getSrc(),
                                                     op.getIndices())
                       .getResult());
        },
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          b.create<mlir::scf::YieldOp>(loc, op.getFallback());
        });
    return success();
  }
};

struct RewriteShuffleReduce : mlir::OpRewritePattern<ShuffleReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ShuffleReduceOp op, mlir::PatternRewriter& rewriter) const override {
    int max_distance =
        mlir::cast<mlir::IntegerAttr>(op->getAttr("max_distance")).getInt();
    // TODO(jreiffers): Do this in a verifier.
    if (max_distance & (max_distance - 1) || max_distance >= WarpSize()) {
      return op->emitOpError("max_distance must be a power of 2 < WarpSize()");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    mlir::ValueRange values = op.getOperands();
    for (int distance = max_distance; distance > 0; distance /= 2) {
      namespace ml = mlir::LLVM;
      auto shuffle_32 = [&](mlir::Value v) {
        return b
            .create<mlir::gpu::ShuffleOp>(v, distance, WarpSize(),
                                          mlir::gpu::ShuffleMode::DOWN)
            .getShuffleResult();
      };

      auto shuffle_int_or_float = [&](mlir::Value value) {
        auto ty = value.getType();
        int bit_width = ty.getIntOrFloatBitWidth();
        if (bit_width == 32) {
          return shuffle_32(value);
        }
        int n_shuffles = CeilOfRatio(bit_width, 32);
        auto int_ty = b.getIntegerType(bit_width);
        auto padded_int_ty = b.getIntegerType(n_shuffles * 32);
        value = b.create<mlir::arith::BitcastOp>(int_ty, value);
        value = b.create<mlir::arith::ExtUIOp>(padded_int_ty, value);
        auto vector_type = ml::getVectorType(b.getI32Type(), n_shuffles);
        value = b.create<ml::BitcastOp>(vector_type, value);
        mlir::Value result_vec = b.create<ml::UndefOp>(vector_type);
        for (int i = 0; i < n_shuffles; ++i) {
          auto idx = b.create<mlir::arith::ConstantIntOp>(i, 32);
          result_vec = b.create<ml::InsertElementOp>(
              result_vec,
              shuffle_32(b.create<ml::ExtractElementOp>(value, idx)), idx);
        }
        value = b.create<ml::BitcastOp>(padded_int_ty, result_vec);
        value = b.create<mlir::arith::TruncIOp>(int_ty, value);
        value = b.create<ml::BitcastOp>(ty, value);
        return value;
      };

      auto shuffle = [&](mlir::Value value) -> mlir::Value {
        if (mlir::isa<mlir::ComplexType>(value.getType())) {
          return b.create<mlir::complex::CreateOp>(
              value.getType(),
              shuffle_int_or_float(b.create<mlir::complex::ReOp>(value)),
              shuffle_int_or_float(b.create<mlir::complex::ImOp>(value)));
        }
        if (value.getType().isUnsignedInteger()) {
          auto ty = value.getType();
          auto signless_ty = b.getIntegerType(ty.getIntOrFloatBitWidth());
          value = b.create<mlir::UnrealizedConversionCastOp>(
                       mlir::TypeRange{signless_ty}, value)
                      .getResult(0);
          value = shuffle_int_or_float(value);
          value = b.create<mlir::UnrealizedConversionCastOp>(
                       mlir::TypeRange{ty}, value)
                      .getResult(0);
          return value;
        }
        return shuffle_int_or_float(value);
      };

      llvm::SmallVector<mlir::Value> args = values;
      for (auto value : values) {
        args.push_back(shuffle(value));
      }
      values = b.create<PureCallOp>(op.getResultTypes(),
                                    op.getReducerAttr().getAttr(), args)
                   .getResults();
    }
    rewriter.replaceOp(op, values);
    return success();
  }
};

class LowerXlaGpuToScfPass
    : public impl::LowerXlaGpuToScfPassBase<LowerXlaGpuToScfPass> {
 public:
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewritePredicatedInsert, RewritePredicatedExtract,
                 RewriteShuffleReduce>(&getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<::mlir::Pass> CreateLowerXlaGpuToScfPass() {
  return std::make_unique<LowerXlaGpuToScfPass>();
}

}  // namespace gpu
}  // namespace xla
