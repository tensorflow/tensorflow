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
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/service/gpu/fusions/ir/xla_gpu_ops.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

#define GEN_PASS_DEF_LOWERXLAGPUTOSCFPASS
#define GEN_PASS_DEF_LOWERXLAGPULOOPSTOSCFPASS
#include "xla/service/gpu/fusions/transforms/passes.h.inc"

using mlir::ImplicitLocOpBuilder;
using mlir::Location;
using mlir::OpBuilder;
using mlir::SmallVector;
using mlir::success;
using mlir::Value;
using mlir::ValueRange;
using mlir::scf::IfOp;

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

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ValueRange values = op.getOperands();
    for (int distance = max_distance; distance > 0; distance /= 2) {
      namespace ml = mlir::LLVM;
      auto shuffle_32 = [&](Value v) {
        return b
            .create<mlir::gpu::ShuffleOp>(v, distance, WarpSize(),
                                          mlir::gpu::ShuffleMode::DOWN)
            .getShuffleResult();
      };

      auto shuffle_int_or_float = [&](Value value) {
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
        if (n_shuffles > 1) {
          // Don't generate vectors if the size is 1.
          auto vector_type = ml::getVectorType(b.getI32Type(), n_shuffles);
          value = b.create<ml::BitcastOp>(vector_type, value);
          Value result_vec = b.create<ml::UndefOp>(vector_type);
          for (int i = 0; i < n_shuffles; ++i) {
            auto idx = b.create<mlir::arith::ConstantIntOp>(i, 32);
            result_vec = b.create<ml::InsertElementOp>(
                result_vec,
                shuffle_32(b.create<ml::ExtractElementOp>(value, idx)), idx);
          }
          value = b.create<ml::BitcastOp>(padded_int_ty, result_vec);
        } else {
          value = shuffle_32(value);
        }
        value = b.create<mlir::arith::TruncIOp>(int_ty, value);
        value = b.create<ml::BitcastOp>(ty, value);
        return value;
      };

      auto shuffle = [&](Value value) -> Value {
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

      SmallVector<Value> args = values;
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

struct RewriteXlaGpuLoop : mlir::OpRewritePattern<LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      LoopOp op, mlir::PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    IndexingMap indexing_map = op.getIndexingMap();
    SmallVector<Value, 4> lbs, ubs, steps;
    mlir_converter::GetLoopBoundsFromIndexingMap(b, indexing_map, &lbs, &ubs,
                                                 &steps);
    mlir::scf::LoopNest loop_nest = mlir::scf::buildLoopNest(
        b, loc, lbs, ubs, steps, op.getInits(),
        [&](OpBuilder& nested_builder, Location loc, ValueRange symbol_values,
            ValueRange iter_args) -> mlir::scf::ValueVector {
          mlir::ImplicitLocOpBuilder nested_b(loc, nested_builder);
          auto is_in_bounds = mlir_converter::CheckConstraints(
              indexing_map, op.getDims(), symbol_values, nested_b);
          auto if_op = nested_b.create<mlir::scf::IfOp>(
              is_in_bounds,
              [&](OpBuilder& then_builder, Location then_loc) -> void {
                SmallVector<Value, 4> bb_args(symbol_values);
                bb_args.append(iter_args.begin(), iter_args.end());

                mlir::Block* then_block = then_builder.getInsertionBlock();
                OpBuilder::InsertionGuard guard(rewriter);
                rewriter.setInsertionPointToStart(then_block);
                rewriter.mergeBlocks(op.getBody(), then_block, bb_args);

                auto old_terminator = then_block->getTerminator();
                then_builder.create<mlir::scf::YieldOp>(
                    then_loc, old_terminator->getOperands());
                old_terminator->erase();
              },
              [&](OpBuilder& else_b, Location else_loc) {
                else_b.create<mlir::scf::YieldOp>(loc, iter_args);
              });
          return if_op.getResults();
        });
    rewriter.replaceOp(op, loop_nest.results);
    return mlir::success();
  }
};

class LowerXlaGpuToScfPass
    : public impl::LowerXlaGpuToScfPassBase<LowerXlaGpuToScfPass> {
 public:
  void runOnOperation() override {
    auto* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewritePredicatedInsert, RewritePredicatedExtract,
                 RewriteShuffleReduce>(ctx);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class LowerXlaGpuLoopsToScfPass
    : public impl::LowerXlaGpuLoopsToScfPassBase<LowerXlaGpuLoopsToScfPass> {
 public:
  void runOnOperation() override {
    auto* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewriteXlaGpuLoop>(ctx);
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

std::unique_ptr<::mlir::Pass> CreateLowerXlaGpuLoopsToScfPass() {
  return std::make_unique<LowerXlaGpuLoopsToScfPass>();
}

}  // namespace gpu
}  // namespace xla
