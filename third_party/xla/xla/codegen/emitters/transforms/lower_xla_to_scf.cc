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
#include <cstdint>
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/util.h"

namespace xla {
namespace emitters {
namespace {

#define GEN_PASS_DEF_LOWERXLATOSCFPASS
#define GEN_PASS_DEF_LOWERXLALOOPSTOSCFPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

using mlir::ImplicitLocOpBuilder;
using mlir::Location;
using mlir::OpBuilder;
using mlir::SmallVector;
using mlir::success;
using mlir::Value;
using mlir::ValueRange;
using mlir::scf::IfOp;

struct RewritePredicatedInsert : mlir::OpRewritePattern<PredicatedInsertOp> {
  RewritePredicatedInsert(mlir::MLIRContext* context,
                          const LowerXlaToScfPassOptions& options)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      PredicatedInsertOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(
        op, op.getCondition(),
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          mlir::scf::YieldOp::create(
              b, loc,
              mlir::tensor::InsertOp::create(b, loc, op.getValue(),
                                             op.getDest(), op.getIndices())
                  .getResult());
        },
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          mlir::scf::YieldOp::create(b, loc, op.getDest());
        });
    return success();
  }
};

struct RewritePredicatedExtract : mlir::OpRewritePattern<PredicatedExtractOp> {
  RewritePredicatedExtract(mlir::MLIRContext* context,
                           const LowerXlaToScfPassOptions& options)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      PredicatedExtractOp op, mlir::PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::scf::IfOp>(
        op, op.getCondition(),
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          mlir::scf::YieldOp::create(b, loc,
                                     mlir::tensor::ExtractOp::create(
                                         b, loc, op.getSrc(), op.getIndices())
                                         .getResult());
        },
        [&](mlir::OpBuilder& b, mlir::Location loc) {
          mlir::scf::YieldOp::create(b, loc, op.getFallback());
        });
    return success();
  }
};

struct RewriteShuffleReduce : mlir::OpRewritePattern<gpu::ShuffleReduceOp> {
  const int64_t warp_size;

  RewriteShuffleReduce(mlir::MLIRContext* context,
                       const LowerXlaToScfPassOptions& options)
      : OpRewritePattern(context), warp_size(options.warp_size) {}

  mlir::LogicalResult matchAndRewrite(
      gpu::ShuffleReduceOp op, mlir::PatternRewriter& rewriter) const override {
    int max_distance =
        mlir::cast<mlir::IntegerAttr>(op->getAttr("max_distance")).getInt();
    // TODO(jreiffers): Do this in a verifier.
    if (max_distance & (max_distance - 1) || max_distance >= warp_size) {
      return op->emitOpError("max_distance must be a power of 2 < warp_size_");
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ValueRange values = op.getOperands();
    for (int distance = max_distance; distance > 0; distance /= 2) {
      namespace ml = mlir::LLVM;
      auto shuffle_32 = [&](Value v) {
        return mlir::gpu::ShuffleOp::create(b, v, distance, warp_size,
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
        value = mlir::arith::BitcastOp::create(b, int_ty, value);
        value = mlir::arith::ExtUIOp::create(b, padded_int_ty, value);
        if (n_shuffles > 1) {
          // Don't generate vectors if the size is 1.
          auto vector_type = ml::getVectorType(b.getI32Type(), n_shuffles);
          value = ml::BitcastOp::create(b, vector_type, value);
          Value result_vec = ml::UndefOp::create(b, vector_type);
          for (int i = 0; i < n_shuffles; ++i) {
            auto idx = mlir::arith::ConstantIntOp::create(b, i, 32);
            result_vec = ml::InsertElementOp::create(
                b, result_vec,
                shuffle_32(ml::ExtractElementOp::create(b, value, idx)), idx);
          }
          value = ml::BitcastOp::create(b, padded_int_ty, result_vec);
        } else {
          value = shuffle_32(value);
        }
        value = mlir::arith::TruncIOp::create(b, int_ty, value);
        value = ml::BitcastOp::create(b, ty, value);
        return value;
      };

      auto shuffle = [&](Value value) -> Value {
        if (mlir::isa<mlir::ComplexType>(value.getType())) {
          return mlir::complex::CreateOp::create(
              b, value.getType(),
              shuffle_int_or_float(mlir::complex::ReOp::create(b, value)),
              shuffle_int_or_float(mlir::complex::ImOp::create(b, value)));
        }
        if (value.getType().isUnsignedInteger()) {
          auto ty = value.getType();
          auto signless_ty = b.getIntegerType(ty.getIntOrFloatBitWidth());
          value = mlir::UnrealizedConversionCastOp::create(
                      b, mlir::TypeRange{signless_ty}, value)
                      .getResult(0);
          value = shuffle_int_or_float(value);
          value = mlir::UnrealizedConversionCastOp::create(
                      b, mlir::TypeRange{ty}, value)
                      .getResult(0);
          return value;
        }
        return shuffle_int_or_float(value);
      };

      SmallVector<Value> args = values;
      for (auto value : values) {
        args.push_back(shuffle(value));
      }
      values = PureCallOp::create(b, op.getResultTypes(),
                                  op.getCombinerAttr().getAttr(), args)
                   .getResults();
    }
    rewriter.replaceOp(op, values);
    return success();
  }
};

struct RewriteXlaLoop : mlir::OpRewritePattern<LoopOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      LoopOp op, mlir::PatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    IndexingMap indexing_map = op.getIndexingMap();
    SmallVector<Value, 4> lbs, ubs, steps;
    emitters::GetLoopBoundsFromIndexingMap(b, indexing_map, &lbs, &ubs, &steps);
    mlir::scf::LoopNest loop_nest = mlir::scf::buildLoopNest(
        b, loc, lbs, ubs, steps, op.getInits(),
        [&](OpBuilder& nested_builder, Location loc, ValueRange symbol_values,
            ValueRange iter_args) -> mlir::scf::ValueVector {
          mlir::ImplicitLocOpBuilder nested_b(loc, nested_builder);
          auto is_in_bounds = emitters::CheckConstraints(
              indexing_map, op.getDims(), symbol_values, nested_b);
          auto if_op = mlir::scf::IfOp::create(
              nested_b, is_in_bounds,
              [&](OpBuilder& then_builder, Location then_loc) -> void {
                ImplicitLocOpBuilder then_b(then_loc, then_builder);
                mlir::IRMapping mapping;
                mapping.map(op.getInductionVars(), symbol_values);
                mapping.map(op.getIndexingMapResults(),
                            emitters::ApplyIndexing(indexing_map, op.getDims(),
                                                    symbol_values, then_b));
                mapping.map(op.getRegionIterArgs(), iter_args);
                mlir::Block* old_block = op.getBody();
                for (auto& old_op : old_block->without_terminator()) {
                  then_b.clone(old_op, mapping);
                }
                SmallVector<Value, 4> then_results;
                for (auto result : old_block->getTerminator()->getOperands()) {
                  then_results.push_back(mapping.lookupOrDefault(result));
                }
                mlir::scf::YieldOp::create(then_b, then_results);
              },
              [&](OpBuilder& else_b, Location else_loc) {
                mlir::scf::YieldOp::create(else_b, loc, iter_args);
              });
          return if_op.getResults();
        });
    rewriter.replaceOp(op, loop_nest.results);
    return mlir::success();
  }
};

mlir::VectorType getThreadLevelVectorType(
    gpu::IndexedVectorType indexed_vector) {
  auto data_type = indexed_vector.getElementType();
  SmallVector<int64_t> vector_dims;
  if (auto complex = mlir::dyn_cast<mlir::ComplexType>(data_type)) {
    vector_dims.push_back(2);
    data_type = complex.getElementType();
  }
  IndexingMap map = indexed_vector.getIndexingMapAttr().getIndexingMap();
  for (auto bound : map.GetSymbolBounds()) {
    vector_dims.push_back(bound.GetLoopTripCount());
  }
  return mlir::VectorType::get(vector_dims, data_type);
}

struct RewriteInsert : mlir::OpRewritePattern<gpu::InsertOp> {
  RewriteInsert(mlir::MLIRContext* context,
                const LowerXlaToScfPassOptions& options)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      gpu::InsertOp op, mlir::PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto i0 = mlir::arith::ConstantIndexOp::create(b, 0);
    auto i1 = mlir::arith::ConstantIndexOp::create(b, 1);
    auto convert = mlir::UnrealizedConversionCastOp::create(
                       b, getThreadLevelVectorType(op.getSource().getType()),
                       op.getSource())
                       .getResult(0);
    // InsertOp's map attribute (op.getMap()) is a mapping from
    //    indexed_vector index -> tensor index.
    // We get indexed_vector index by using its encoding map (source_map).
    // So we loop over indexed_vector encoding map and use the results as the
    // dimensions for InsertOp's map in order to get the final tensor index.
    auto source_map = op.getSource().getType().getIndexingMapAttr();
    auto loop = LoopOp::create(
        b, source_map, op.getIndices(), ValueRange{op.getDest()},
        [&](OpBuilder&, Location, ValueRange ivs, ValueRange map_results,
            ValueRange iter_args) {
          SmallVector<mlir::OpFoldResult> vector_offset(ivs);
          Value scalar;
          if (auto complex = mlir::dyn_cast<mlir::ComplexType>(
                  op.getSource().getType().getElementType())) {
            vector_offset.insert(vector_offset.begin(), i0.getResult());
            auto real =
                mlir::vector::ExtractOp::create(b, convert, vector_offset);
            vector_offset.front() = i1.getResult();
            auto imag =
                mlir::vector::ExtractOp::create(b, convert, vector_offset);
            scalar = mlir::complex::CreateOp::create(b, complex, real, imag)
                         .getResult();
          } else {
            scalar = mlir::vector::ExtractOp::create(b, convert, vector_offset)
                         .getResult();
          }
          auto tensor_indices = ApplyIndexingOp::create(
              b, map_results, ValueRange(), op.getMap().getIndexingMap());
          Value new_tensor = mlir::tensor::InsertOp::create(
              b, scalar, iter_args.back(), tensor_indices.getResults());
          YieldOp::create(b, new_tensor);
        });
    rewriter.replaceOp(op, loop->getResults());

    return success();
  }
};

class LowerXlaToScfPass
    : public impl::LowerXlaToScfPassBase<LowerXlaToScfPass> {
 public:
  explicit LowerXlaToScfPass(const LowerXlaToScfPassOptions& options)
      : options_(options) {}

  void runOnOperation() override {
    auto* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewritePredicatedInsert, RewritePredicatedExtract,
                 RewriteShuffleReduce, RewriteInsert>(ctx, options_);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }

 private:
  const LowerXlaToScfPassOptions options_;
};

class LowerXlaLoopsToScfPass
    : public impl::LowerXlaLoopsToScfPassBase<LowerXlaLoopsToScfPass> {
 public:
  void runOnOperation() override {
    auto* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewriteXlaLoop>(ctx);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<::mlir::Pass> CreateLowerXlaToScfPass(const int64_t warp_size) {
  LowerXlaToScfPassOptions options;
  options.warp_size = warp_size;
  return std::make_unique<LowerXlaToScfPass>(options);
}

std::unique_ptr<::mlir::Pass> CreateLowerXlaLoopsToScfPass() {
  return std::make_unique<LowerXlaLoopsToScfPass>();
}

}  // namespace emitters
}  // namespace xla
