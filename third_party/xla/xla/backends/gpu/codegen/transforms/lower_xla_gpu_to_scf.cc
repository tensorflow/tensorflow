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
#include "xla/backends/gpu/codegen/ir/xla_gpu_ops.h"
#include "xla/backends/gpu/codegen/transforms/passes.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

#define GEN_PASS_DEF_LOWERXLAGPUTOSCFPASS
#define GEN_PASS_DEF_LOWERXLAGPULOOPSTOSCFPASS
#include "xla/backends/gpu/codegen/transforms/passes.h.inc"

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
                          const LowerXlaGpuToScfPassOptions& options)
      : OpRewritePattern(context) {}

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
  RewritePredicatedExtract(mlir::MLIRContext* context,
                           const LowerXlaGpuToScfPassOptions& options)
      : OpRewritePattern(context) {}

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
  const int64_t warp_size;

  RewriteShuffleReduce(mlir::MLIRContext* context,
                       const LowerXlaGpuToScfPassOptions& options)
      : OpRewritePattern(context), warp_size(options.warp_size) {}

  mlir::LogicalResult matchAndRewrite(
      ShuffleReduceOp op, mlir::PatternRewriter& rewriter) const override {
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
        return b
            .create<mlir::gpu::ShuffleOp>(v, distance, warp_size,
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
                                    op.getCombinerAttr().getAttr(), args)
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
                ImplicitLocOpBuilder then_b(then_loc, then_builder);
                mlir::IRMapping mapping;
                mapping.map(op.getInductionVars(), symbol_values);
                mapping.map(
                    op.getIndexingMapResults(),
                    mlir_converter::ApplyIndexing(indexing_map, op.getDims(),
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
                then_b.create<mlir::scf::YieldOp>(then_results);
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

mlir::VectorType getThreadLevelVectorType(IndexedVectorType indexed_vector) {
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

struct RewriteMaterialize : mlir::OpRewritePattern<MaterializeOp> {
  RewriteMaterialize(mlir::MLIRContext* context,
                     const LowerXlaGpuToScfPassOptions& options)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      MaterializeOp op, mlir::PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto i0 = b.create<mlir::arith::ConstantIndexOp>(0);
    auto i1 = b.create<mlir::arith::ConstantIndexOp>(1);

    auto vec_type = getThreadLevelVectorType(op.getResult().getType());
    auto maybe_complex_data_type = op.getResult().getType().getElementType();
    auto data_type = vec_type.getElementType();
    Value init_vec;
    if (mlir::isa<mlir::IntegerType>(data_type)) {
      init_vec = b.create<mlir::arith::ConstantOp>(mlir::DenseElementsAttr::get(
          vec_type, b.getIntegerAttr(data_type, 0)));
    } else if (mlir::isa<mlir::FloatType>(data_type)) {
      init_vec = b.create<mlir::arith::ConstantOp>(
          mlir::DenseElementsAttr::get(vec_type, b.getFloatAttr(data_type, 0)));
    } else {
      return op->emitOpError("invalid data type");
    }

    auto loop = b.create<LoopOp>(
        op.getMapAttr(), op.getIndices(), ValueRange{init_vec},
        [&](OpBuilder&, Location, ValueRange ivs, ValueRange map_results,
            ValueRange iter_args) {
          auto args = SmallVector<Value, 4>(op.getInput());
          args.insert(args.end(), map_results.begin(), map_results.end());
          SmallVector<mlir::Type, 1> types{maybe_complex_data_type};
          auto call_result =
              b.create<PureCallOp>(op.getCalleeAttr(), ValueRange{args}, types)
                  .getResult(0);
          SmallVector<mlir::OpFoldResult> offset(ivs);
          auto old_vec = iter_args.back();
          Value new_vec;
          if (mlir::isa<mlir::ComplexType>(call_result.getType())) {
            auto real = b.create<mlir::complex::ReOp>(call_result);
            auto imag = b.create<mlir::complex::ImOp>(call_result);
            offset.insert(offset.begin(), i0.getResult());
            new_vec = b.create<mlir::vector::InsertOp>(real, old_vec, offset);
            offset.front() = i1.getResult();
            new_vec = b.create<mlir::vector::InsertOp>(imag, new_vec, offset);
          } else {
            new_vec =
                b.create<mlir::vector::InsertOp>(call_result, old_vec, offset);
          }
          b.create<YieldOp>(new_vec);
        });
    auto convert = b.create<mlir::UnrealizedConversionCastOp>(
                        op.getResult().getType(), loop->getResults())
                       .getResult(0);
    rewriter.replaceOp(op, convert);
    return success();
  }
};

struct RewriteInsert : mlir::OpRewritePattern<InsertOp> {
  RewriteInsert(mlir::MLIRContext* context,
                const LowerXlaGpuToScfPassOptions& options)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      InsertOp op, mlir::PatternRewriter& rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto i0 = b.create<mlir::arith::ConstantIndexOp>(0);
    auto i1 = b.create<mlir::arith::ConstantIndexOp>(1);
    auto convert =
        b.create<mlir::UnrealizedConversionCastOp>(
             getThreadLevelVectorType(op.getSource().getType()), op.getSource())
            .getResult(0);
    // InsertOp's map attribute (op.getMap()) is a mapping from
    //    indexed_vector index -> tensor index.
    // We get indexed_vector index by using its encoding map (source_map).
    // So we loop over indexed_vector encoding map and use the results as the
    // dimensions for InsertOp's map in order to get the final tensor index.
    auto source_map = op.getSource().getType().getIndexingMapAttr();
    auto loop = b.create<LoopOp>(
        source_map, op.getIndices(), ValueRange{op.getDest()},
        [&](OpBuilder&, Location, ValueRange ivs, ValueRange map_results,
            ValueRange iter_args) {
          SmallVector<mlir::OpFoldResult> vector_offset(ivs);
          Value scalar;
          if (auto complex = mlir::dyn_cast<mlir::ComplexType>(
                  op.getSource().getType().getElementType())) {
            vector_offset.insert(vector_offset.begin(), i0.getResult());
            auto real =
                b.create<mlir::vector::ExtractOp>(convert, vector_offset);
            vector_offset.front() = i1.getResult();
            auto imag =
                b.create<mlir::vector::ExtractOp>(convert, vector_offset);
            scalar = b.create<mlir::complex::CreateOp>(complex, real, imag)
                         .getResult();
          } else {
            scalar = b.create<mlir::vector::ExtractOp>(convert, vector_offset)
                         .getResult();
          }
          auto tensor_indices = b.create<ApplyIndexingOp>(
              map_results, ValueRange(), op.getMap().getIndexingMap());
          Value new_tensor = b.create<mlir::tensor::InsertOp>(
              scalar, iter_args.back(), tensor_indices.getResults());
          b.create<YieldOp>(new_tensor);
        });
    rewriter.replaceOp(op, loop->getResults());

    return success();
  }
};

class LowerXlaGpuToScfPass
    : public impl::LowerXlaGpuToScfPassBase<LowerXlaGpuToScfPass> {
 public:
  explicit LowerXlaGpuToScfPass(const LowerXlaGpuToScfPassOptions& options)
      : options_(options) {}

  void runOnOperation() override {
    auto* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);
    patterns.add<RewritePredicatedInsert, RewritePredicatedExtract,
                 RewriteShuffleReduce, RewriteMaterialize, RewriteInsert>(
        ctx, options_);
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }

 private:
  const LowerXlaGpuToScfPassOptions options_;
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

std::unique_ptr<::mlir::Pass> CreateLowerXlaGpuToScfPass(
    const int64_t warp_size) {
  LowerXlaGpuToScfPassOptions options;
  options.warp_size = warp_size;
  return std::make_unique<LowerXlaGpuToScfPass>(options);
}

std::unique_ptr<::mlir::Pass> CreateLowerXlaGpuLoopsToScfPass() {
  return std::make_unique<LowerXlaGpuLoopsToScfPass>();
}

}  // namespace gpu
}  // namespace xla
