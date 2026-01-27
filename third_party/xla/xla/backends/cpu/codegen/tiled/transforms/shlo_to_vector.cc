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
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/cpu/codegen/tiled/transforms/lowering_utils.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"
#include "xla/backends/cpu/codegen/tiled/transforms/vectorized_reduce_emitter.h"

namespace xla::cpu {

#define GEN_PASS_DECL_SHLOTOVECTORPASS
#define GEN_PASS_DEF_SHLOTOVECTORPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

mlir::AffineMapAttr GetOperandIndexingMap(
    mlir::OpBuilder& builder, int64_t iterator_count, int64_t rank,
    llvm::ArrayRef<int64_t> batch_dims,
    llvm::ArrayRef<int64_t> contracting_dims, int64_t free_dim_offset) {
  llvm::SmallVector<unsigned> targets(rank, -1);
  unsigned idx = 0;
  for (int64_t dim : batch_dims) {
    targets[dim] = idx++;
  }
  for (int64_t dim : contracting_dims) {
    targets[dim] = idx++;
  }
  for (unsigned& target : targets) {
    if (target == -1) {
      target = free_dim_offset + idx++;
    }
  }
  auto affine_map = mlir::AffineMap::getMultiDimMapWithTargets(
      iterator_count, targets, builder.getContext());

  return mlir::AffineMapAttr::get(affine_map);
}

mlir::AffineMapAttr GetOutputIndexingMap(mlir::OpBuilder& builder,
                                         int64_t iterator_count,
                                         int64_t batch_dim_count,
                                         int64_t contracting_dim_count) {
  llvm::SmallVector<unsigned> targets(iterator_count - contracting_dim_count);
  unsigned idx = 0;
  for (int64_t dim = 0; dim != batch_dim_count; ++dim) {
    targets[dim] = idx++;
  }
  idx += contracting_dim_count;
  int64_t total_free_dims =
      iterator_count - batch_dim_count - contracting_dim_count;
  for (int64_t dim = 0; dim != total_free_dims; ++dim) {
    targets[batch_dim_count + dim] = idx++;
  }
  auto affine_map = mlir::AffineMap::getMultiDimMapWithTargets(
      iterator_count, targets, builder.getContext());

  return mlir::AffineMapAttr::get(affine_map);
}

mlir::ArrayAttr GetIteratorTypes(mlir::OpBuilder& builder,
                                 int64_t iterator_count,
                                 int64_t batch_dim_count,
                                 int64_t contracting_dim_count) {
  llvm::SmallVector<mlir::Attribute> iterator_types;
  iterator_types.reserve(iterator_count);
  for (int64_t dim = 0; dim != batch_dim_count; ++dim) {
    iterator_types.push_back(builder.getAttr<mlir::vector::IteratorTypeAttr>(
        mlir::vector::IteratorType::parallel));
  }
  for (int64_t dim = 0; dim != contracting_dim_count; ++dim) {
    iterator_types.push_back(builder.getAttr<mlir::vector::IteratorTypeAttr>(
        mlir::vector::IteratorType::reduction));
  }
  int64_t free_dims = iterator_count - batch_dim_count - contracting_dim_count;
  for (int64_t dim = 0; dim != free_dims; ++dim) {
    iterator_types.push_back(builder.getAttr<mlir::vector::IteratorTypeAttr>(
        mlir::vector::IteratorType::parallel));
  }

  return mlir::ArrayAttr::get(builder.getContext(), iterator_types);
}

// Lowers from stablehlo.dot_general to vector.contract.
// The vector contract is very general as described here:
// https://mlir.llvm.org/docs/Dialects/Vector/#vectorcontract-vectorcontractionop
// In this lowering the iteration order attribute passed is of the form:
// (batch..., contracting..., free_lhs..., free_rhs...)
// TODO(willfroom): Check if there is any performance impact on the order.
struct LowerDotGeneral : mlir::OpRewritePattern<mlir::stablehlo::DotGeneralOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::DotGeneralOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto lhs_vector = ReadTensorToVector(rewriter, op.getLhs());
    auto lhs_rank = lhs_vector.getType().getRank();

    auto rhs_vector = ReadTensorToVector(rewriter, op.getRhs());
    auto rhs_rank = rhs_vector.getType().getRank();

    // TODO(willfroom): Ensure this is being folded into the accumulator in the
    // dot loop.
    mlir::Value accumulator =
        GetAccumulator(rewriter, op->getLoc(), op.getType());

    mlir::stablehlo::DotDimensionNumbersAttr dimension_numbers =
        op.getDotDimensionNumbers();

    llvm::ArrayRef<int64_t> lhs_batch =
        dimension_numbers.getLhsBatchingDimensions();
    llvm::ArrayRef<int64_t> lhs_contracting =
        dimension_numbers.getLhsContractingDimensions();

    llvm::ArrayRef<int64_t> rhs_batch =
        dimension_numbers.getRhsBatchingDimensions();
    llvm::ArrayRef<int64_t> rhs_contracting =
        dimension_numbers.getRhsContractingDimensions();

    int64_t lhs_free_dims =
        lhs_rank - lhs_batch.size() - lhs_contracting.size();
    int64_t rhs_free_dims =
        rhs_rank - rhs_batch.size() - rhs_contracting.size();
    int64_t iterator_count = lhs_batch.size() + lhs_contracting.size() +
                             lhs_free_dims + rhs_free_dims;

    mlir::Attribute lhs_indexing_map = GetOperandIndexingMap(
        rewriter, iterator_count, lhs_rank, lhs_batch, lhs_contracting, 0);
    mlir::Attribute rhs_indexing_map =
        GetOperandIndexingMap(rewriter, iterator_count, rhs_rank, rhs_batch,
                              rhs_contracting, lhs_free_dims);
    mlir::Attribute output_indexing_map = GetOutputIndexingMap(
        rewriter, iterator_count, lhs_batch.size(), lhs_contracting.size());

    mlir::ArrayAttr indexing_maps = rewriter.getArrayAttr(
        {lhs_indexing_map, rhs_indexing_map, output_indexing_map});
    mlir::ArrayAttr iterator_types = GetIteratorTypes(
        rewriter, iterator_count, lhs_batch.size(), lhs_contracting.size());

    mlir::Value result = mlir::vector::ContractionOp::create(
        rewriter, op->getLoc(), lhs_vector, rhs_vector, accumulator,
        indexing_maps, iterator_types);

    rewriter.replaceOp(op, WriteVectorToTensor(rewriter, result));

    return mlir::success();
  }

 private:
  mlir::Value GetAccumulator(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::RankedTensorType result_type) const {
    mlir::Type element_type = result_type.getElementType();
    auto zero_const = mlir::arith::ConstantOp::create(
        builder, loc, element_type, builder.getZeroAttr(element_type));

    if (result_type.getRank() == 0) {
      return zero_const;
    }

    auto result_vector_type = GetVectorType(result_type);
    return mlir::vector::BroadcastOp::create(builder, loc, result_vector_type,
                                             zero_const);
  }
};

struct LowerTranspose : mlir::OpRewritePattern<mlir::stablehlo::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::TransposeOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value source_vector = ReadTensorToVector(rewriter, op.getOperand());

    mlir::TypedValue<mlir::VectorType> dest_vector =
        mlir::vector::TransposeOp::create(rewriter, op->getLoc(), source_vector,
                                          op.getPermutation());

    mlir::Value dest_tensor = WriteVectorToTensor(rewriter, dest_vector);

    rewriter.replaceAllUsesWith(op, dest_tensor);
    return mlir::success();
  }
};

struct LowerReduce : mlir::OpRewritePattern<mlir::stablehlo::ReduceOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReduceOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (op.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "reduce op with multiple results is not supported");
    }

    auto source_tensor = mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(
        op.getInputs().front());
    mlir::Value result_tensor = op.getResult(0);
    auto result_type =
        mlir::cast<mlir::RankedTensorType>(result_tensor.getType());

    mlir::Value init_value = mlir::tensor::ExtractOp::create(
        rewriter, op->getLoc(), result_type.getElementType(),
        op.getInitValues().front());

    // Ensure the reduction dimensions are sorted so we can easily check if the
    // minor dimension is reduced.
    llvm::SmallVector<int64_t> reduction_dims(op.getDimensions());
    absl::c_sort(reduction_dims);

    mlir::Value reduced_vector = EmitVectorizedReduction(
        rewriter, op->getLoc(), result_type, source_tensor, init_value,
        reduction_dims, op.getBody().front());

    rewriter.replaceOp(op, reduced_vector);

    return mlir::success();
  }
};

struct LowerBroadcastInDim
    : mlir::OpRewritePattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::BroadcastInDimOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto source_vector = ReadTensorToVector(rewriter, op.getOperand());
    auto result_vector_type = GetVectorType(op.getType());

    llvm::ArrayRef<int64_t> source_shape = source_vector.getType().getShape();
    llvm::ArrayRef<int64_t> broadcast_dims = op.getBroadcastDimensions();

    // First create an intermediate vector with the rank of the result vector
    // but with the broadcasted dimensions set to the source shape with all
    // additional dimensions set to 1.
    llvm::SmallVector<int64_t> intermediate_shape(result_vector_type.getRank(),
                                                  1);
    for (auto [input_dim, result_dim] : llvm::enumerate(broadcast_dims)) {
      intermediate_shape[result_dim] = source_shape[input_dim];
    }
    mlir::Value intermediate_vector = mlir::vector::ShapeCastOp::create(
        rewriter, op->getLoc(),
        mlir::VectorType::get(intermediate_shape,
                              result_vector_type.getElementType()),
        source_vector);
    // Now that all the inserted dimensions are size 1 we can legally call
    // broadcast even if they are not the most major dimensions.
    mlir::Value broadcast_op = mlir::vector::BroadcastOp::create(
        rewriter, op->getLoc(), result_vector_type, intermediate_vector);

    rewriter.replaceOp(op, WriteVectorToTensor(rewriter, broadcast_op));
    return mlir::success();
  }
};

struct LowerIota : mlir::OpRewritePattern<mlir::stablehlo::IotaOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::IotaOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (op.getType().getRank() != 1) {
      return rewriter.notifyMatchFailure(
          op, "iota op with rank != 1 is not supported");
    }

    auto result_type = op.getType();
    auto element_type = result_type.getElementType();
    int64_t iota_size = result_type.getNumElements();

    llvm::SmallVector<mlir::Attribute> iota_values(iota_size);
    for (int idx = 0; idx != iota_size; ++idx) {
      iota_values[idx] = rewriter.getIntegerAttr(element_type, idx);
    }

    mlir::Value iota_const = mlir::arith::ConstantOp::create(
        rewriter, op->getLoc(),
        mlir::DenseElementsAttr::get(result_type, iota_values));

    rewriter.replaceOp(op, iota_const);
    return mlir::success();
  }
};

class ShloToVectorPass : public impl::ShloToVectorPassBase<ShloToVectorPass> {
 public:
  using ShloToVectorPassBase::ShloToVectorPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerDotGeneral, LowerReduce, LowerBroadcastInDim, LowerIota>(
        context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateShloToVectorPass() {
  return std::make_unique<ShloToVectorPass>();
}

}  // namespace xla::cpu
