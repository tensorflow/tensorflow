/* Copyright 2026 The OpenXLA Authors.

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
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/backends/cpu/codegen/tiled/transforms/lowering_utils.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"

namespace xla::cpu {

#define GEN_PASS_DEF_VECTORIZEXTILEPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;
namespace mm = ::mlir::math;
namespace ms = ::mlir::scf;
namespace mv = ::mlir::vector;
namespace shlo = ::mlir::stablehlo;

using ::mlir::Value;
using ::mlir::ValueRange;

xla::SymbolicMap GetBoundsCheckSymbolicMap(mlir::MLIRContext* ctx,
                                           llvm::ArrayRef<int64_t> memref_shape,
                                           llvm::ArrayRef<int64_t> tile_shape) {
  int64_t rank = memref_shape.size();
  llvm::SmallVector<SymbolicExpr> results;
  results.reserve(rank);
  for (auto [index, memref_and_tile_size] :
       llvm::enumerate(llvm::zip(memref_shape, tile_shape))) {
    auto [memref_dim, tile_dim] = memref_and_tile_size;
    results.push_back(-xla::CreateDimExpr(index, ctx) + memref_dim - tile_dim);
  }
  return xla::SymbolicMap::Get(ctx, rank, 0, results);
}

Value GetIsInBoundsCondition(mlir::OpBuilder& builder, mlir::Location loc,
                             ValueRange offsets, Value memref,
                             llvm::ArrayRef<int64_t> tile_shape) {
  auto memref_shape = mlir::cast<mlir::MemRefType>(memref.getType()).getShape();
  int rank = memref_shape.size();

  xla::SymbolicMap symbolic_map =
      GetBoundsCheckSymbolicMap(builder.getContext(), memref_shape, tile_shape);

  std::vector<xla::IndexingMap::Variable> vars(
      rank, xla::IndexingMap::Variable{std::numeric_limits<int64_t>::min(),
                                       std::numeric_limits<int64_t>::max()});

  xla::IndexingMap indexing_map(symbolic_map,
                                /*dimensions=*/std::move(vars),
                                /*range_vars=*/{}, /*rt_vars=*/{});

  auto apply_indexing =
      xla::ApplyIndexingOp::create(builder, loc, offsets, indexing_map);

  Value is_in_bounds = nullptr;
  Value zero = ma::ConstantIndexOp::create(builder, loc, 0);
  for (Value val : apply_indexing.getResults()) {
    Value cmp =
        ma::CmpIOp::create(builder, loc, ma::CmpIPredicate::sge, val, zero);
    is_in_bounds = is_in_bounds != nullptr
                       ? ma::AndIOp::create(builder, loc, is_in_bounds, cmp)
                       : cmp;
  }
  return is_in_bounds;
}

Value GetMask(mlir::OpBuilder& builder, mlir::Location loc, ValueRange offsets,
              Value memref, mlir::VectorType vector_type) {
  auto memref_shape = mlir::cast<mlir::MemRefType>(memref.getType()).getShape();
  auto tile_shape = vector_type.getShape();
  int64_t rank = memref_shape.size();

  llvm::SmallVector<Value> mask_dims;
  mask_dims.reserve(rank);
  Value zero = ma::ConstantIndexOp::create(builder, loc, 0);
  for (auto [memref_dim, tile_dim, offset] :
       llvm::zip(memref_shape, tile_shape, offsets)) {
    Value tile_size = ma::ConstantIndexOp::create(builder, loc, tile_dim);
    Value dim_size = ma::ConstantIndexOp::create(builder, loc, memref_dim);
    Value rem = ma::SubIOp::create(builder, loc, dim_size, offset);
    Value min_val = ma::MinSIOp::create(builder, loc, tile_size, rem);
    Value max_val = ma::MaxSIOp::create(builder, loc, zero, min_val);
    mask_dims.push_back(max_val);
  }
  auto mask_type =
      mlir::VectorType::get(vector_type.getShape(), builder.getI1Type());
  return mv::CreateMaskOp::create(builder, loc, mask_type, mask_dims);
}

mlir::ArrayAttr GetInBoundsAttr(mlir::OpBuilder& builder, int64_t rank,
                                bool in_bounds) {
  return builder.getBoolArrayAttr(llvm::SmallVector<bool>(rank, in_bounds));
}

struct ConvertExtractTile
    : public mlir::OpConversionPattern<xtile::ExtractTileOp> {
  using mlir::OpConversionPattern<xtile::ExtractTileOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::ExtractTileOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::VectorType result_vector_type = mlir::cast<mlir::VectorType>(
        this->getTypeConverter()->convertType(op.getType()));

    ValueRange offsets = adaptor.getOffsets();
    Value source_memref = adaptor.getSource();

    Value pad = ma::ConstantOp::create(
        rewriter, loc, result_vector_type.getElementType(),
        rewriter.getZeroAttr(result_vector_type.getElementType()));

    if (result_vector_type.getRank() == 0) {
      mlir::AffineMap permutation_map =
          mlir::vector::getTransferMinorIdentityMap(
              mlir::cast<mlir::ShapedType>(source_memref.getType()),
              result_vector_type);
      mlir::AffineMapAttr permutation_map_attr =
          mlir::AffineMapAttr::get(permutation_map);
      mlir::ArrayAttr in_bounds_attr =
          GetInBoundsAttr(rewriter, result_vector_type.getRank(), true);
      rewriter.replaceOpWithNewOp<mv::TransferReadOp>(
          op, result_vector_type, source_memref, offsets, permutation_map_attr,
          pad, /*mask=*/Value(), in_bounds_attr);
      return mlir::success();
    }

    // Check if in bounds
    Value is_in_bounds = GetIsInBoundsCondition(
        rewriter, loc, offsets, source_memref, result_vector_type.getShape());

    // Generate scf.if
    ms::IfOp if_op = ms::IfOp::create(rewriter, loc, result_vector_type,
                                      is_in_bounds, /*withElseRegion=*/true);

    // In-bounds branch (Then)
    rewriter.setInsertionPointToStart(if_op.thenBlock());

    mlir::AffineMap permutation_map = mlir::vector::getTransferMinorIdentityMap(
        mlir::cast<mlir::ShapedType>(source_memref.getType()),
        result_vector_type);
    mlir::AffineMapAttr permutation_map_attr =
        mlir::AffineMapAttr::get(permutation_map);
    mlir::ArrayAttr in_bounds_attr =
        GetInBoundsAttr(rewriter, result_vector_type.getRank(), true);

    Value in_bounds_read = mv::TransferReadOp::create(
        rewriter, loc, result_vector_type, source_memref, offsets,
        permutation_map_attr, pad, /*mask=*/Value(), in_bounds_attr);
    ms::YieldOp::create(rewriter, loc, in_bounds_read);

    // Out-of-bounds branch (Else)
    rewriter.setInsertionPointToStart(if_op.elseBlock());

    Value mask =
        GetMask(rewriter, loc, offsets, source_memref, result_vector_type);

    mlir::ArrayAttr out_of_bounds_attr =
        GetInBoundsAttr(rewriter, result_vector_type.getRank(), false);

    Value masked_read = mlir::vector::TransferReadOp::create(
        rewriter, loc, result_vector_type, source_memref, offsets,
        permutation_map_attr, pad, mask, out_of_bounds_attr);
    ms::YieldOp::create(rewriter, loc, masked_read);

    rewriter.replaceOp(op, if_op.getResult(0));
    return mlir::success();
  }
};

struct ConvertInsertTile
    : public mlir::OpConversionPattern<xtile::InsertTileOp> {
  using mlir::OpConversionPattern<xtile::InsertTileOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::InsertTileOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Location loc = op.getLoc();
    Value source_vector = adaptor.getSource();
    mlir::VectorType source_vector_type =
        mlir::cast<mlir::VectorType>(source_vector.getType());

    ValueRange offsets = adaptor.getOffsets();
    Value dest_memref = adaptor.getDestination();

    if (source_vector_type.getRank() == 0) {
      mlir::AffineMap permutation_map = mv::getTransferMinorIdentityMap(
          mlir::cast<mlir::ShapedType>(dest_memref.getType()),
          source_vector_type);
      auto permutation_map_attr = mlir::AffineMapAttr::get(permutation_map);
      mlir::ArrayAttr in_bounds_attr =
          GetInBoundsAttr(rewriter, source_vector_type.getRank(), true);

      rewriter.replaceOpWithNewOp<mv::TransferWriteOp>(
          op, source_vector, dest_memref, offsets, permutation_map_attr,
          /*mask=*/Value(), in_bounds_attr);
      return mlir::success();
    }

    // Check if in bounds
    Value is_in_bounds = GetIsInBoundsCondition(
        rewriter, loc, offsets, dest_memref, source_vector_type.getShape());

    // Generate scf.if (no results)
    ms::IfOp if_op = ms::IfOp::create(rewriter, loc, is_in_bounds,
                                      /*withElseRegion=*/true);

    // In-bounds branch (Then)
    rewriter.setInsertionPointToStart(if_op.thenBlock());

    mlir::AffineMap permutation_map = mv::getTransferMinorIdentityMap(
        mlir::cast<mlir::ShapedType>(dest_memref.getType()),
        source_vector_type);
    auto permutation_map_attr = mlir::AffineMapAttr::get(permutation_map);
    mlir::ArrayAttr in_bounds_attr =
        GetInBoundsAttr(rewriter, source_vector_type.getRank(), true);

    mv::TransferWriteOp::create(rewriter, loc, source_vector, dest_memref,
                                offsets, permutation_map_attr, /*mask=*/Value(),
                                in_bounds_attr);

    // Out-of-bounds branch (Else)
    rewriter.setInsertionPointToStart(if_op.elseBlock());

    Value mask = GetMask(rewriter, op.getLoc(), offsets, dest_memref,
                         source_vector_type);

    mlir::ArrayAttr out_of_bounds_attr =
        GetInBoundsAttr(rewriter, source_vector_type.getRank(), false);

    mlir::vector::TransferWriteOp::create(
        rewriter, op.getLoc(), source_vector, dest_memref, offsets,
        permutation_map_attr, mask, out_of_bounds_attr);

    rewriter.eraseOp(op);
    return mlir::success();
  }
};

struct VectorizeMaskOp : public mlir::OpConversionPattern<xtile::MaskOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      xtile::MaskOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type new_type = getTypeConverter()->convertType(op.getType());
    if (!new_type) {
      return mlir::failure();
    }
    auto vector_type = mlir::cast<mlir::VectorType>(new_type);
    if (vector_type.getRank() == 0) {
      rewriter.replaceOp(op, adaptor.getSource());
      return mlir::success();
    }

    mlir::Location loc = op.getLoc();
    llvm::SmallVector<Value> mask_dims;
    mask_dims.reserve(vector_type.getRank());
    for (int64_t bound : op.getBounds()) {
      mask_dims.push_back(ma::ConstantIndexOp::create(rewriter, loc, bound));
    }

    auto mask_type =
        mlir::VectorType::get(vector_type.getShape(), rewriter.getI1Type());
    Value mask = mv::CreateMaskOp::create(rewriter, loc, mask_type, mask_dims);

    Value passthrough =
        mv::BroadcastOp::create(rewriter, loc, vector_type, adaptor.getValue());

    rewriter.replaceOpWithNewOp<ma::SelectOp>(op, mask, adaptor.getSource(),
                                              passthrough);
    return mlir::success();
  }
};

struct VectorizeBroadcastInDimOp
    : public mlir::OpConversionPattern<shlo::BroadcastInDimOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      shlo::BroadcastInDimOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type new_type = this->getTypeConverter()->convertType(op.getType());
    if (!new_type) {
      return mlir::failure();
    }
    auto result_vector_type = mlir::cast<mlir::VectorType>(new_type);
    // When broadcasting a tensor.from_elements(scalar), we can directly
    // broadcast the source scalar.
    if (auto from_elements =
            op.getOperand().getDefiningOp<mlir::tensor::FromElementsOp>()) {
      if (from_elements.getElements().size() == 1) {
        rewriter.replaceOpWithNewOp<mv::BroadcastOp>(
            op, result_vector_type, from_elements.getElements().front());
        return mlir::success();
      }
    }
    Value source_vector = adaptor.getOperand();
    auto source_vector_type =
        mlir::cast<mlir::VectorType>(source_vector.getType());

    llvm::ArrayRef<int64_t> source_shape = source_vector_type.getShape();
    llvm::ArrayRef<int64_t> broadcast_dims = op.getBroadcastDimensions();

    llvm::SmallVector<int64_t> intermediate_shape(result_vector_type.getRank(),
                                                  1);
    for (auto [input_dim, result_dim] : llvm::enumerate(broadcast_dims)) {
      intermediate_shape[result_dim] = source_shape[input_dim];
    }

    auto intermediate_vector_type = mlir::VectorType::get(
        intermediate_shape, result_vector_type.getElementType());

    mlir::Value intermediate_vector = mlir::vector::ShapeCastOp::create(
        rewriter, op->getLoc(), intermediate_vector_type, source_vector);

    rewriter.replaceOpWithNewOp<mv::BroadcastOp>(op, result_vector_type,
                                                 intermediate_vector);
    return mlir::success();
  }
};

struct VectorizeDotGeneralOp
    : public mlir::OpConversionPattern<shlo::DotGeneralOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      shlo::DotGeneralOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (mlir::isa<mlir::ComplexType>(op.getType().getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "complex types are not supported by vector operations");
    }

    mlir::Operation* add_op;
    Value accumulator;
    if (mlir::failed(GetFusedAddUnit(op, rewriter, add_op, accumulator))) {
      return mlir::failure();
    }

    auto lhs_vector =
        mlir::dyn_cast<mlir::TypedValue<mlir::VectorType>>(adaptor.getLhs());
    auto rhs_vector =
        mlir::dyn_cast<mlir::TypedValue<mlir::VectorType>>(adaptor.getRhs());
    if (!lhs_vector || !rhs_vector) {
      return mlir::failure();
    }

    int64_t lhs_rank = lhs_vector.getType().getRank();
    int64_t rhs_rank = rhs_vector.getType().getRank();

    mlir::VectorType result_vector_type = mlir::cast<mlir::VectorType>(
        this->getTypeConverter()->convertType(op.getType()));

    Value acc_vector = accumulator;
    if (acc_vector.getType() != result_vector_type) {
      acc_vector = this->getTypeConverter()->materializeTargetConversion(
          rewriter, op.getLoc(), result_vector_type, acc_vector);
    }

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

    rewriter.setInsertionPoint(add_op);
    mlir::Value result = mlir::vector::ContractionOp::create(
        rewriter, op->getLoc(), lhs_vector, rhs_vector, acc_vector,
        indexing_maps, iterator_types);

    rewriter.replaceOp(add_op, result);
    rewriter.eraseOp(op);

    return mlir::success();
  }
};

absl::StatusOr<mv::CombiningKind> GetCombiningKind(
    mlir::Block& reduction_body) {
  mlir::Operation* terminator = reduction_body.getTerminator();
  if (!terminator || terminator->getNumOperands() == 0) {
    return absl::InternalError("No reduction combiner");
  }
  mlir::Operation* op = terminator->getOperand(0).getDefiningOp();
  if (!op) {
    return absl::InternalError("No reduction combiner");
  }
  for (mlir::Value operand : op->getOperands()) {
    if (operand.getDefiningOp()) {
      return absl::InternalError("Non trivial reduction combiner");
    }
  }
  if (auto kind = mlir::linalg::getCombinerOpKind(op)) {
    return *kind;
  }
  return absl::InternalError("Unsupported reduction combiner");
}

struct VectorizeReduceOp : public mlir::OpConversionPattern<shlo::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      shlo::ReduceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (op.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "reduce op with multiple results is not supported");
    }
    auto input_vector = adaptor.getInputs().front();
    auto input_vector_type =
        mlir::cast<mlir::VectorType>(input_vector.getType());
    auto kind = GetCombiningKind(op.getBody().front());
    if (!kind.ok()) {
      return rewriter.notifyMatchFailure(op, kind.status().message());
    }

    mlir::Location loc = op.getLoc();
    mlir::Value init_value =
        mv::ExtractOp::create(rewriter, loc, adaptor.getInitValues().front());
    mlir::Type dest_type =
        this->getTypeConverter()->convertType(op.getResultTypes().front());
    auto dest_vector_type = mlir::dyn_cast<mlir::VectorType>(dest_type);
    if (!dest_type) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }

    llvm::SmallVector<bool> reduction_mask(input_vector_type.getRank(), false);
    for (int64_t dim : op.getDimensions()) {
      reduction_mask[dim] = true;
    }

    bool reduce_all_dims =
        op.getDimensions().size() == input_vector_type.getRank();
    mlir::Value acc = reduce_all_dims
                          ? init_value
                          : mv::BroadcastOp::create(
                                rewriter, loc, dest_vector_type, init_value);
    mlir::Value reduction = mv::MultiDimReductionOp::create(
        rewriter, loc, input_vector, acc, reduction_mask, *kind);

    mlir::Value result = reduction;
    if (result.getType() != dest_type) {
      result = mv::BroadcastOp::create(rewriter, loc, dest_type, result);
    }
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

struct VectorizeFromElementsOp
    : public mlir::OpConversionPattern<mlir::tensor::FromElementsOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::FromElementsOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type new_type = getTypeConverter()->convertType(op.getType());
    if (!new_type) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mv::FromElementsOp>(op, new_type,
                                                    adaptor.getElements());
    return mlir::success();
  }
};

struct VectorizeExtractOp
    : public mlir::OpConversionPattern<mlir::tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type new_type = getTypeConverter()->convertType(op.getType());
    if (!new_type) {
      return mlir::failure();
    }
    llvm::SmallVector<int64_t> static_indices;
    for (mlir::Value idx : adaptor.getIndices()) {
      auto const_idx = idx.getDefiningOp<ma::ConstantIndexOp>();
      if (!const_idx) {
        return rewriter.notifyMatchFailure(op, "non-constant extract index");
      }
      static_indices.push_back(const_idx.value());
    }
    rewriter.replaceOpWithNewOp<mv::ExtractOp>(op, adaptor.getTensor(),
                                               static_indices);
    return mlir::success();
  }
};

struct VectorizeTransposeOp
    : public mlir::OpConversionPattern<shlo::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      shlo::TransposeOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type new_type = this->getTypeConverter()->convertType(op.getType());
    if (!new_type) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mv::TransposeOp>(
        op, new_type, adaptor.getOperand(), op.getPermutation());
    return mlir::success();
  }
};

struct VectorizeSliceOp : public mlir::OpConversionPattern<shlo::SliceOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      shlo::SliceOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type new_type = this->getTypeConverter()->convertType(op.getType());
    if (!new_type) {
      return mlir::failure();
    }
    auto res_vec_ty = mlir::cast<mlir::VectorType>(new_type);
    llvm::ArrayRef<int64_t> shape = res_vec_ty.getShape();
    int64_t rank = res_vec_ty.getRank();

    bool all_unit_strides =
        llvm::all_of(op.getStrides(), [](int64_t s) { return s == 1; });
    if (all_unit_strides) {
      llvm::SmallVector<int64_t> offsets(op.getStartIndices().begin(),
                                         op.getStartIndices().end());
      llvm::SmallVector<int64_t> strides(rank, 1);
      llvm::SmallVector<int64_t> sizes(shape.begin(), shape.end());
      rewriter.replaceOpWithNewOp<mv::ExtractStridedSliceOp>(
          op, adaptor.getOperand(), offsets, sizes, strides);
      return mlir::success();
    }

    mlir::Value res = ma::ConstantOp::create(rewriter, op.getLoc(), res_vec_ty,
                                             rewriter.getZeroAttr(res_vec_ty));
    llvm::SmallVector<int64_t> start_indices(op.getStartIndices().begin(),
                                             op.getStartIndices().end());
    llvm::SmallVector<int64_t> strides(op.getStrides().begin(),
                                       op.getStrides().end());

    llvm::SmallVector<int64_t> curr_dst(rank, 0);
    llvm::SmallVector<int64_t> curr_src(rank, 0);

    auto emit_elements = [&](auto& self, int64_t dim) -> void {
      if (dim == rank) {
        mlir::Value elem = mv::ExtractOp::create(
            rewriter, op.getLoc(), adaptor.getOperand(), curr_src);
        res = mv::InsertOp::create(rewriter, op.getLoc(), elem, res, curr_dst);
        return;
      }
      for (int64_t i = 0; i < shape[dim]; ++i) {
        curr_dst[dim] = i;
        curr_src[dim] = start_indices[dim] + i * strides[dim];
        self(self, dim + 1);
      }
    };
    emit_elements(emit_elements, 0);

    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

struct VectorizeConcatenateOp
    : public mlir::OpConversionPattern<shlo::ConcatenateOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      shlo::ConcatenateOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type new_type = this->getTypeConverter()->convertType(op.getType());
    if (!new_type) {
      return mlir::failure();
    }
    auto res_vec_ty = mlir::cast<mlir::VectorType>(new_type);
    mlir::Value res = ma::ConstantOp::create(rewriter, op.getLoc(), res_vec_ty,
                                             rewriter.getZeroAttr(res_vec_ty));
    int64_t current_offset = 0;
    uint64_t dim = op.getDimension();
    for (mlir::Value input : adaptor.getInputs()) {
      auto in_vec_ty = mlir::cast<mlir::VectorType>(input.getType());
      llvm::SmallVector<int64_t> offsets(res_vec_ty.getRank(), 0);
      offsets[dim] = current_offset;
      llvm::SmallVector<int64_t> strides(in_vec_ty.getRank(), 1);
      res = mv::InsertStridedSliceOp::create(rewriter, op.getLoc(), input, res,
                                             rewriter.getI64ArrayAttr(offsets),
                                             rewriter.getI64ArrayAttr(strides));
      current_offset += in_vec_ty.getDimSize(dim);
    }
    rewriter.replaceOp(op, res);
    return mlir::success();
  }
};

struct VectorizeReshapeOp : public mlir::OpConversionPattern<shlo::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      shlo::ReshapeOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type new_type = this->getTypeConverter()->convertType(op.getType());
    if (!new_type) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<mv::ShapeCastOp>(op, new_type,
                                                 adaptor.getOperand());
    return mlir::success();
  }
};

struct VectorizeConstantOp : public mlir::OpConversionPattern<ma::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ConstantOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(op.getType());
    if (!tensor_type) {
      return mlir::failure();
    }
    mlir::VectorType vector_type = mlir::cast<mlir::VectorType>(
        this->getTypeConverter()->convertType(tensor_type));
    auto old_attr = mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    if (!old_attr) {
      return mlir::failure();
    }
    auto new_attr = old_attr.reshape(vector_type);
    rewriter.replaceOpWithNewOp<ma::ConstantOp>(op, vector_type, new_attr);
    return mlir::success();
  }
};

struct VectorizeIotaOp : public mlir::OpConversionPattern<shlo::IotaOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      shlo::IotaOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    if (mlir::isa<mlir::ComplexType>(op.getType().getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "complex types are not supported by vector operations");
    }
    int64_t iota_dim = op.getIotaDimension();
    mlir::Type new_type = this->getTypeConverter()->convertType(op.getType());
    if (!new_type) {
      return mlir::failure();
    }
    auto result_vector_type = mlir::cast<mlir::VectorType>(new_type);
    int64_t iota_size = result_vector_type.getShape()[iota_dim];
    auto i32_1d_vector_type =
        mlir::VectorType::get({iota_size}, rewriter.getI32Type());

    llvm::SmallVector<mlir::Attribute> iota_values(iota_size);
    for (int idx = 0; idx != iota_size; ++idx) {
      iota_values[idx] = rewriter.getI32IntegerAttr(idx);
    }

    mlir::Value iota_const = ma::ConstantOp::create(
        rewriter, op->getLoc(),
        mlir::DenseElementsAttr::get(i32_1d_vector_type, iota_values));

    auto i32_result_vector_type = mlir::VectorType::get(
        result_vector_type.getShape(), rewriter.getI32Type());

    if (result_vector_type.getRank() > 1) {
      llvm::SmallVector<int64_t> intermediate_shape(
          result_vector_type.getRank(), 1);
      intermediate_shape[iota_dim] = iota_size;
      auto intermediate_vector_type =
          mlir::VectorType::get(intermediate_shape, rewriter.getI32Type());

      mlir::Value intermediate_vector = mlir::vector::ShapeCastOp::create(
          rewriter, op->getLoc(), intermediate_vector_type, iota_const);

      iota_const = mlir::vector::BroadcastOp::create(
          rewriter, op->getLoc(), i32_result_vector_type, intermediate_vector);
    }

    mlir::ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
    auto converted_or =
        xtile::LowerConvert(builder, op->getLoc(), iota_const,
                            i32_result_vector_type, result_vector_type);
    if (!converted_or.ok()) {
      return rewriter.notifyMatchFailure(
          op, absl::StrCat("Type conversion not supported: ",
                           converted_or.status().message()));
    }

    rewriter.replaceOp(op, converted_or.value());
    return mlir::success();
  }
};

template <typename OpTy>
struct VectorizeElementwiseOp : public mlir::OpConversionPattern<OpTy> {
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      mlir::ConversionPatternRewriter& rewriter) const override {
    mlir::Type new_type = this->getTypeConverter()->convertType(op.getType());
    if (!new_type) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<OpTy>(op, new_type, adaptor.getOperands(),
                                      op->getAttrs());
    return mlir::success();
  }
};

template <typename... Ops>
void populateVectorizePatterns(mlir::TypeConverter& converter,
                               mlir::RewritePatternSet& patterns) {
  patterns.add<VectorizeElementwiseOp<Ops>...>(converter,
                                               patterns.getContext());
}

class VectorizeXTilePass
    : public impl::VectorizeXTilePassBase<VectorizeXTilePass> {
 public:
  using VectorizeXTilePassBase::VectorizeXTilePassBase;

  void runOnOperation() override {
    mlir::TypeConverter type_converter;
    auto unrealized_conversion_cast =
        [](mlir::OpBuilder& builder, mlir::Type result_type, ValueRange inputs,
           mlir::Location loc) -> Value {
      return mlir::UnrealizedConversionCastOp::create(builder, loc, result_type,
                                                      inputs)
          .getResult(0);
    };
    type_converter.addTargetMaterialization(unrealized_conversion_cast);
    type_converter.addSourceMaterialization(unrealized_conversion_cast);
    type_converter.addConversion(
        [](mlir::Type type) -> std::optional<mlir::Type> {
          if (auto tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
            return mlir::VectorType::get(tensor_type.getShape(),
                                         tensor_type.getElementType());
          }
          return type;
        });

    mlir::MLIRContext* context = &getContext();
    mlir::ConversionTarget target(*context);
    target.addLegalDialect<
        ma::ArithDialect, mlir::memref::MemRefDialect, mm::MathDialect,
        ms::SCFDialect, mlir::tensor::TensorDialect, mv::VectorDialect,
        shlo::StablehloDialect, xla::XlaDialect, xtile::XTileDialect>();
    target.addIllegalOp<shlo::BroadcastInDimOp, shlo::ConcatenateOp,
                        shlo::DotGeneralOp, shlo::IotaOp, shlo::ReduceOp,
                        shlo::ReshapeOp, shlo::SliceOp, shlo::TransposeOp,
                        mlir::tensor::ExtractOp, mlir::tensor::FromElementsOp,
                        xtile::ExtractTileOp, xtile::InsertTileOp,
                        xtile::MaskOp>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addDynamicallyLegalDialect<ma::ArithDialect, mm::MathDialect>(
        [&](mlir::Operation* op) { return type_converter.isLegal(op); });

    mlir::RewritePatternSet patterns(context);
    patterns
        .add<ConvertExtractTile, ConvertInsertTile, VectorizeBroadcastInDimOp,
             VectorizeConcatenateOp, VectorizeConstantOp, VectorizeDotGeneralOp,
             VectorizeExtractOp, VectorizeFromElementsOp, VectorizeIotaOp,
             VectorizeMaskOp, VectorizeReduceOp, VectorizeReshapeOp,
             VectorizeSliceOp, VectorizeTransposeOp>(type_converter, context);
    populateVectorizePatterns<
        ma::AddFOp, ma::AddIOp, ma::SubFOp, ma::SubIOp, ma::MulFOp, ma::MulIOp,
        ma::DivFOp, ma::DivSIOp, ma::DivUIOp, ma::RemFOp, ma::RemSIOp,
        ma::RemUIOp, ma::MaximumFOp, ma::MaxSIOp, ma::MaxUIOp, ma::MinimumFOp,
        ma::MinSIOp, ma::MinUIOp, ma::AndIOp, ma::OrIOp, ma::XOrIOp, ma::NegFOp,
        ma::SelectOp, ma::CmpFOp, ma::CmpIOp, ma::ExtFOp, ma::TruncFOp,
        ma::ExtSIOp, ma::ExtUIOp, ma::FPToSIOp, ma::FPToUIOp, ma::SIToFPOp,
        ma::UIToFPOp, ma::TruncIOp, ma::IndexCastOp, mm::AbsIOp, mm::AbsFOp,
        mm::CeilOp, mm::FloorOp, mm::RoundEvenOp, mm::AcosOp, mm::AcoshOp,
        mm::AsinOp, mm::AsinhOp, mm::Atan2Op, mm::AtanhOp, mm::CosOp,
        mm::CoshOp, mm::ExpOp, mm::ErfOp, mm::ExpM1Op, mm::LogOp, mm::Log1pOp,
        mm::IPowIOp, mm::PowFOp, mm::RsqrtOp, mm::SinOp, mm::SinhOp, mm::SqrtOp,
        mm::TanOp, mm::TanhOp, mm::CbrtOp, mm::IsFiniteOp>(type_converter,
                                                           patterns);
    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        type_converter, patterns, target);

    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        type_converter, patterns, target);
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace xla::cpu
