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
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
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
    if (is_in_bounds) {
      is_in_bounds = ma::AndIOp::create(builder, loc, is_in_bounds, cmp);
    } else {
      is_in_bounds = cmp;
    }
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

    // Check if in bounds
    Value is_in_bounds = GetIsInBoundsCondition(
        rewriter, loc, offsets, source_memref, result_vector_type.getShape());

    Value pad = ma::ConstantOp::create(
        rewriter, loc, result_vector_type.getElementType(),
        rewriter.getZeroAttr(result_vector_type.getElementType()));

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
    target.addIllegalOp<shlo::TransposeOp, xtile::ExtractTileOp,
                        xtile::InsertTileOp>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addDynamicallyLegalDialect<ma::ArithDialect, mm::MathDialect>(
        [&](mlir::Operation* op) { return type_converter.isLegal(op); });

    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertExtractTile, ConvertInsertTile, VectorizeTransposeOp>(
        type_converter, context);
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

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace xla::cpu
