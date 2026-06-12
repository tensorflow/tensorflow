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

#include <memory>
#include <utility>

#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/codegen/xtile/ir/xtile_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLACONVERTUNSUPPORTEDTYPESPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

bool FindScaledDotOp(const ModuleOp& module) {
  auto walk_result = module->walk([&](Operation* op) {
    if (auto extSI = dyn_cast<triton::DotScaledOp>(op)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return walk_result.wasInterrupted();
}

template <typename OpType>
struct GenericOpConversionPattern final : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpType op, typename OpType::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Operation* replacement = rewriter.clone(*op);
    replacement->setOperands(adaptor.getOperands());
    const TypeConverter* converter = this->getTypeConverter();
    for (auto result : replacement->getResults()) {
      result.setType(converter->convertType(result.getType()));
    }
    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct ConstantOpConversionPattern final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::ConstantOp op, arith::ConstantOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Type new_type = getTypeConverter()->convertType(op.getType());
    if (new_type == op.getType()) {
      return failure();
    }
    auto dense_attr = llvm::dyn_cast<DenseElementsAttr>(op.getValue());
    if (!dense_attr) {
      return failure();
    }
    auto new_shaped_type = llvm::cast<ShapedType>(new_type);
    auto new_attr = DenseElementsAttr::getFromRawBuffer(
        new_shaped_type, dense_attr.getRawData());
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, new_attr);
    return success();
  }
};

struct TransposeOpConversionPattern final
    : public OpConversionPattern<mlir::stablehlo::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mlir::stablehlo::TransposeOp op,
      mlir::stablehlo::TransposeOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value converted_input = adaptor.getOperand();
    auto input_shape =
        mlir::cast<ShapedType>(converted_input.getType()).getShape();
    auto permutation = op.getPermutation();
    SmallVector<int64_t> new_shape(permutation.size());
    for (int i = 0; i < permutation.size(); ++i) {
      new_shape[i] = input_shape[permutation[i]];
    }
    Type new_result_type = mlir::RankedTensorType::get(
        new_shape,
        mlir::cast<ShapedType>(converted_input.getType()).getElementType());
    rewriter.replaceOpWithNewOp<mlir::stablehlo::TransposeOp>(
        op, new_result_type, converted_input, op.getPermutationAttr());
    return success();
  }
};

struct TransOpConversionPattern final : public OpConversionPattern<TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TransOp op, TransOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Value converted_input = adaptor.getSrc();
    auto input_shape =
        mlir::cast<ShapedType>(converted_input.getType()).getShape();
    auto order = op.getOrder();
    SmallVector<int64_t> new_shape(order.size());
    for (int i = 0; i < order.size(); ++i) {
      new_shape[i] = input_shape[order[i]];
    }
    Type new_result_type = mlir::RankedTensorType::get(
        new_shape,
        mlir::cast<ShapedType>(converted_input.getType()).getElementType());
    rewriter.replaceOpWithNewOp<TransOp>(op, new_result_type, converted_input,
                                         op.getOrderAttr());
    return success();
  }
};

Value LookThroughCast(Value val) {
  if (auto cast_op = val.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    return cast_op.getOperand(0);
  }
  return val;
}

struct DotScaledOpConversionPattern final
    : public OpConversionPattern<triton::DotScaledOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      triton::DotScaledOp op, triton::DotScaledOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Value> operands = {
        LookThroughCast(adaptor.getA()), LookThroughCast(adaptor.getB()),
        LookThroughCast(adaptor.getC()), LookThroughCast(adaptor.getAScale()),
        LookThroughCast(adaptor.getBScale())};
    Type result_type = adaptor.getC().getType();
    rewriter.replaceOpWithNewOp<triton::DotScaledOp>(op, result_type, operands,
                                                     op->getAttrs());
    return success();
  }
};

template <>
LogicalResult
GenericOpConversionPattern<::xla::xtile::ExtractTileOp>::matchAndRewrite(
    ::xla::xtile::ExtractTileOp op,
    ::xla::xtile::ExtractTileOp::Adaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto* ctx = op.getContext();
  SmallVector<Value> operands(adaptor.getOperands().begin(),
                              adaptor.getOperands().end());
  if (op.getResult().getType().getElementType() == Float4E2M1FNType::get(ctx)) {
    int rank = mlir::cast<mlir::MemRefType>(op.getSource().getType()).getRank();
    Value last_offset = operands[rank];
    rewriter.setInsertionPoint(op);
    Location loc = op.getLoc();
    auto const_attr = rewriter.getIntegerAttr(last_offset.getType(), 2);
    auto const_op = rewriter.create<arith::ConstantOp>(loc, const_attr);
    Value divided_offset =
        rewriter.create<arith::DivSIOp>(loc, last_offset, const_op);
    operands[rank] = divided_offset;
  }

  ::xla::xtile::ExtractTileOp replacement =
      mlir::cast<::xla::xtile::ExtractTileOp>(rewriter.clone(*op));
  if (op.getResult().getType().getElementType() == Float4E2M1FNType::get(ctx)) {
    auto full_tile_shape = op.getFullTileShape().vec();
    full_tile_shape[full_tile_shape.size() - 1] = full_tile_shape.back() / 2;
    replacement.setFullTileShape(full_tile_shape);
  }
  replacement->setOperands(operands);
  const TypeConverter* converter = this->getTypeConverter();
  for (auto result : replacement->getResults()) {
    result.setType(converter->convertType(result.getType()));
  }
  rewriter.replaceOp(op, replacement);
  return success();
}

class TritonXLAConvertUnsupportedTypesPass
    : public impl::TritonXLAConvertUnsupportedTypesPassBase<
          TritonXLAConvertUnsupportedTypesPass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    if (!FindScaledDotOp(getOperation())) {
      return;
    }
    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([&](Float8E8M0FNUType type) {
      return IntegerType::get(type.getContext(), 8);
    });
    converter.addConversion([&](ShapedType type) {
      if (llvm::isa<Float4E2M1FNType>(type.getElementType())) {
        auto shape = type.getShape().vec();
        shape.back() /= 2;
        return type.clone(shape, IntegerType::get(type.getContext(), 8));
      }
      return type.clone(converter.convertType(type.getElementType()));
    });

    converter.addConversion([&](triton::PointerType type) {
      return triton::PointerType::get(
          converter.convertType(type.getPointeeType()), type.getAddressSpace());
    });
    converter.addConversion([&](FunctionType type) -> Type {
      SmallVector<Type> new_inputs, new_results;
      if (failed(converter.convertTypes(type.getInputs(), new_inputs)) ||
          failed(converter.convertTypes(type.getResults(), new_results))) {
        return nullptr;
      }
      return type.clone(new_inputs, new_results);
    });

    auto* ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<::xla::xtile::EntryFuncOp>(
        [&](::xla::xtile::EntryFuncOp op) {
          return converter.isSignatureLegal(op.getFunctionType()) &&
                 converter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<triton::DotScaledOp>(
        [&](triton::DotScaledOp op) {
          return !mlir::isa<Float4E2M1FNType>(
              mlir::cast<ShapedType>(op.getA().getType()).getElementType());
        });
    target.addDynamicallyLegalOp<triton::TransOp>([&](triton::TransOp op) {
      return !mlir::isa<Float4E2M1FNType>(
          mlir::cast<ShapedType>(op.getSrc().getType()).getElementType());
    });
    target.addDynamicallyLegalOp<mlir::stablehlo::TransposeOp>(
        [&](mlir::stablehlo::TransposeOp op) {
          return !mlir::isa<Float4E2M1FNType>(
              mlir::cast<ShapedType>(op.getOperand().getType())
                  .getElementType());
        });
    target.markUnknownOpDynamicallyLegal(
        [&](Operation* op) { return converter.isLegal(op); });

    RewritePatternSet patterns(ctx);
    patterns.add<
        // go/keep-sorted start
        ConstantOpConversionPattern, DotScaledOpConversionPattern,
        GenericOpConversionPattern<::xla::xtile::ExtractTileOp>,
        GenericOpConversionPattern<::xla::xtile::InsertTileOp>,
        GenericOpConversionPattern<BroadcastOp>,
        GenericOpConversionPattern<ExpandDimsOp>,
        GenericOpConversionPattern<ReshapeOp>,
        GenericOpConversionPattern<SplatOp>,
        GenericOpConversionPattern<arith::BitcastOp>,
        GenericOpConversionPattern<arith::SelectOp>,
        GenericOpConversionPattern<mlir::stablehlo::ReshapeOp>,
        GenericOpConversionPattern<tensor::ExtractOp>,
        TransOpConversionPattern, TransposeOpConversionPattern
        // go/keep-sorted end
        >(converter, ctx);
    scf::populateSCFStructuralTypeConversions(converter, patterns);
    populateFunctionOpInterfaceTypeConversionPattern<::xla::xtile::EntryFuncOp>(
        patterns, converter);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> CreateTritonXLAConvertUnsupportedTypesPass() {
  return std::make_unique<TritonXLAConvertUnsupportedTypesPass>();
}

}  // namespace mlir::triton::xla
