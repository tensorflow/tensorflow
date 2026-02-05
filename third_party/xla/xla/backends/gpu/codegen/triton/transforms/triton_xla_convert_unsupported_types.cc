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

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
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

template <>
LogicalResult
GenericOpConversionPattern<::xla::xtile::ExtractTileOp>::matchAndRewrite(
    ::xla::xtile::ExtractTileOp op,
    ::xla::xtile::ExtractTileOp::Adaptor adaptor,
    ConversionPatternRewriter& rewriter) const {
  auto* ctx = op.getContext();
  ::xla::xtile::ExtractTileOp replacement =
      mlir::cast<::xla::xtile::ExtractTileOp>(rewriter.clone(*op));
  if (op.getResult().getType().getElementType() == Float4E2M1FNType::get(ctx)) {
    auto full_tile_shape = op.getFullTileShape().vec();
    full_tile_shape[full_tile_shape.size() - 1] = full_tile_shape.back() / 2;
    replacement.setFullTileShape(full_tile_shape);
  }
  replacement->setOperands(adaptor.getOperands());
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
    target.markUnknownOpDynamicallyLegal(
        [&](Operation* op) { return converter.isLegal(op); });

    RewritePatternSet patterns(ctx);
    patterns.add<
        // go/keep-sorted start
        GenericOpConversionPattern<::xla::xtile::ExtractTileOp>,
        GenericOpConversionPattern<::xla::xtile::InsertTileOp>,
        GenericOpConversionPattern<BroadcastOp>,
        GenericOpConversionPattern<DotScaledOp>,
        GenericOpConversionPattern<ExpandDimsOp>,
        GenericOpConversionPattern<ReshapeOp>,
        GenericOpConversionPattern<TransOp>,
        GenericOpConversionPattern<arith::BitcastOp>
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
