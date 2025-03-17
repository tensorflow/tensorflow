/* Copyright 2022 The OpenXLA Authors.

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

// This file implements logic for lowering HLO dialect to LHLO dialect.

#include <memory>
#include <optional>
#include <utility>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_HLOLEGALIZETOARITHMETICPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct RngGetAndUpdateStatePattern
    : public OpConversionPattern<mhlo::XlaRngGetAndUpdateStateOp> {
  using OpConversionPattern<
      mhlo::XlaRngGetAndUpdateStateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::XlaRngGetAndUpdateStateOp op,
      XlaRngGetAndUpdateStateOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // Get various type related information
    auto loc = op->getLoc();

    const auto globalName = rewriter.getStringAttr("rng_state");
    constexpr auto initialSeed = 0x7012395ull;
    auto seedType = rewriter.getIntegerType(128);
    auto memrefType = MemRefType::get({}, seedType);

    auto resultType = op.getType();
    auto wordSize = resultType.getElementType().getIntOrFloatBitWidth();
    auto smallerIntType = rewriter.getIntegerType(wordSize);
    auto numElements = resultType.getNumElements();

    // Get or define the global variable
    auto* globalOp = mlir::SymbolTable::lookupNearestSymbolFrom(op, globalName);
    if (!globalOp) {
      auto* parent = mlir::SymbolTable::getNearestSymbolTable(op);
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&parent->getRegions().front().front());

      const auto priv = rewriter.getStringAttr("private");
      auto initialValue = mlir::DenseElementsAttr::get(
          mlir::RankedTensorType::get({}, seedType),
          rewriter.getIntegerAttr(seedType, initialSeed));
      globalOp = rewriter.create<memref::GlobalOp>(
          loc, globalName, priv, memrefType, initialValue, /*constant=*/false,
          /*alignment=*/IntegerAttr());
    }
    assert(isa<memref::GlobalOp>(globalOp) &&
           "rng_state was defined somewhere else, not as a global op");

    // Get and update
    Value rngState =
        rewriter.create<memref::GetGlobalOp>(loc, memrefType, globalName);
    Value oldVal = rewriter.create<memref::LoadOp>(loc, rngState);
    Value delta = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(seedType,
                                     static_cast<int64_t>(adaptor.getDelta())));
    Value newVal = rewriter.create<arith::AddIOp>(loc, oldVal, delta);
    (void)rewriter.create<memref::StoreOp>(loc, newVal, rngState);

    // Create the proper return type by packing the old seed into a tensor
    SmallVector<Value> pieces;
    for (int i = (numElements - 1) * wordSize; i >= 0; i -= wordSize) {
      Value shiftDistance = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIntegerAttr(seedType, i));
      pieces.push_back(rewriter.create<arith::TruncIOp>(
          loc, smallerIntType,
          rewriter.create<arith::ShRUIOp>(loc, oldVal, shiftDistance)));
    }

    // Obtain a tensor with the correct shape and bit widths but the incorrect
    // integer signedness, then cast the tensor to the correct signedness to
    // ensure that unrealized casts will successfully lower later.
    Value resultTensor = rewriter.create<tensor::FromElementsOp>(
        loc, mlir::RankedTensorType::get(resultType.getShape(), smallerIntType),
        pieces);
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(op, resultType,
                                                            resultTensor);
    return success();
  }
};

template <typename OpTy>
struct ScalarHloToArithmeticPattern : public OpConversionPattern<OpTy> {
  ScalarHloToArithmeticPattern(
      TypeConverter& typeConverter, MLIRContext* context,
      llvm::function_ref<bool(Operation*)> filterFn = nullptr,
      PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(typeConverter, context, benefit),
        filterFn(filterFn) {}

  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (filterFn && !filterFn(op)) return failure();

    auto isScalar = [&](Value v) {
      return mlir::cast<ShapedType>(v.getType()).getRank() == 0;
    };

    if (!llvm::all_of(adaptor.getOperands(), isScalar))
      return rewriter.notifyMatchFailure(op, "All operands must be scalar.");

    auto loc = op.getLoc();

    std::optional<ShapedType> resultTy;
    resultTy = mlir::dyn_cast<ShapedType>(
        this->typeConverter->convertType(op->getResultTypes().front()));

    SmallVector<Value> operands;
    for (auto operand : adaptor.getOperands()) {
      operands.push_back(
          rewriter.create<tensor::ExtractOp>(loc, operand, ValueRange()));
    }
    Value scalarResult = mhlo::MhloOpToStdScalarOp::mapOp(
        op, resultTy->getElementType(), operands, /*attributes=*/std::nullopt,
        &rewriter);
    if (!scalarResult) return failure();
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, *resultTy,
                                                        scalarResult);
    return success();
  }

 private:
  llvm::function_ref<bool(Operation*)> filterFn;
};

struct HloLegalizeToArithmeticPass
    : public impl::HloLegalizeToArithmeticPassBase<
          HloLegalizeToArithmeticPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect,
                    tensor::TensorDialect>();
  }

 public:
  void runOnOperation() override {
    auto& context = getContext();
    RewritePatternSet patterns(&context);
    ConversionTarget target(context);

    populateHloToArithmeticConversionPatterns(&patterns);

    target.addIllegalOp<XlaRngGetAndUpdateStateOp>();
    target.addLegalDialect<arith::ArithDialect, BuiltinDialect,
                           memref::MemRefDialect, tensor::TensorDialect>();

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

void populateHloToArithmeticConversionPatterns(RewritePatternSet* patterns) {
  patterns->add<RngGetAndUpdateStatePattern>(patterns->getContext());
}

void populateScalarHloToArithmeticConversionPatterns(
    MLIRContext* context, TypeConverter& typeConverter,
    RewritePatternSet* patterns,
    llvm::function_ref<bool(Operation*)> filterFn) {
  // clang-format off
  patterns->add<
      ScalarHloToArithmeticPattern<mhlo::AbsOp>,
      ScalarHloToArithmeticPattern<mhlo::AddOp>,
      ScalarHloToArithmeticPattern<mhlo::AndOp>,
      ScalarHloToArithmeticPattern<mhlo::Atan2Op>,
      ScalarHloToArithmeticPattern<mhlo::BitcastConvertOp>,
      ScalarHloToArithmeticPattern<mhlo::CbrtOp>,
      ScalarHloToArithmeticPattern<mhlo::CeilOp>,
      ScalarHloToArithmeticPattern<mhlo::ClampOp>,
      ScalarHloToArithmeticPattern<mhlo::ClzOp>,
      ScalarHloToArithmeticPattern<mhlo::CompareOp>,
      ScalarHloToArithmeticPattern<mhlo::ComplexOp>,
      ScalarHloToArithmeticPattern<mhlo::ConvertOp>,
      ScalarHloToArithmeticPattern<mhlo::CopyOp>,
      ScalarHloToArithmeticPattern<mhlo::CosineOp>,
      ScalarHloToArithmeticPattern<mhlo::DivOp>,
      ScalarHloToArithmeticPattern<mhlo::ErfOp>,
      ScalarHloToArithmeticPattern<mhlo::ExpOp>,
      ScalarHloToArithmeticPattern<mhlo::Expm1Op>,
      ScalarHloToArithmeticPattern<mhlo::FloorOp>,
      ScalarHloToArithmeticPattern<mhlo::ImagOp>,
      ScalarHloToArithmeticPattern<mhlo::IsFiniteOp>,
      ScalarHloToArithmeticPattern<mhlo::Log1pOp>,
      ScalarHloToArithmeticPattern<mhlo::LogOp>,
      ScalarHloToArithmeticPattern<mhlo::LogisticOp>,
      ScalarHloToArithmeticPattern<mhlo::MaxOp>,
      ScalarHloToArithmeticPattern<mhlo::MinOp>,
      ScalarHloToArithmeticPattern<mhlo::MulOp>,
      ScalarHloToArithmeticPattern<mhlo::NegOp>,
      ScalarHloToArithmeticPattern<mhlo::NotOp>,
      ScalarHloToArithmeticPattern<mhlo::OrOp>,
      ScalarHloToArithmeticPattern<mhlo::PopulationCountOp>,
      ScalarHloToArithmeticPattern<mhlo::PowOp>,
      ScalarHloToArithmeticPattern<mhlo::RealOp>,
      ScalarHloToArithmeticPattern<mhlo::ReducePrecisionOp>,
      ScalarHloToArithmeticPattern<mhlo::RemOp>,
      ScalarHloToArithmeticPattern<mhlo::RoundNearestEvenOp>,
      ScalarHloToArithmeticPattern<mhlo::RoundOp>,
      ScalarHloToArithmeticPattern<mhlo::RsqrtOp>,
      ScalarHloToArithmeticPattern<mhlo::SelectOp>,
      ScalarHloToArithmeticPattern<mhlo::ShiftLeftOp>,
      ScalarHloToArithmeticPattern<mhlo::ShiftRightArithmeticOp>,
      ScalarHloToArithmeticPattern<mhlo::ShiftRightLogicalOp>,
      ScalarHloToArithmeticPattern<mhlo::SignOp>,
      ScalarHloToArithmeticPattern<mhlo::SineOp>,
      ScalarHloToArithmeticPattern<mhlo::SqrtOp>,
      ScalarHloToArithmeticPattern<mhlo::SubtractOp>,
      ScalarHloToArithmeticPattern<mhlo::TanOp>,
      ScalarHloToArithmeticPattern<mhlo::TanhOp>,
      ScalarHloToArithmeticPattern<mhlo::XorOp>
  >(typeConverter, context, filterFn);
  // clang-format on
}

std::unique_ptr<OperationPass<ModuleOp>> createLegalizeToArithmeticPass() {
  return std::make_unique<HloLegalizeToArithmeticPass>();
}

}  // namespace mhlo
}  // namespace mlir
