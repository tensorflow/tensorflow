/* Copyright 2020 The OpenXLA Authors.

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

#include <optional>
#include <utility>
#include <vector>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mhlo/utils/type_conversion.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_CHLOLEGALIZETOHLOPASS
#define GEN_PASS_DEF_CHLOLEGALIZETOHIGHLEVELMHLOPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

ChloLegalizeToHighLevelMhloPassOptions FromPassOptions(
    bool enableAcosh, bool enableAcos, bool enableAsin, bool enableAsinh,
    bool enableAtanh, bool enableCosh, bool enableSinh) {
  ChloLegalizeToHighLevelMhloPassOptions options;
  options.enable_acosh_ = enableAcosh;
  options.enable_acos_ = enableAcos;
  options.enable_atanh_ = enableAtanh;
  options.enable_cosh_ = enableCosh;
  options.enable_sinh_ = enableSinh;
  options.enable_asin_ = enableAsin;
  options.enable_asinh_ = enableAsinh;
  return options;
}

static bool qualifiesForDirectMhloLoweringAcos(chlo::AcosOp op) {
  return llvm::isa<FloatType>(getElementTypeOrSelf(op.getType()));
}

static bool qualifiesForDirectMhloLoweringAcosh(chlo::AcoshOp op) {
  return llvm::isa<FloatType>(getElementTypeOrSelf(op.getType()));
}

static bool qualifiesForDirectMhloLoweringAtanh(chlo::AtanhOp op) {
  return llvm::isa<FloatType>(getElementTypeOrSelf(op.getType()));
}

static bool qualifiesForDirectMhloLoweringCosh(chlo::CoshOp op) {
  return llvm::isa<FloatType>(getElementTypeOrSelf(op.getType()));
}

static bool qualifiesForDirectMhloLoweringSinh(chlo::SinhOp op) {
  return llvm::isa<FloatType>(getElementTypeOrSelf(op.getType()));
}

static bool qualifiesForDirectMhloLoweringAsin(chlo::AsinOp op) {
  return llvm::isa<FloatType>(getElementTypeOrSelf(op.getType()));
}

static bool qualifiesForDirectMhloLoweringAsinh(chlo::AsinhOp op) {
  return llvm::isa<FloatType>(getElementTypeOrSelf(op.getType()));
}

struct ChloLegalizeToHighLevelMhloPass
    : public impl::ChloLegalizeToHighLevelMhloPassBase<
          ChloLegalizeToHighLevelMhloPass> {
  ChloLegalizeToHighLevelMhloPass() = default;
  explicit ChloLegalizeToHighLevelMhloPass(
      ChloLegalizeToHighLevelMhloPassOptions options)
      : impl::ChloLegalizeToHighLevelMhloPassBase<
            ChloLegalizeToHighLevelMhloPass>(options) {}

  void runOnOperation() override {
    MLIRContext& context = getContext();
    ConversionTarget conversionTarget(context);
    RewritePatternSet conversionPatterns(&context);

    chlo::populateChloToHighLevelMhloOpPatterns(
        &context, &conversionPatterns,
        FromPassOptions(enable_acosh_, enable_acos_, enable_asin_,
                        enable_asinh_, enable_atanh_, enable_cosh_,
                        enable_sinh_));

    // Consider the mhlo dialect legal for tests. Also add helper dialects
    // that are needed by the patterns.
    conversionTarget.addLegalDialect<chlo::ChloDialect, mhlo::MhloDialect>();
    if (enable_acosh_) {
      conversionTarget.addDynamicallyLegalOp<chlo::AcoshOp>(
          [](chlo::AcoshOp op) {
            return !qualifiesForDirectMhloLoweringAcosh(op);
          });
    }
    if (enable_acos_) {
      conversionTarget.addDynamicallyLegalOp<chlo::AcosOp>([](chlo::AcosOp op) {
        return !qualifiesForDirectMhloLoweringAcos(op);
      });
    }
    if (enable_atanh_) {
      conversionTarget.addDynamicallyLegalOp<chlo::AtanhOp>(
          [](chlo::AtanhOp op) {
            return !qualifiesForDirectMhloLoweringAtanh(op);
          });
    }
    if (enable_cosh_) {
      conversionTarget.addDynamicallyLegalOp<chlo::CoshOp>([](chlo::CoshOp op) {
        return !qualifiesForDirectMhloLoweringCosh(op);
      });
    }
    if (enable_sinh_) {
      conversionTarget.addDynamicallyLegalOp<chlo::SinhOp>([](chlo::SinhOp op) {
        return !qualifiesForDirectMhloLoweringSinh(op);
      });
    }
    if (enable_asin_) {
      conversionTarget.addDynamicallyLegalOp<chlo::AsinOp>([](chlo::AsinOp op) {
        return !qualifiesForDirectMhloLoweringAsin(op);
      });
    }
    if (enable_asinh_) {
      conversionTarget.addDynamicallyLegalOp<chlo::AsinhOp>(
          [](chlo::AsinhOp op) {
            return !qualifiesForDirectMhloLoweringAsinh(op);
          });
    }
    conversionTarget.addIllegalOp<chlo::TopKOp, chlo::ErfOp, chlo::RaggedDotOp,
                                  chlo::ScanOp>();

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }
  }
};

struct ChloLegalizeToHloPass
    : public impl::ChloLegalizeToHloPassBase<ChloLegalizeToHloPass> {
  using ChloLegalizeToHloPassBase::ChloLegalizeToHloPassBase;

  void runOnOperation() override {
    MLIRContext& context = getContext();
    ConversionTarget conversionTarget(context);
    RewritePatternSet conversionPatterns(&context);

    stablehlo::StablehloToHloTypeConverter typeConverter;
    chlo::populateChloToHloPatterns(&context, &typeConverter,
                                    &conversionPatterns);

    // Consider the mhlo dialect legal for tests. Also add helper dialects
    // that are needed by the patterns.
    conversionTarget
        .addIllegalDialect<chlo::ChloDialect, stablehlo::StablehloDialect>();
    conversionTarget.addLegalDialect<
        MhloDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect,
        mlir::tensor::TensorDialect, mlir::shape::ShapeDialect>();

    if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                      std::move(conversionPatterns)))) {
      return signalPassFailure();
    }
  }
};

LogicalResult convertScanChloToMhlo(chlo::ScanOp op,
                                    PatternRewriter& rewriter) {
  auto mhloOp = mhlo::ScanOp::create(
      rewriter, op.getLoc(), op.getOutputs().getTypes(),
      op.getInits().getTypes(), op.getInputs(), op.getInits(),
      op.getDimensionAttr(), op.getIsReverseAttr(), op.getIsAssociativeAttr());

  rewriter.inlineRegionBefore(op.getBody(), mhloOp.getBody(),
                              mhloOp.getBody().end());

  // Convert terminator to mhlo.return
  for (auto& block : mhloOp.getBody()) {
    Operation* terminator = block.getTerminator();
    rewriter.setInsertionPoint(terminator);
    rewriter.replaceOpWithNewOp<mhlo::ReturnOp>(terminator,
                                                terminator->getOperands());
  }

  rewriter.replaceOp(op, mhloOp.getResults());
  return success();
}

LogicalResult convertRaggedDotChloToMhlo(chlo::RaggedDotOp raggedDotOp,
                                         PatternRewriter& rewriter) {
  auto moduleOp = raggedDotOp->getParentOfType<ModuleOp>();

  OpBuilder builder(moduleOp.getBodyRegion());
  builder.setInsertionPointToStart(&moduleOp.getBodyRegion().front());

  auto chloRaggedDotDimNums = raggedDotOp.getRaggedDotDimensionNumbers();
  auto dotDimNums = mhlo::DotDimensionNumbersAttr::get(
      builder.getContext(), chloRaggedDotDimNums.getLhsBatchingDimensions(),
      chloRaggedDotDimNums.getRhsBatchingDimensions(),
      chloRaggedDotDimNums.getLhsContractingDimensions(),
      chloRaggedDotDimNums.getRhsContractingDimensions());
  auto raggedDotDimNums = mhlo::RaggedDotDimensionNumbersAttr::get(
      builder.getContext(), dotDimNums,
      chloRaggedDotDimNums.getLhsRaggedDimensions(),
      chloRaggedDotDimNums.getRhsGroupDimensions());

  auto mhloPrecision =
      [](chlo::Precision precision) -> std::optional<mhlo::Precision> {
    switch (precision) {
      case chlo::Precision::DEFAULT:
        return mhlo::Precision::DEFAULT;
      case chlo::Precision::HIGH:
        return mhlo::Precision::HIGH;
      case chlo::Precision::HIGHEST:
        return mhlo::Precision::HIGHEST;
    }
  };
  ArrayAttr precisionConfig = rewriter.getArrayAttr({});
  if (raggedDotOp.getPrecisionConfig().has_value()) {
    SmallVector<Attribute> vector;
    for (auto configValue : raggedDotOp.getPrecisionConfig()
                                .value()
                                .getAsRange<chlo::PrecisionAttr>()) {
      vector.push_back(mhlo::PrecisionAttr::get(
          raggedDotOp.getContext(),
          mhloPrecision(configValue.getValue()).value()));
    }
    precisionConfig = rewriter.getArrayAttr(vector);
  }

  auto mhloOp = mhlo::RaggedDotOp::create(
      rewriter, raggedDotOp.getLoc(), raggedDotOp.getResult().getType(),
      raggedDotOp.getLhs(), raggedDotOp.getRhs(), raggedDotOp.getGroupSizes(),
      raggedDotDimNums, precisionConfig);
  std::optional<NamedAttribute> frontendAttributes =
      raggedDotOp->getAttrDictionary().getNamed("mhlo.frontend_attributes");
  if (frontendAttributes.has_value()) {
    std::vector<NamedAttribute> attributes =
        mhloOp->getDiscardableAttrDictionary().getValue().vec();
    attributes.push_back(frontendAttributes.value());
    mhloOp->setDiscardableAttrs(rewriter.getDictionaryAttr(attributes));
  }

  rewriter.replaceOp(raggedDotOp, mhloOp.getOperation());
  return success();
}

LogicalResult convertAcoshChloToMhlo(chlo::AcoshOp op,
                                     PatternRewriter& rewriter) {
  if (!mhlo::qualifiesForDirectMhloLoweringAcosh(op)) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<mhlo::AcoshOp>(op, op->getOperands());
  return success();
}

LogicalResult convertAcosChloToMhlo(chlo::AcosOp op,
                                    PatternRewriter& rewriter) {
  if (!mhlo::qualifiesForDirectMhloLoweringAcos(op)) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<mhlo::AcosOp>(op, op->getOperands());
  return success();
}

LogicalResult convertAtanhChloToMhlo(chlo::AtanhOp op,
                                     PatternRewriter& rewriter) {
  if (!mhlo::qualifiesForDirectMhloLoweringAtanh(op)) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<mhlo::AtanhOp>(op, op->getOperands());
  return success();
}

LogicalResult convertCoshChloToMhlo(chlo::CoshOp op,
                                    PatternRewriter& rewriter) {
  if (!mhlo::qualifiesForDirectMhloLoweringCosh(op)) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<mhlo::CoshOp>(op, op->getOperands());
  return success();
}

LogicalResult convertSinhChloToMhlo(chlo::SinhOp op,
                                    PatternRewriter& rewriter) {
  if (!mhlo::qualifiesForDirectMhloLoweringSinh(op)) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<mhlo::SinhOp>(op, op->getOperands());
  return success();
}

LogicalResult convertAsinChloToMhlo(chlo::AsinOp op,
                                    PatternRewriter& rewriter) {
  if (!mhlo::qualifiesForDirectMhloLoweringAsin(op)) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<mhlo::AsinOp>(op, op->getOperands());
  return success();
}

LogicalResult convertAsinhChloToMhlo(chlo::AsinhOp op,
                                     PatternRewriter& rewriter) {
  if (!mhlo::qualifiesForDirectMhloLoweringAsinh(op)) {
    return failure();
  }
  rewriter.replaceOpWithNewOp<mhlo::AsinhOp>(op, op->getOperands());
  return success();
}

}  // namespace

ChloLegalizeToHighLevelMhloPassOptions getDefaultChloToHighLevelMhloOptions() {
  return ChloLegalizeToHighLevelMhloPassOptions();
}

ChloLegalizeToHighLevelMhloPassOptions getGpuChloToHighLevelMhloOptions() {
  ChloLegalizeToHighLevelMhloPassOptions opts;
  opts.enable_acosh_ = true;
  opts.enable_acos_ = true;
  opts.enable_atanh_ = true;
  opts.enable_cosh_ = true;
  opts.enable_sinh_ = true;
  opts.enable_asin_ = true;
  opts.enable_asinh_ = true;
  return opts;
}

}  // namespace mhlo

namespace chlo {
namespace {
#include "chlo_legalize_to_hlo/generated_chlo_legalize_to_hlo.inc"
}  // namespace

void populateChloToHighLevelMhloOpPatterns(
    MLIRContext*, RewritePatternSet* patterns,
    const mhlo::ChloLegalizeToHighLevelMhloPassOptions& options) {
  constexpr unsigned kBenefit = 10;
  if (options.enable_acosh_) {
    patterns->add(mhlo::convertAcoshChloToMhlo, kBenefit);
  }
  if (options.enable_acos_) {
    patterns->add(mhlo::convertAcosChloToMhlo, kBenefit);
  }
  if (options.enable_atanh_) {
    patterns->add(mhlo::convertAtanhChloToMhlo, kBenefit);
  }
  if (options.enable_cosh_) {
    patterns->add(mhlo::convertCoshChloToMhlo, kBenefit);
  }
  if (options.enable_sinh_) {
    patterns->add(mhlo::convertSinhChloToMhlo, kBenefit);
  }
  if (options.enable_asin_) {
    patterns->add(mhlo::convertAsinChloToMhlo, kBenefit);
  }
  if (options.enable_asinh_) {
    patterns->add(mhlo::convertAsinhChloToMhlo, kBenefit);
  }
  patterns->add(mhlo::convertRaggedDotChloToMhlo, kBenefit);
  patterns->add(mhlo::convertScanChloToMhlo, kBenefit);
  populateWithGenerated(*patterns);
}

void populateChloToHighLevelMhloOpPatterns(MLIRContext* context,
                                           RewritePatternSet* patterns) {
  populateChloToHighLevelMhloOpPatterns(
      context, patterns, mhlo::ChloLegalizeToHighLevelMhloPassOptions());
}

void populateChloToHloPatterns(MLIRContext* context,
                               TypeConverter* typeConverter,
                               RewritePatternSet* patterns) {
  populateChloToHighLevelMhloOpPatterns(context, patterns);
  stablehlo::populateChloToStablehloPatterns(context, patterns);
  stablehlo::populateStablehloToHloPatterns(patterns, typeConverter, context);
}

}  // namespace chlo
}  // namespace mlir
