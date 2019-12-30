//===- ConvertStandardToSPIRVPass.cpp - Convert Std Ops to SPIR-V Ops -----===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert MLIR standard ops into the SPIR-V
// ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

/// A simple pattern for rewriting function signature to convert arguments of
/// functions to be of valid SPIR-V types.
class FuncOpConversion final : public SPIRVOpLowering<FuncOp> {
public:
  using SPIRVOpLowering<FuncOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// A pass converting MLIR Standard operations into the SPIR-V dialect.
class ConvertStandardToSPIRVPass
    : public ModulePass<ConvertStandardToSPIRVPass> {
  void runOnModule() override;
};
} // namespace

PatternMatchResult
FuncOpConversion::matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const {
  auto fnType = funcOp.getType();
  if (fnType.getNumResults()) {
    return matchFailure();
  }

  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  {
    for (auto argType : enumerate(funcOp.getType().getInputs())) {
      auto convertedType = typeConverter.convertType(argType.value());
      signatureConverter.addInputs(argType.index(), convertedType);
    }
  }

  rewriter.updateRootInPlace(funcOp, [&] {
    funcOp.setType(rewriter.getFunctionType(
        signatureConverter.getConvertedTypes(), llvm::None));
    rewriter.applySignatureConversion(&funcOp.getBody(), signatureConverter);
  });
  return matchSuccess();
}

void ConvertStandardToSPIRVPass::runOnModule() {
  OwningRewritePatternList patterns;
  auto context = &getContext();
  auto module = getModule();

  SPIRVTypeConverter typeConverter;
  populateStandardToSPIRVPatterns(context, typeConverter, patterns);
  patterns.insert<FuncOpConversion>(context, typeConverter);
  ConversionTarget target(*(module.getContext()));
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });

  if (failed(applyPartialConversion(module, target, patterns))) {
    return signalPassFailure();
  }
}

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createConvertStandardToSPIRVPass() {
  return std::make_unique<ConvertStandardToSPIRVPass>();
}

static PassRegistration<ConvertStandardToSPIRVPass>
    pass("convert-std-to-spirv", "Convert Standard Ops to SPIR-V dialect");
