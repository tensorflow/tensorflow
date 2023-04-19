/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "deallocation/IR/deallocation_ops.h"  // IWYU pragma: keep
#include "deallocation/transforms/passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"  // IWYU pragma: keep
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"      // IWYU pragma: keep
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // IWYU pragma: keep
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"  // IWYU pragma: keep
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "transforms/passes.h"

namespace mlir {

#define GEN_PASS_DEF_GENERICHOSTTOLLVMPASS
#include "transforms/passes.h.inc"

namespace {

class GenericHostToLLVMPass
    : public impl::GenericHostToLLVMPassBase<GenericHostToLLVMPass> {
 public:
  using GenericHostToLLVMPassBase::GenericHostToLLVMPassBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Populate type conversions.
    MLIRContext* ctx = m.getContext();
    LLVMTypeConverter typeConverter(ctx);

    {
      // Perform progressive lowering of vector operations on slices and all
      // vector contraction operations. Also applies folding and DCE.

      RewritePatternSet patterns(&getContext());
      vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      vector::populateVectorContractLoweringPatterns(
          patterns, vector::VectorTransformsOptions());
      vector::populateVectorMaskOpLoweringPatterns(patterns);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      vector::populateVectorTransposeLoweringPatterns(
          patterns, vector::VectorTransformsOptions());
      // Vector transfer ops with rank > 1 should be lowered with VectorToSCF.
      vector::populateVectorTransferLoweringPatterns(patterns,
                                                     /*maxTransferRank=*/1);
      (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    }

    LLVMConversionTarget target(*ctx);

    // Populate patterns.
    RewritePatternSet patterns(&getContext());
    populateAffineToStdConversionPatterns(patterns);
    arith::populateArithExpandOpsPatterns(patterns);
    memref::populateExpandOpsPatterns(patterns);
    memref::populateExpandStridedMetadataPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns, false);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    populateComplexToLLVMConversionPatterns(typeConverter, patterns);
    populateLinalgToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLibmConversionPatterns(patterns);
    deallocation::populateDeallocationToLLVMConversionPatterns(typeConverter,
                                                               patterns);

    // Vector patterns.
    vector::populateVectorMaskMaterializationPatterns(patterns, true);
    vector::populateVectorTransferLoweringPatterns(patterns);
    populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
    populateVectorToLLVMConversionPatterns(typeConverter, patterns);
    if (enableAvx2) {
      configureX86VectorLegalizeForExportTarget(target);
      populateX86VectorLegalizeForLLVMExportPatterns(typeConverter, patterns);
    }

    //  Setup target.
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<arith::ArithDialect, func::FuncDialect,
                             complex::ComplexDialect, math::MathDialect>();
    // Mark modules as legal.
    target.addLegalOp<ModuleOp>();
    // Unrealized conversion casts are cleaned up by a separate pass.
    target.addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace hlo {

std::unique_ptr<OperationPass<ModuleOp>> createGenericHostToLLVMPass(
    const GenericHostToLLVMPassOptions& options) {
  return std::make_unique<GenericHostToLLVMPass>(options);
}

}  // namespace hlo
}  // namespace mlir
