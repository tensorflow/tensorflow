/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering LHLO GPU dialect to TFRT CUDA
// dialect.

#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/lmhlo_to_gpu.h"

#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime

namespace tensorflow {

void populateCclConversionPattern(RewritePatternSet&, TypeConverter&);
void populateCholeskyConversionPattern(RewritePatternSet&, TypeConverter&);
void populateConvolutionConversionPattern(RewritePatternSet&, TypeConverter&);
void populateCustomCallConversionPattern(RewritePatternSet&, TypeConverter&);
void populateFftConversionPattern(RewritePatternSet&, TypeConverter&);
void populateGemmConversionPattern(RewritePatternSet&, TypeConverter&);
void populateInfeedAndOutfeedConversionPattern(RewritePatternSet&,
                                               TypeConverter&);
void populateReplicaAndPartitionConversionPattern(RewritePatternSet&,
                                                  TypeConverter&);
void populateTriangularSolveConversionPattern(RewritePatternSet&,
                                              TypeConverter&);

namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/gpu_passes.h.inc"

struct ConvertLmhloToGpuPass
    : public ConvertLmhloToGpuPassBase<ConvertLmhloToGpuPass> {
 private:
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::gpu::GPUDialect, tfrt::compiler::TFRTDialect,
                    tfrt::gpu::GpuDialect,
                    xla::gpu::XlirDialect>();
  }
};

}  // namespace

void ConvertLmhloToGpuPass::runOnOperation() {
  auto* context = &getContext();
  TypeConverter converter = tfrt::gpu::CreateMemrefToTfrtGpuConverter();

  RewritePatternSet patterns(context);
  populateCclConversionPattern(patterns, converter);
  populateCholeskyConversionPattern(patterns, converter);
  populateConvolutionConversionPattern(patterns, converter);
  populateCustomCallConversionPattern(patterns, converter);
  populateGemmConversionPattern(patterns, converter);
  populateInfeedAndOutfeedConversionPattern(patterns, converter);
  populateReplicaAndPartitionConversionPattern(patterns, converter);
  populateTriangularSolveConversionPattern(patterns, converter);
  populateFftConversionPattern(patterns, converter);

  patterns.insert(+[](lmhlo::TerminatorOp op, PatternRewriter& rewriter) {
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, op->getOperands());
    return success();
  });

  tfrt::gpu::PopulateMemrefConversionPatterns(patterns, converter);

  ConversionTarget target(*context);
  target.addIllegalDialect<lmhlo::LmhloDialect, lmhlo_gpu::LmhloGpuDialect>();
  target.addIllegalOp<memref::ReinterpretCastOp, memref::ViewOp,
                      memref::AllocaOp, memref::AllocOp, memref::DeallocOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return converter.isSignatureLegal(op.getFunctionType()) &&
           converter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<tfrt::compiler::CallOp, tfrt::compiler::ReturnOp,
                               tfrt::compiler::WhileOp, func::CallOp,
                               func::ReturnOp>(
      [&](Operation* op) { return converter.isLegal(op); });
  target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertLmhloToGpuPass() {
  return std::make_unique<ConvertLmhloToGpuPass>();
}

void registerConvertLmhloToGpuPass() {
  ::mlir::registerPass([] { return createConvertLmhloToGpuPass(); });
}

}  // namespace tensorflow
