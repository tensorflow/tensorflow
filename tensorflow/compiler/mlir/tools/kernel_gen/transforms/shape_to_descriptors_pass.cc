/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file combines patterns for lowering shape dialect to standard ops,
// structured control flow and descriptors.

#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Shape/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_DEF_SHAPETODESCRIPTORSPASS
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct ShapeToDescriptorsPass
    : public impl::ShapeToDescriptorsPassBase<ShapeToDescriptorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

 public:
  void runOnOperation() override {
    MLIRContext &ctx = getContext();

    // Setup target legality.
    ConversionTarget target(ctx);
    target.addIllegalDialect<shape::ShapeDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<math::MathDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    // Don't mark the primary Cstr/Assuming ops as illegal, so they can be
    // lowered at a later time to assertions.
    target.addLegalOp<shape::AssumingOp, shape::AssumingYieldOp,
                      shape::AssumingAllOp, shape::CstrRequireOp>();

    // Setup conversion patterns.
    RewritePatternSet patterns(&getContext());
    populateShapeRewritePatterns(patterns);
    populateShapeToStandardConversionPatterns(patterns);

    // Apply conversion.
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp> > CreateShapeToDescriptorsPass() {
  return std::make_unique<ShapeToDescriptorsPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
