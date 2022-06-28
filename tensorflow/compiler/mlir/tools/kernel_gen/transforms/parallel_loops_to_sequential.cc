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

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"

namespace mlir {
namespace kernel_gen {
namespace transforms {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

struct ParallelLoopsToSequentialPass
    : public ParallelLoopsToSequentialBase<ParallelLoopsToSequentialPass> {
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    mlir::populateSCFToControlFlowConversionPatterns(patterns);

    mlir::ConversionTarget target(getContext());
    target.addIllegalOp<mlir::scf::ParallelOp>();
    target.addLegalOp<mlir::scf::ForOp, mlir::scf::IfOp>();
    target.markUnknownOpDynamicallyLegal([](mlir::Operation*) { return true; });
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateParallelLoopsToSequential() {
  return std::make_unique<ParallelLoopsToSequentialPass>();
}

}  // namespace transforms
}  // namespace kernel_gen
}  // namespace mlir
