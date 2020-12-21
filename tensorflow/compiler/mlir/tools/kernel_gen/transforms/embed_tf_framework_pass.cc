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

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/passes.h"
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/rewriters.h"

namespace mlir {
namespace kernel_gen {
namespace tf_framework {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tools/kernel_gen/transforms/kernel_gen_passes.h.inc"

bool IsNotInsideTfEntryFunction(Operation* op) {
  auto func = op->getParentOfType<FuncOp>();
  return !func->hasAttrOfType<UnitAttr>(TFFrameworkDialect::kTFEntryAttrName);
}

// The pass rewrites the function marked with `tf_entry` attribute.
// * adds tf_framework::OpKernelContextType argument to the function,
// * std.alloc becomes tf_framework.alloc_raw,
// * std.dealloc becomes tf_framework.dealloc_raw.
class EmbedTFFrameworkFunctionAndAllocPass
    : public EmbedTFFrameworkFunctionAndAllocPassBase<
          EmbedTFFrameworkFunctionAndAllocPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect>();
  }

 public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Populate patterns.
    OwningRewritePatternList patterns;
    PopulateEmbedTFFrameworkFunctionAndAllocConversionPatterns(m.getContext(),
                                                               &patterns);

    // Set target.
    ConversionTarget target(getContext());
    target.addLegalDialect<tf_framework::TFFrameworkDialect>();

    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      if (!op->hasAttrOfType<UnitAttr>(TFFrameworkDialect::kTFEntryAttrName)) {
        return true;
      }
      FunctionType func_type = op.getType();
      return func_type.getNumInputs() > 0 &&
             func_type.getInput(0).isa<OpKernelContextType>();
    });
    target.addDynamicallyLegalOp<AllocOp, DeallocOp>(
        IsNotInsideTfEntryFunction);

    if (failed(applyPartialConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

// The pass rewrites the function marked with `tf_entry` attribute.
// All contained `std.assert` operations are rewritten into calls to
// `tf_framework.report_error` and the required control flow to make
// execution of the function terminate.

class EmbedTFFrameworkAssertPass
    : public EmbedTFFrameworkAssertPassBase<EmbedTFFrameworkAssertPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::kernel_gen::tf_framework::TFFrameworkDialect>();
  }

 public:
  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Populate patterns.
    OwningRewritePatternList patterns;
    PopulateEmbedTFFrameworkAssertConversionPatterns(m.getContext(), &patterns);

    // Set target.
    ConversionTarget target(getContext());
    target.addLegalDialect<tf_framework::TFFrameworkDialect,
                           StandardOpsDialect>();

    target.addDynamicallyLegalOp<AssertOp>(IsNotInsideTfEntryFunction);

    if (failed(applyPartialConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp> >
CreateEmbedTFFrameworkFunctionAndAllocPass() {
  return std::make_unique<EmbedTFFrameworkFunctionAndAllocPass>();
}

std::unique_ptr<OperationPass<ModuleOp> > CreateEmbedTFFrameworkAssertPass() {
  return std::make_unique<EmbedTFFrameworkAssertPass>();
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
