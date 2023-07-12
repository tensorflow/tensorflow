/* Copyright 2023 The JAX Authors.

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

#include "tensorflow/compiler/xla/python/refine_polymorphic_shapes.h"

#include "absl/status/status.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Bytecode/BytecodeWriter.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/transforms/Passes.h"  // from @stablehlo
#include "tensorflow/compiler/xla/mlir/utils/error_util.h"
#include "tensorflow/tsl/platform/errors.h"

namespace xla {

absl::Status RefinePolymorphicShapes(mlir::ModuleOp module) {
  mlir::MLIRContext *context = module->getContext();
  if (VLOG_IS_ON(3)) context->disableMultithreading();

  // Verify the module before running passes on it.
  // If the module doesn't pass verification, all sorts of weirdness might
  // happen if we run the pass manager.
  mlir::BaseScopedDiagnosticHandler diag_handler(context);

  if (mlir::failed(mlir::verify(module))) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module verification failed: ",
                     diag_handler.ConsumeStatus().ToString()));
  }

  mlir::PassManager pm(context);
  if (VLOG_IS_ON(3)) {
    auto print_before = [](mlir::Pass *, mlir::Operation *) { return true; };
    auto print_after = [](mlir::Pass *, mlir::Operation *) { return true; };
    pm.enableIRPrinting(print_before, print_after, /*printModuleScope=*/true,
                        /*printAfterOnlyOnChange=*/true);
  }

  // TODO(necula): we should not need the inliner.
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::stablehlo::createStablehloRefineShapesPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createStablehloCanonicalizeDynamismPass());
  if (!mlir::succeeded(pm.run(module))) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module shape refinement failed: ",
                     diag_handler.ConsumeStatus().ToString()));
  }
  return ValidateStaticShapes(module);
}

absl::Status RefinePolymorphicShapes(llvm::StringRef module_str,
                                     llvm::raw_ostream &os) {
  mlir::MLIRContext context;
  if (VLOG_IS_ON(3)) context.disableMultithreading();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::stablehlo::StablehloDialect>();
  context.loadDialect<mlir::chlo::ChloDialect>();

  mlir::DialectRegistry registry;
  mlir::func::registerAllExtensions(registry);
  context.appendDialectRegistry(registry);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(
          llvm::StringRef(module_str.data(), module_str.size()), &context);
  if (!module) {
    return absl::InvalidArgumentError("Cannot parse module.");
  }
  TF_RETURN_IF_ERROR(RefinePolymorphicShapes(*module));
  if (mlir::failed(mlir::writeBytecodeToFile(*module, os))) {
    return absl::InternalError("Cannot serialize module.");
  }
  return absl::OkStatus();
}

absl::Status ValidateStaticShapes(mlir::ModuleOp module) {
  mlir::BaseScopedDiagnosticHandler diag_handler(module->getContext());
  bool moduleHasDynamicShapes = false;

  module->walk([&](mlir::Operation *op) {
    // It's sufficient to only check results because operands either come from
    // results or from block arguments which are checked below.
    auto hasDynamicShape = [](mlir::Value value) {
      auto shaped_type = value.getType().dyn_cast<mlir::ShapedType>();
      return shaped_type ? !shaped_type.hasStaticShape() : false;
    };
    bool opHasDynamicShapes = false;
    opHasDynamicShapes |= llvm::any_of(op->getResults(), hasDynamicShape);
    for (mlir::Region &region : op->getRegions()) {
      opHasDynamicShapes |=
          llvm::any_of(region.getArguments(), hasDynamicShape);
    }
    if (opHasDynamicShapes) {
      moduleHasDynamicShapes = true;
      op->emitOpError() << "has dynamic shapes";
    }
  });

  if (moduleHasDynamicShapes) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module has dynamic shapes: ",
                     diag_handler.ConsumeStatus().ToString()));
  }
  return absl::OkStatus();
}

}  // namespace xla
