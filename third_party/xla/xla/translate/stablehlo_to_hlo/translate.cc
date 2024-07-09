/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/translate/stablehlo_to_hlo/translate.h"

#include <memory>
#include <utility>

#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/translate/mhlo_to_hlo/translate.h"

namespace xla {

mlir::LogicalResult StablehloToHloTranslateFunction(mlir::ModuleOp module,
                                                    llvm::raw_ostream& output,
                                                    bool emit_return_tuple,
                                                    bool emit_use_tuple_arg) {
  if (!module) return mlir::failure();

  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  if (failed(pm.run(module))) {
    module->dump();
    return mlir::failure();
  }

  return xla::MlirHloToHloTranslateFunction(module, output, emit_return_tuple,
                                            emit_use_tuple_arg);
}

mlir::LogicalResult StablehloToHloTextTranslateFunction(
    mlir::ModuleOp module, llvm::raw_ostream& output, bool emit_return_tuple,
    bool emit_use_tuple_arg, bool print_layouts, bool print_large_constants,
    bool print_sugar, bool via_builder, bool with_layouts) {
  if (!module) return mlir::failure();

  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  if (failed(pm.run(module))) {
    module->dump();
    return mlir::failure();
  }

  return xla::MlirHloToHloTextTranslateFunction(
      module, output, emit_return_tuple, emit_use_tuple_arg, print_layouts,
      print_large_constants, print_sugar, via_builder, with_layouts);
}

mlir::LogicalResult StablehloToHloTextMain(
    std::unique_ptr<llvm::MemoryBuffer> buffer,
    llvm::raw_ostream& output_stream, bool emit_return_tuple,
    bool emit_use_tuple_arg, bool print_layouts, bool print_large_constants,
    bool print_sugar, bool via_builder, bool with_layouts) {
  auto source_mgr = std::make_shared<llvm::SourceMgr>();
  source_mgr->AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();

  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(*source_mgr, &context);

  return xla::StablehloToHloTextTranslateFunction(
      *module, output_stream, emit_return_tuple, emit_use_tuple_arg,
      print_layouts, print_large_constants, print_sugar, via_builder,
      with_layouts);
}

}  //  namespace xla
