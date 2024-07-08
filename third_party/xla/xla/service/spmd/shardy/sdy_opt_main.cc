/* Copyright 2024 The OpenXLA Authors.

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

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/InitAllPasses.h"  // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "third_party/openxla/shardy/src/shardy/dialect/sdy/ir/dialect.h"
#include "third_party/openxla/shardy/src/shardy/dialect/sdy/transforms/passes.h"
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

int main(int argc, char** argv) {
  mlir::registerAllPasses();
  mlir::mhlo::registerAllMhloPasses();

  mlir::DialectRegistry dialects;
  dialects.insert<mlir::func::FuncDialect, mlir::mhlo::MhloDialect,
                  mlir::sdy::SdyDialect, mlir::stablehlo::StablehloDialect>();
  mlir::func::registerAllExtensions(dialects);

  // Register all SDY passes and pipelines.
  mlir::sdy::registerAllSdyPassesAndPipelines();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "XLA SDY pass driver\n", dialects));
}
