/* Copyright 2022 Google Inc. All Rights Reserved.

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

#include "llvm/Support/InitLLVM.h"
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/InitAllPasses.h"  // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/compiler/xla/mlir/framework/ir/xla_framework.h"
#include "tensorflow/compiler/xla/mlir/framework/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_lhlo_with_xla/mhlo_to_lhlo_with_xla.h"
#include "tensorflow/tsl/platform/init_main.h"

int main(int argc, char **argv) {
  // TODO(jreiffers): Move this to a more appropriate place. It is used by both
  // translate/mhlo_to_lhlo_with_xla and mlir/framework for testing.
  llvm::InitLLVM y(argc, argv);
  int dummyArgc = 1;
  tsl::port::InitMain(argv[0], &dummyArgc, &argv);

  mlir::registerAllPasses();
  mlir::RegisterMhloToLhloWithXlaPass();
  mlir::mhlo::registerXlaFrameworkPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::xla_framework::XLAFrameworkDialect>();
  return failed(mlir::MlirOptMain(
      argc, argv, "xla translate test pass driver\n", registry));
}
