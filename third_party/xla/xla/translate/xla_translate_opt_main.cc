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

#include "llvm/Support/LogicalResult.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"
#include "xla/mlir/framework/ir/xla_framework.h"
#include "xla/mlir/framework/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "tsl/platform/init_main.h"

int main(int argc, char **argv) {
  int dummyArgc = 1;
  tsl::port::InitMain(argv[0], &dummyArgc, &argv);

  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::xla_framework::registerXlaFrameworkPasses();
  registry.insert<mlir::xla_framework::XLAFrameworkDialect>();
  return failed(mlir::MlirOptMain(
      argc, argv, "xla translate test pass driver\n", registry));
}
