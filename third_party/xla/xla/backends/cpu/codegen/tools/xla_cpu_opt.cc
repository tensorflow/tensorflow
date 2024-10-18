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

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "xla/backends/cpu/codegen/ir/xla_cpu_dialect.h"
#include "xla/backends/cpu/codegen/transforms/passes.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, xla::cpu::XlaCpuDialect>();

  // Register builtin MLIR passes.
  mlir::func::registerAllExtensions(registry);
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();

  // Register XLA:CPU passes.
  xla::cpu::registerXlaCpuTransformsPasses();

  return mlir::failed(
      MlirOptMain(argc, argv, "XLA:CPU Pass Driver\n", registry));
}
