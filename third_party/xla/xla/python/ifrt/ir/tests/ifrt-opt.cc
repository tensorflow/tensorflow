/* Copyright 2023 The OpenXLA Authors.

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

#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Tools/mlir-opt/MlirOptMain.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/transforms/built_in_spmd_expansions.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  registry.insert<xla::ifrt::IfrtDialect>();
  xla::ifrt::registerIfrtIrPasses();

  xla::ifrt::AttachBuiltInSpmdExpansions(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "IFRT dialect driver\n", registry));
}
