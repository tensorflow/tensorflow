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
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry =
      xla::cpu::FusionCompiler::CreateDialectRegistry(true);

  mlir::registerAllPasses();

  xla::emitters::registerTransformsPasses();
  xla::cpu::registerXlaCpuTransformsPasses();
  xla::cpu::registerXTileCpuTransformsPasses();
  xla::xtile::registerXTileTransformsPasses();
  mlir::stablehlo::registerStablehloLinalgTransformsPasses();

  return mlir::failed(MlirOptMain(
      argc, argv, "XLA:CPU Fusion compiler pass driver\n", registry));
}
