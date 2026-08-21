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
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/xtile/ir/transforms/passes.h"

struct XtileCpuPassOptions
    : public mlir::PassPipelineOptions<XtileCpuPassOptions> {
  Option<bool> fast_min_max{
      *this, "fast_min_max",
      llvm::cl::desc("Whether to enable fast min/max operations."),
      llvm::cl::init(false)};
};

int main(int argc, char** argv) {
  mlir::DialectRegistry registry =
      xla::cpu::FusionCompiler::CreateDialectRegistry();

  mlir::registerAllPasses();
  xla::emitters::registerTransformsPasses();
  xla::cpu::registerXlaCpuTransformsPasses();
  xla::cpu::registerXTileCpuTransformsPasses();
  xla::xtile::registerXTileTransformsPasses();
  mlir::stablehlo::registerStablehloLinalgTransformsPasses();

  mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
      "xtile-cpu-xtile-to-vector",
      "Run the conversion from XTile to Vector dialect.",
      [](mlir::OpPassManager& pm) {
        xla::cpu::AddXtileToVectorPasses(pm, /*msan_enabled=*/false);
      });
  mlir::PassPipelineRegistration<mlir::EmptyPipelineOptions>(
      "xtile-cpu-new-xtile-to-vector",
      "Run the conversion from XTile to Vector dialect.",
      [](mlir::OpPassManager& pm) { xla::cpu::AddNewXtileToVectorPasses(pm); });
  mlir::PassPipelineRegistration<XtileCpuPassOptions>(
      "xtile-cpu-vector-to-llvm",
      "Run the conversion from Vector to LLVM dialect.",
      [](mlir::OpPassManager& pm, const XtileCpuPassOptions& options) {
        xla::cpu::AddVectorToLLVMPasses(pm, options.fast_min_max);
      });
  return mlir::failed(MlirOptMain(
      argc, argv, "XLA:CPU Fusion compiler pass driver\n", registry));
}
