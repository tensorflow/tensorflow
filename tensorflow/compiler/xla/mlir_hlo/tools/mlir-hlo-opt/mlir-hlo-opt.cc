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

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/test_passes.h"
#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/passes.h"
#include "lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "mhlo/IR/register.h"
#include "mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/gpu_passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"
#include "thlo/IR/thlo_ops.h"
#include "thlo/transforms/passes.h"

using namespace mlir;

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::hlo::registerLMHLOTransformsPasses();
  mlir::registerLMHLOGPUTransformsPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::thlo::registerAllThloPasses();
  mlir::gml_st::registerGmlStPasses();
  mlir::gml_st::registerGmlStTestPasses();

  struct HloToGpuPipelineOptions
      : public PassPipelineOptions<HloToGpuPipelineOptions> {
    ListOption<int64_t> blockTileDim{
        *this, "block-tile",
        llvm::cl::desc("dimensions of the subproblem processed by the block")};
    ListOption<int64_t> warpTileDim{
        *this, "warp-tile",
        llvm::cl::desc("dimensions of the subproblem processed by the warp")};
    ListOption<int64_t> threadTileDim{
        *this, "thread-tile",
        llvm::cl::desc("dimensions of the subproblem processed by the thread")};
    Option<bool> experimentalSoftmax{
        *this, "experimental-softmax",
        llvm::cl::desc(
            "enable the experimental variant of this pipeline for softmax"),
        llvm::cl::init(false)};
  };
  mlir::PassPipelineRegistration<HloToGpuPipelineOptions>(
      "hlo-to-gpu-pipeline",
      "Pipeline to transform HLO to LLVM + NVVM dialects.",
      [](OpPassManager& pm, const HloToGpuPipelineOptions& opts) {
        return createHloToGpuPipeline(pm, opts.blockTileDim, opts.warpTileDim,
                                      opts.threadTileDim,
                                      opts.experimentalSoftmax);
      });

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::lmhlo::LmhloDialect, mlir::lmhlo_gpu::LmhloGpuDialect,
                  mlir::gml_st::GmlStDialect, mlir::thlo::THLODialect>();

  return failed(
      mlir::MlirOptMain(argc, argv, "MLIR HLO pass driver\n", registry));
}
