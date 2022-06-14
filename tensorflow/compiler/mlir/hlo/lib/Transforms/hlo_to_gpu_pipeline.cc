/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

/// This files contains a pipeline which converts HLO operations to GPU kernels
/// written in a combination of LLVM and NVVM dialects.

#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/gpu_passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using ::mlir::func::FuncOp;
using ::mlir::gpu::GPUModuleOp;

// TODO(b/233761238): We only want to have this pipeline temporarily, as it is
// not yet clear how exactly it will look like. The goal is to merge this with
// the unified kernel generator + autofusion + XLA Next pipeline once we have
// it, and once this code stabilizes.
void mlir::createHloToGpuPipeline(OpPassManager &pm,
                                  ArrayRef<int64_t> tileSizes,
                                  ArrayRef<int64_t> unrollFactors) {
  // HLO -> Loops
  pm.addNestedPass<FuncOp>(mhlo::createLegalizeHloToLinalgPass());
  pm.addNestedPass<FuncOp>(createLinalgElementwiseOpFusionPass());
  pm.addNestedPass<FuncOp>(createLinalgInitTensorToAllocTensorPass());
  pm.addPass(CreateComputeOpAndFuncBufferizePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createConvertLinalgToParallelLoopsPass());
  pm.addNestedPass<FuncOp>(bufferization::createBufferDeallocationPass());
  // Loops -> GPU
  pm.addNestedPass<FuncOp>(CreateTileLoopsPass(tileSizes, unrollFactors));
  pm.addNestedPass<FuncOp>(createGpuMapParallelLoopsPass());
  pm.addNestedPass<FuncOp>(createLoopInvariantCodeMotionPass());
  pm.addNestedPass<FuncOp>(createParallelLoopToGpuPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addPass(createGpuLauchSinkIndexComputationsPass());
  constexpr llvm::StringRef kGpuDataLayoutSpec =
      "#dlti.dl_spec<#dlti.dl_entry<index,32:i32>>";
  pm.addPass(createGpuKernelOutliningPass(kGpuDataLayoutSpec));
  pm.addNestedPass<GPUModuleOp>(createLowerAffinePass());
  pm.addNestedPass<GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<GPUModuleOp>(createConvertSCFToCFPass());
  // GPU -> low-level IR
  pm.addNestedPass<GPUModuleOp>(CreateGpuKernelToNvvmPass());
  pm.addPass(CreatePropagateStaticShapesToKernelPass());
}
