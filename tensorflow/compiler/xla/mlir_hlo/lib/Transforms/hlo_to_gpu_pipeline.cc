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

#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/gpu_passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
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
  pm.addNestedPass<FuncOp>(hlo::createUnbufferizePass());

  // HLO -> Loops
  pm.addNestedPass<FuncOp>(mhlo::createLegalizeHloToLinalgPass());
  SmallVector<SmallVector<int64_t>> tilingSizes = {
      SmallVector<int64_t>(tileSizes.begin(), tileSizes.end()),
      SmallVector<int64_t>(unrollFactors.begin(), unrollFactors.end()),
      // Force the innermost ploop to be a point to avoid temp alloc()s.
      // TODO(csigg): vectorize instead.
      SmallVector<int64_t>(tileSizes.size(), 1)};

  pm.addNestedPass<FuncOp>(gml_st::createDeprecatedTilingPass(tilingSizes));
  pm.addNestedPass<FuncOp>(gml_st::createDeprecatedFusionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<FuncOp>(gml_st::createComposeSetOpsPass());

  // Bufferization-related passes.
  pm.addNestedPass<FuncOp>(createLinalgInitTensorToAllocTensorPass());
  pm.addPass(hlo::createOneShotBufferizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<FuncOp>(bufferization::createBufferDeallocationPass());

  // Convert Linalg + GmlSt to SCF loops.
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.addNestedPass<FuncOp>(gml_st::createVectorizeGmlStLoopsPass());
  pm.addNestedPass<FuncOp>(gml_st::createGmlStToScfPass());

  // Loops -> GPU
  pm.addNestedPass<FuncOp>(createGpuMapParallelLoopsPass());
  pm.addNestedPass<FuncOp>(createLoopInvariantCodeMotionPass());
  pm.addNestedPass<FuncOp>(createParallelLoopToGpuPass());
  pm.addNestedPass<FuncOp>(arith::createArithmeticExpandOpsPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addPass(createGpuLauchSinkIndexComputationsPass());
  constexpr llvm::StringRef kGpuDataLayoutSpec =
      "#dlti.dl_spec<#dlti.dl_entry<index,32:i32>>";
  pm.addPass(createGpuKernelOutliningPass(kGpuDataLayoutSpec));
  pm.addNestedPass<GPUModuleOp>(createForLoopSpecializationPass());
  pm.addNestedPass<GPUModuleOp>(createLowerAffinePass());
  pm.addNestedPass<GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<GPUModuleOp>(createConvertSCFToCFPass());
  // GPU -> low-level IR
#if TENSORFLOW_USE_ROCM
  pm.addNestedPass<GPUModuleOp>(createGpuKernelToRocdlPass());
#else
  pm.addNestedPass<GPUModuleOp>(createGpuKernelToNvvmPass());
#endif
  pm.addPass(createPropagateStaticShapesToKernelPass());
  // Some instructions crash ptxas down the line if they have debug info
  // attached.
  pm.addNestedPass<GPUModuleOp>(createStripDebugInfoPass());
  pm.addNestedPass<FuncOp>(hlo::createAllocToArgPass());
}
