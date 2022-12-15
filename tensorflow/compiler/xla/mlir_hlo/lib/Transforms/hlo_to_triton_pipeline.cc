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

/// This files contains a pipeline which converts HLO operations to Triton
/// kernels written in Triton Dialect.
/// TODO(b/261710844): Define and actually generate Triton dialect.

#include "gml_st/transforms/passes.h"
#include "mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/gpu_passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
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

static constexpr const char* kBlockDistributionLabel = "block";

void mlir::createHloToTritonPipeline(OpPassManager& pm,
                                     ArrayRef<int64_t> blockTileDim) {
  pm.addNestedPass<FuncOp>(hlo::createUnbufferizePass());
  pm.addPass(createCanonicalizerPass());  // Clean up get_tuple_element.
  pm.addPass(createCSEPass());  // Combine repeated subtract(broadcast).

  // HLO -> Linalg
  pm.addNestedPass<FuncOp>(mhlo::createChloLegalizeToHloPass());
  pm.addPass(createCanonicalizerPass());  // Clean up shape.assuming ops.
  pm.addNestedPass<FuncOp>(
      mhlo::createLegalizeHloToLinalgPass(/*enablePrimitiveOps=*/true));

  // Tile softmax-like pattern

  // Simplify unit dimension.
  pm.addPass(mlir::createLinalgFoldUnitExtentDimsPass());
  // Collapse all but the trailing reduction/bcast dimension.
  pm.addNestedPass<FuncOp>(
      gml_st::createCollapseShapePass({/*retainTrailingDims=*/1}));
  // Merge multiple occurences of collapsed operand. This is needed to detect
  // the softmax pattern later.
  pm.addNestedPass<FuncOp>(mlir::createCSEPass());

  // Tile parallel dimensions of the softmax-like patterns and distribute them
  // across warps. Warps remain independant of each other.
  pm.addNestedPass<FuncOp>(gml_st::createGreedyTilingAndFusionPass(
      /*distribute=*/true, blockTileDim, kBlockDistributionLabel));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // TODO(b/261711206): Pad to power-of-2

  pm.addNestedPass<FuncOp>(gml_st::createVectorizeGmlStLoopsPass());

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Bufferization-related passes
  pm.addNestedPass<FuncOp>(bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(hlo::createOneShotBufferizePass());
  // We do not deallocate buffers, since grid-level buffers get converted into
  // functions arguments, while block- (and lower-)level buffers become shared
  // memory. None of which have to be deallocated.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Canonicalize away memory copies into itself.
  pm.addPass(createCanonicalizerPass());

  // GmlSt -> Triton
  pm.addNestedPass<FuncOp>(
      gml_st::createGmlStSimtfyPass(kBlockDistributionLabel));
  pm.addNestedPass<FuncOp>(gml_st::createGmlStToGpuPass());
  pm.addNestedPass<FuncOp>(gml_st::createGmlStToScfPass());
  pm.addNestedPass<FuncOp>(arith::createArithExpandOpsPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addPass(createGpuLauchSinkIndexComputationsPass());
  pm.addPass(createGpuKernelOutliningPass());
  pm.addNestedPass<GPUModuleOp>(createForLoopSpecializationPass());
  pm.addNestedPass<GPUModuleOp>(hlo::createUnrollLoopsPass());
  pm.addNestedPass<GPUModuleOp>(createLowerAffinePass());
  pm.addNestedPass<GPUModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<GPUModuleOp>(createConvertSCFToCFPass());

  /// TODO(b/261710844): Define and actually generate Triton dialect.

  // This is added as a global (instead of nested) pass to also remove duplicate
  // constants on the host side of the code.
  pm.addPass(createCSEPass());
  pm.addNestedPass<FuncOp>(hlo::createAllocToArgPass());
}
