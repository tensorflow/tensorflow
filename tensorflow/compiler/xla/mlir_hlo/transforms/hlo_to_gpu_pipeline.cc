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

#include <memory>

#include "gml_st/transforms/passes.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "transforms/gpu_passes.h"
#include "transforms/passes.h"

using namespace mlir;
using ::mlir::func::FuncOp;
using ::mlir::gpu::GPUModuleOp;

static constexpr const char *kBlockDistributionLabel = "block";
static constexpr const char *kWarpDistributionLabel = "warp";
static constexpr const char *kThreadDistributionLabel = "thread";

namespace {

SmallVector<Operation *> detectSinkableTransferRead(
    vector::TransferReadOp transferReadOp) {
  SmallVector<Operation *> currentOps;
  currentOps.push_back(transferReadOp);
  for (Value index : transferReadOp.getIndices()) {
    auto indexDefOp = index.getDefiningOp<arith::ConstantOp>();
    if (!indexDefOp) return {};
    currentOps.push_back(indexDefOp);
  }
  auto paddingDefOp =
      transferReadOp.getPadding().getDefiningOp<arith::ConstantOp>();
  if (!paddingDefOp) return {};
  currentOps.push_back(paddingDefOp);
  if (currentOps.size() != transferReadOp.getTransferRank() + 2) return {};
  return currentOps;
}

SmallVector<Operation *> detectSinkableScalar(
    Operation *scalarOp, const SmallVector<Operation *> &opsToSink) {
  SmallVector<Operation *> currentOps;
  currentOps.push_back(scalarOp);
  for (Value operand : scalarOp->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) return {};

    if (isa<arith::ConstantOp, memref::LoadOp>(defOp) ||
        llvm::is_contained(opsToSink, defOp)) {
      currentOps.push_back(defOp);
    }
  }
  if (currentOps.size() != scalarOp->getNumOperands() + 1) return {};
  return currentOps;
}

LogicalResult sinkOperationsIntoLaunchOp(gpu::LaunchOp launchOp) {
  Region &launchOpBody = launchOp.getBody();

  // Identify uses from values defined outside of the scope of the launch
  // operation.
  SetVector<Value> valuesFromAbove;
  getUsedValuesDefinedAbove(launchOpBody, valuesFromAbove);

  SetVector<Operation *> producers;
  for (Value value : valuesFromAbove) {
    Operation *operandOp = value.getDefiningOp();
    if (!operandOp) continue;
    producers.insert(operandOp);
  }

  // Find operations that can be sunk into the scope.
  SmallVector<Operation *> opsToSink;
  bool foundSinkableOps = true;
  while (foundSinkableOps && !producers.empty()) {
    foundSinkableOps = false;
    SmallVector<Operation *> producersToDrop;
    for (Operation *producer : producers) {
      if (producer->getNumResults() != 1) {
        producersToDrop.push_back(producer);
      }
      if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(producer)) {
        SmallVector<Operation *> vectorTransferReadDAG =
            detectSinkableTransferRead(transferReadOp);
        if (!vectorTransferReadDAG.empty()) {
          for (Operation *op : llvm::reverse(vectorTransferReadDAG)) {
            opsToSink.push_back(op);
          }
          foundSinkableOps = true;
        }
        producersToDrop.push_back(producer);
      }
      if (!isa<ShapedType>(producer->getResult(0).getType())) {
        SmallVector<Operation *> scalarOpDAG =
            detectSinkableScalar(producer, opsToSink);
        if (!scalarOpDAG.empty()) {
          for (Operation *op : llvm::reverse(scalarOpDAG)) {
            opsToSink.push_back(op);
          }
          foundSinkableOps = true;
          producersToDrop.push_back(producer);
        }
      }
    }
    for (Operation *producerToDrop : producersToDrop)
      producers.remove(producerToDrop);
  }

  // Insert operations so that the defs get cloned before uses.
  mlir::IRMapping map;
  OpBuilder builder(launchOpBody);
  for (Operation *op : opsToSink) {
    Operation *clonedOp = builder.clone(*op, map);
    // Only replace uses within the launch op.
    for (auto pair : llvm::zip(op->getResults(), clonedOp->getResults())) {
      replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair),
                                 launchOp.getBody());
    }
  }
  return success();
}

}  // namespace

/// Pass that moves ops which are likely an index computation into gpu.launch
/// body.
class GPUSinkVectorArgsInGPULaunchPass
    : public mlir::PassWrapper<GPUSinkVectorArgsInGPULaunchPass,
                               mlir::OperationPass<> > {
 public:
  void runOnOperation() override {
    Operation *op = getOperation();
    if (op->walk([](gpu::LaunchOp launch) {
            // Pull in instructions that can be sunk
            if (failed(sinkOperationsIntoLaunchOp(launch)))
              return WalkResult::interrupt();

            return WalkResult::advance();
          }).wasInterrupted())
      signalPassFailure();
  }
};

// TODO(b/233761238): We only want to have this pipeline temporarily, as it is
// not yet clear how exactly it will look like. The goal is to merge this with
// the unified kernel generator + autofusion + XLA Next pipeline once we have
// it, and once this code stabilizes.
void mlir::createHloToGpuPipeline(OpPassManager &pm,
                                  ArrayRef<int64_t> blockTileDim,
                                  ArrayRef<int64_t> warpTileDim,
                                  ArrayRef<int64_t> threadTileDim,
                                  bool experimentalSoftmax) {
  pm.addNestedPass<FuncOp>(hlo::createUnbufferizePass());
  pm.addPass(createCanonicalizerPass());  // Clean up get_tuple_element.
  pm.addPass(createCSEPass());  // Combine repeated subtract(broadcast).

  // HLO -> Linalg
  pm.addNestedPass<FuncOp>(mhlo::createChloLegalizeToHloPass());
  pm.addPass(createCanonicalizerPass());  // Clean up shape.assuming ops.
  // Tiling either for softmax or for elementwise
  if (experimentalSoftmax) {
    pm.addNestedPass<FuncOp>(
        mhlo::createLegalizeHloToLinalgPass(/*enablePrimitiveOps=*/true));

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
    pm.addNestedPass<FuncOp>(gml_st::createGreedyFusionPass(
        /*distribute=*/true, blockTileDim, kBlockDistributionLabel));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<FuncOp>(gml_st::createGreedyFusionPass(
        /*distribute=*/true, warpTileDim, kWarpDistributionLabel));
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());

    // GPU-specific tiling for ops on the warp level.
    pm.addNestedPass<FuncOp>(gml_st::createTilingGpuWarpPass());
    pm.addNestedPass<FuncOp>(gml_st::createScalarizationPass());

    pm.addNestedPass<FuncOp>(gml_st::createVectorizeForGPUPass(
        /*vectorizeGmlStOps=*/true, /*distributionLabels=*/{
            kWarpDistributionLabel, kThreadDistributionLabel}));
  } else {
    pm.addNestedPass<FuncOp>(
        mhlo::createLegalizeHloToLinalgPass(/*enablePrimitiveOps=*/false));

    pm.addNestedPass<FuncOp>(gml_st::createTilingCwisePass(
        /*distribute=*/true, blockTileDim, kBlockDistributionLabel));
    pm.addNestedPass<FuncOp>(gml_st::createTilingCwisePass(
        /*distribute=*/true, warpTileDim, kWarpDistributionLabel));
    pm.addNestedPass<FuncOp>(gml_st::createTilingCwisePass(
        /*distribute=*/true, threadTileDim, kThreadDistributionLabel));
    // Convert the inner dimension into a sequential loop over all elements.
    pm.addNestedPass<FuncOp>(gml_st::createTilingCwisePass(
        /*distribute=*/false, /*tileSizes=*/1));
    pm.addNestedPass<FuncOp>(gml_st::createScalarizationPass());
  }

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

  // GmlSt -> GPU
  pm.addNestedPass<FuncOp>(
      gml_st::createGmlStSimtfyPass(kBlockDistributionLabel));
  pm.addNestedPass<FuncOp>(
      gml_st::createGmlStToGpuPass(kWarpDistributionLabel));
  pm.addNestedPass<FuncOp>(gml_st::createGmlStToScfPass());
  pm.addNestedPass<FuncOp>(arith::createArithExpandOpsPass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addPass(std::make_unique<GPUSinkVectorArgsInGPULaunchPass>());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addNestedPass<FuncOp>(mlir::createGpuLauchSinkIndexComputationsPass());

  constexpr llvm::StringRef kGpuDataLayoutSpec =
      "#dlti.dl_spec<#dlti.dl_entry<index,32:i32>>";
  pm.addPass(createGpuKernelOutliningPass(kGpuDataLayoutSpec));
  pm.addNestedPass<GPUModuleOp>(createForLoopSpecializationPass());
  pm.addNestedPass<GPUModuleOp>(hlo::createUnrollLoopsPass());
  // Fold loads from subviews to optimize index computations.
  pm.addNestedPass<GPUModuleOp>(memref::createFoldMemRefAliasOpsPass());
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
  // This is added as a global (instead of nested) pass to also remove duplicate
  // constants on the host side of the code.
  pm.addPass(createCSEPass());
  // Some instructions crash ptxas down the line if they have debug info
  // attached.
  pm.addNestedPass<GPUModuleOp>(createStripDebugInfoPass());
  pm.addNestedPass<FuncOp>(hlo::createAllocToArgPass());
}
