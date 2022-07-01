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
#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Transforms/GPUPassDetail.h"
#include "mlir-hlo/Transforms/gpu_passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {

namespace {
class GpuFusionRewritePass
    : public GpuFusionRewritePassBase<GpuFusionRewritePass> {
 public:
  explicit GpuFusionRewritePass() = default;
  using Pass::runPipeline;  // Give FusionRewritePattern access.

 private:
  void getDependentDialects(DialectRegistry& registry) const override;
  void runOnOperation() override;
};

// Rewrites `lmhlo.fusion` to `gpu.launch_func` for fusion regions that the
// HLO to GPU pipeline can handle.
class FusionRewritePattern : public OpRewritePattern<lmhlo::FusionOp> {
 public:
  explicit FusionRewritePattern(GpuFusionRewritePass& parentPass,
                                SymbolTable& symbolTable,
                                PassManager& hloToGpuPipeline);

 private:
  LogicalResult matchAndRewrite(lmhlo::FusionOp fusionOp,
                                PatternRewriter& rewriter) const override;

  // Returns whether all ops in fusionOp's region are legal to rewritableTarget.
  bool isRewritable(lmhlo::FusionOp fusionOp) const;

  // Annotates gpu.launch_func with attribute specifying written operands.
  //
  //   gpu.launch_func ..., %memref, ...
  //   %tensor = bufferize.to_tensor %memref
  //   memref.tensor_store %tensor, %argN
  //
  // is replaced with:
  //
  //   gpu.launch_func ..., %argN, ... { written_operands = [..., unit, ...] }
  //
  // The 'written_operands' attribute is used later to retrieve which
  // gpu.launch_func arguments are written vs. just read.
  static void annotateLaunchFunc(func::FuncOp funcOp,
                                 PatternRewriter& rewriter);

  // Returns target where lowerable fusion ops are marked legal.
  static ConversionTarget getRewritableTarget(MLIRContext* ctx);

  GpuFusionRewritePass& parentPass;
  SymbolTable& symbolTable;
  PassManager& hloToGpuPipeline;
  ConversionTarget rewritableTarget = getRewritableTarget(getContext());
};
}  // namespace

// Name of the 'gpu.launch_func' attribute which specifies the written operands.
static constexpr llvm::StringLiteral kWrittenOperandsAttrName =
    "written_operands";

void GpuFusionRewritePass::getDependentDialects(
    DialectRegistry& registry) const {
  OpPassManager passManager;
  createHloToGpuPipeline(passManager, /*tileSizes=*/{}, /*unrollFactors=*/{});
  passManager.getDependentDialects(registry);
}

void GpuFusionRewritePass::runOnOperation() {
  SymbolTable symbolTable(getOperation());
  // Note: passManager.enableIRPrinting() doesn't do anything on dynamic pass
  // pipelines. Printing needs to be enabled on the parent pass manager.
  PassManager passManager(&getContext(), getOperation().getOperationName());
  // TODO(csigg): don't hardcode block size and elements per thread.
  createHloToGpuPipeline(passManager, /*tileSizes=*/256, /*unrollFactors=*/{4});
  auto pattern =
      std::make_unique<FusionRewritePattern>(*this, symbolTable, passManager);
  mlir::FrozenRewritePatternSet patterns({&getContext(), std::move(pattern)});
  auto callback = [&](lmhlo::FusionOp fusion) {
    if (failed(applyOpPatternsAndFold(fusion, patterns)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  };
  if (getOperation().walk(callback).wasInterrupted())
    return signalPassFailure();
}

FusionRewritePattern::FusionRewritePattern(GpuFusionRewritePass& parentPass,
                                           SymbolTable& symbolTable,
                                           PassManager& hloToGpuPipeline)
    : OpRewritePattern<lmhlo::FusionOp>::OpRewritePattern(
          hloToGpuPipeline.getContext()),
      parentPass(parentPass),
      symbolTable(symbolTable),
      hloToGpuPipeline(hloToGpuPipeline) {}

LogicalResult FusionRewritePattern::matchAndRewrite(
    lmhlo::FusionOp fusionOp, PatternRewriter& rewriter) const {
  // If fusion_op (including its region) is not legal by rewriteable_target,
  // we expect lowering to GPU to fail or produce incorrect results.
  if (!isRewritable(fusionOp))
    return rewriter.notifyMatchFailure(fusionOp, "not rewritable");

  // Collect values in fusion region defined above.
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(fusionOp->getRegions(), captures);

  // Create a new module with a function, clone fusion region into it.
  Location loc = fusionOp.getLoc();
  auto moduleOp = rewriter.create<ModuleOp>(loc);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  auto funcType =
      rewriter.getFunctionType(TypeRange(captures.getArrayRef()), llvm::None);
  auto funcOp = rewriter.create<func::FuncOp>(loc, "fusion", funcType);
  rewriter.setInsertionPointToEnd(funcOp.addEntryBlock());
  BlockAndValueMapping mapping;
  for (const auto& [from, to] :
       llvm::zip_first(captures, funcOp.getArguments())) {
    mapping.map(from, to);
  }
  rewriter.cloneRegionBefore(fusionOp.getRegion(), funcOp.getRegion(),
                             funcOp.end(), mapping);
  rewriter.mergeBlocks(&funcOp.back(), &funcOp.front());

  // Run the HLO to GPU pass pipeline.
  if (failed(parentPass.runPipeline(hloToGpuPipeline, moduleOp)))
    return rewriter.notifyMatchFailure(fusionOp, "failed to run pipeline");

  // Clone the (single) gpu module with the device function.
  rewriter.setInsertionPoint(fusionOp->getParentOfType<func::FuncOp>());
  for (auto gpuModuleOp : moduleOp.getBodyRegion().getOps<gpu::GPUModuleOp>()) {
    StringAttr symbol =
        symbolTable.insert(rewriter.clone(*gpuModuleOp.getOperation()));
    if (failed(symbolTable.replaceAllSymbolUses(gpuModuleOp, symbol, funcOp)))
      return rewriter.notifyMatchFailure(fusionOp, "failed to replace symbol");
  }
  // Add 'gpu.container_module' attribute to parent module.
  fusionOp->getParentOfType<ModuleOp>()->setAttr(
      gpu::GPUDialect::getContainerModuleAttrName(), rewriter.getUnitAttr());

  // Annotate gpu.launch_func with attribute specifying written operands.
  annotateLaunchFunc(funcOp, rewriter);

  // Remove dead allocations that were only used by store_op erased above.
  RewritePatternSet patterns(getContext());
  memref::AllocOp::getCanonicalizationPatterns(patterns, getContext());
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
    return rewriter.notifyMatchFailure(fusionOp, "failed to canonicalize");

  // Replace fusion op with host function region.
  rewriter.splitBlock(&funcOp.front(),
                      funcOp.front().getTerminator()->getIterator());
  rewriter.mergeBlockBefore(&funcOp.front(), fusionOp, captures.getArrayRef());

  rewriter.eraseOp(fusionOp);
  rewriter.eraseOp(moduleOp);

  return success();
}

bool FusionRewritePattern::isRewritable(lmhlo::FusionOp fusionOp) const {
  auto callback = [this](Operation* op) {
    if (rewritableTarget.isLegal(op)) return WalkResult::advance();
    return WalkResult::interrupt();
  };
  return !fusionOp.getRegion().walk(callback).wasInterrupted();
}

void FusionRewritePattern::annotateLaunchFunc(func::FuncOp funcOp,
                                              PatternRewriter& rewriter) {
  llvm::SmallDenseMap<Operation*, SmallVector<bool>> writtenOperands;
  funcOp.walk([&](memref::TensorStoreOp storeOp) {
    auto toTensor = storeOp.tensor().getDefiningOp<bufferization::ToTensorOp>();
    assert(toTensor && "not defined by bufferization.to_tensor");
    for (auto& use : toTensor.getMemref().getUses()) {
      Operation* user = use.getOwner();
      if (isa<gpu::LaunchFuncOp>(user)) {
        writtenOperands.try_emplace(user, user->getNumOperands())
            .first->second[use.getOperandNumber()] = true;
        use.set(storeOp.memref());
      }
    }
    rewriter.eraseOp(storeOp);
    rewriter.eraseOp(toTensor);
  });
  for (const auto& [op, vec] : writtenOperands)
    op->setAttr(kWrittenOperandsAttrName, rewriter.getBoolArrayAttr(vec));
}

// Returns whether 'type' is can be lowered by the FusionRewritePattern.
static bool isRewritableType(TensorType type) {
  // Complex types are not yet supported.
  if (type.getElementType().isa<ComplexType>()) return false;
  // Zero ranked shapes are not yet supported.
  if (type.getRank() == 0) return false;
  return true;
}

ConversionTarget FusionRewritePattern::getRewritableTarget(MLIRContext* ctx) {
  ConversionTarget target(*ctx);
  // Mark expected auxiliary ops as legal.
  target.addLegalOp<lmhlo::TerminatorOp>();
  target.addDynamicallyLegalOp<bufferization::ToTensorOp>(
      [&](bufferization::ToTensorOp op) {
        return isRewritableType(op.getType());
      });
  target.addDynamicallyLegalOp<memref::TensorStoreOp>(
      [&](memref::TensorStoreOp op) {
        return isRewritableType(op.tensor().getType().cast<TensorType>());
      });
  // For now, use an explicit allow-list of hlo ops inside the fusion. If any
  // other op is present, the fusion will not be rewritten.
  target.addLegalOp<mhlo::LogOp>();
  target.addLegalOp<mhlo::AbsOp>();
  return target;
}

std::unique_ptr<OperationPass<ModuleOp>> createGpuFusionRewritePass() {
  return std::make_unique<GpuFusionRewritePass>();
}

ArrayAttr getWrittenOperandsAttribute(Operation* op) {
  return op->getAttrOfType<ArrayAttr>(kWrittenOperandsAttrName);
}

}  // namespace mlir
