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
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "llvm/ADT/STLExtras.h"
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
  explicit FusionRewritePattern(MLIRContext* ctx,
                                GpuFusionRewritePass& parentPass,
                                SymbolTable& symbolTable);

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
  auto pattern =
      std::make_unique<FusionRewritePattern>(&getContext(), *this, symbolTable);
  mlir::FrozenRewritePatternSet patterns({&getContext(), std::move(pattern)});
  auto callback = [&](lmhlo::FusionOp fusion) {
    if (failed(applyOpPatternsAndFold(fusion, patterns)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  };
  if (getOperation().walk(callback).wasInterrupted())
    return signalPassFailure();
}

FusionRewritePattern::FusionRewritePattern(MLIRContext* ctx,
                                           GpuFusionRewritePass& parentPass,
                                           SymbolTable& symbolTable)
    : OpRewritePattern<lmhlo::FusionOp>::OpRewritePattern(ctx),
      parentPass(parentPass),
      symbolTable(symbolTable) {}

// Returns the number of elements each thread should handle for 'type'.
// The intention is that loads and stores are vectorized later on to this width
// to maximize memory throughput.
static int64_t getElementsPerThread(TensorType type) {
  // Don't vectorize if the number of elements cannot saturate the GPU.
  // Use a coarse heuristic because we don't know the target GPU here.
  const int64_t kNumFp32AlusOnV100 = 5376;
  if (type.getNumElements() < kNumFp32AlusOnV100) return 1;

  // Vectorize so that loads and stores are 128 bits per thread.
  if (type.getElementType().isIntOrFloat())
    return 128 / type.getElementType().getIntOrFloatBitWidth();

  return 1;  // Default to no vectorization.
}

// Returns the number of threads per block to use for 'type', given the number
// of elements each thread handles. The returned block size is in the [128, 384]
// range, preferrably close to 256 and evenly dividing the number of threads
// required to handle all elements in 'type'.
static int64_t getThreadsPerBlock(TensorType type, int64_t elementsPerThread) {
  int64_t numThreads =
      llvm::divideCeil(type.getNumElements(), elementsPerThread);

  // Use a single block for small problems.
  if (numThreads < 256) return numThreads;

  // Use 256 if that block size evenly divides the problem.
  if (numThreads % 256 == 0) return 256;

  int64_t elementSizeBits = 32;
  if (type.getElementType().isIntOrFloat())
    elementSizeBits = type.getElementType().getIntOrFloatBitWidth();
  int64_t threadSizeBits = elementSizeBits * elementsPerThread;

  // Search block sizes in the [128, 384] range near 256 with decreasing
  // power-of-2 factor, down to a multiple of a cache line (assumed to be 1024
  // bits). Use the first one that evenly divides the problem, which allows the
  // loop tail to be optimized away.
  for (int i = 128; i * threadSizeBits >= 1024; i /= 2) {
    // 2 * i: earlier iterations already handled even multiples of i.
    for (int blockSize = 256 - i; blockSize >= 128; blockSize -= 2 * i)
      if (numThreads % blockSize == 0) return blockSize;
    for (int blockSize = 256 + i; blockSize <= 384; blockSize += 2 * i)
      if (numThreads % blockSize == 0) return blockSize;
  }

  // None of the checked block sizes evenly divides the number of required
  // threads. Use a default of 256 and accept the loop tail.
  return 256;
}

LogicalResult FusionRewritePattern::matchAndRewrite(
    lmhlo::FusionOp fusionOp, PatternRewriter& rewriter) const {
  // If fusion_op (including its region) is not legal by rewriteable_target,
  // we expect lowering to GPU to fail or produce incorrect results.
  if (!isRewritable(fusionOp))
    return rewriter.notifyMatchFailure(fusionOp, "not rewritable");

  auto storeOps = fusionOp.getBody()->getOps<memref::TensorStoreOp>();
  if (storeOps.empty())
    return rewriter.notifyMatchFailure(fusionOp, "no memref.tensor_store ops");

  // Collect values in fusion region defined above.
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(fusionOp->getRegions(), captures);

  // Converts statically shaped types to their 1D equivalent. This only works
  // for element wise fusions and will have to become a more sophisticated
  // pass when e.g. broadcasts are involved.
  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });
  converter.addConversion([](ShapedType type) {
    if (!type.hasStaticShape()) return type;
    return type.clone(type.getNumElements());
  });
  converter.addConversion([&](MemRefType type) {
    if (!type.hasStaticShape() || !type.getLayout().isIdentity()) return type;
    return MemRefType::get(type.getNumElements(), type.getElementType(),
                           MemRefLayoutAttrInterface(), type.getMemorySpace());
  });

  // Create a new module with a function, clone fusion region into it.
  Location loc = fusionOp.getLoc();
  auto moduleOp = rewriter.create<ModuleOp>(loc);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());
  auto argTypes = llvm::to_vector(llvm::map_range(captures, [&](Value value) {
    return converter.convertType(value.getType());
  }));
  auto funcType = rewriter.getFunctionType(argTypes, llvm::None);
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
  funcOp->walk([&](Operation* op) {
    for (auto result : op->getResults())
      result.setType(converter.convertType(result.getType()));
  });

  // Create and run the HLO to GPU pass pipeline.
  auto resultType = (*storeOps.begin()).tensor().getType().cast<TensorType>();
  int64_t unrollFactor = getElementsPerThread(resultType);
  int64_t tileSize = getThreadsPerBlock(resultType, unrollFactor);
  // Note: passManager.enableIRPrinting() doesn't do anything on dynamic pass
  // pipelines. Printing needs to be enabled on the parent pass manager.
  PassManager passManager(getContext());
  createHloToGpuPipeline(passManager, {tileSize},
                         {&unrollFactor, unrollFactor > 1});
  if (failed(parentPass.runPipeline(passManager, moduleOp)))
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

  // Annotate gpu.launch_func loc and attribute specifying written operands.
  funcOp->walk([&](gpu::LaunchFuncOp op) { op->setLoc(loc); });
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
  if (fusionOp.getFusionResults().size() > 1)
    return false;  // Do not rewrite fusions with multiple outputs.
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
    auto toTensor =
        storeOp.getTensor().getDefiningOp<bufferization::ToTensorOp>();
    assert(toTensor && "not defined by bufferization.to_tensor");
    for (auto& use : toTensor.getMemref().getUses()) {
      Operation* user = use.getOwner();
      if (isa<gpu::LaunchFuncOp>(user)) {
        writtenOperands.try_emplace(user, user->getNumOperands())
            .first->second[use.getOperandNumber()] = true;
        use.set(storeOp.getMemref());
      }
    }
    rewriter.eraseOp(storeOp);
    rewriter.eraseOp(toTensor);
  });
  for (const auto& [op, vec] : writtenOperands)
    op->setAttr(kWrittenOperandsAttrName, rewriter.getBoolArrayAttr(vec));
}

// Returns whether 'type' is can be lowered by the FusionRewritePattern.
static bool isRewritableType(Type type) {
  auto shapedType = type.cast<ShapedType>();
  // Complex types are not yet supported.
  if (shapedType.getElementType().isa<ComplexType>()) return false;
  // Zero ranked shapes are not yet supported.
  if (shapedType.getRank() == 0) return false;
  // MemRef types need to have identity layout.
  if (auto memrefType = shapedType.dyn_cast<MemRefType>())
    return memrefType.getLayout().isIdentity();
  // Unsigned integers are not yet supported.
  if (auto intType = shapedType.getElementType().dyn_cast<IntegerType>())
    return !intType.isUnsigned();
  return true;
}

ConversionTarget FusionRewritePattern::getRewritableTarget(MLIRContext* ctx) {
  ConversionTarget target(*ctx);
  // Mark expected auxiliary ops as legal.
  target.addLegalOp<lmhlo::TerminatorOp>();
  target.addDynamicallyLegalOp<bufferization::ToTensorOp>(
      [&](bufferization::ToTensorOp op) {
        return isRewritableType(op.getMemref().getType()) &&
               isRewritableType(op.getType());
      });
  target.addDynamicallyLegalOp<memref::TensorStoreOp>(
      [&](memref::TensorStoreOp op) {
        return isRewritableType(op.getTensor().getType()) &&
               isRewritableType(op.getMemref().getType());
      });
  // For now, use an explicit allow-list of hlo ops inside the fusion. If any
  // other op is present, the fusion will not be rewritten.
  target.addLegalOp<
      mhlo::AddOp, mhlo::AbsOp, mhlo::CbrtOp, mhlo::CeilOp, mhlo::CosineOp,
      mhlo::DivOp, mhlo::ExpOp, mhlo::Expm1Op, mhlo::FloorOp, mhlo::LogOp,
      mhlo::Log1pOp, mhlo::LogisticOp, mhlo::MulOp, mhlo::NegOp, mhlo::RoundOp,
      /*unsupported: mhlo::RoundNearestEvenOp,*/ mhlo::RsqrtOp, mhlo::SignOp,
      mhlo::SineOp, mhlo::SqrtOp, mhlo::SubtractOp, mhlo::TanhOp>();
  return target;
}

std::unique_ptr<OperationPass<ModuleOp>> createGpuFusionRewritePass() {
  return std::make_unique<GpuFusionRewritePass>();
}

ArrayAttr getWrittenOperandsAttribute(Operation* op) {
  return op->getAttrOfType<ArrayAttr>(kWrittenOperandsAttrName);
}

}  // namespace mlir
