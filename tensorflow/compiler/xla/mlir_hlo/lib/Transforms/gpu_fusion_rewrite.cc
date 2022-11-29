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

#include "lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mhlo/IR/hlo_ops.h"
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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {

#define GEN_PASS_DEF_GPUFUSIONREWRITEPASS
#include "mlir-hlo/Transforms/gpu_passes.h.inc"

// Name of the 'gpu.launch_func' attribute which specifies the written operands.
static constexpr llvm::StringLiteral kWrittenOperandsAttrName("lmhlo.written");

namespace {
struct HloToGpuPipelineOptions {
  SmallVector<int64_t> blockTileDim;
  SmallVector<int64_t> warpTileDim;
  SmallVector<int64_t> threadTileDim;
  bool experimentalSoftmax = false;
};

class GpuFusionRewritePass
    : public impl::GpuFusionRewritePassBase<GpuFusionRewritePass> {
 public:
  explicit GpuFusionRewritePass() = default;
  using Pass::runPipeline;  // Give FusionRewritePattern access.

 private:
  void getDependentDialects(DialectRegistry& registry) const override;
  void runOnOperation() override;

  // Rewrites `lmhlo.fusion` to `gpu.launch_func` for fusion regions that the
  // HLO to GPU pipeline can handle.
  LogicalResult rewriteFusionOp(lmhlo::FusionOp fusionOp,
                                RewriterBase& rewriter,
                                const HloToGpuPipelineOptions& options,
                                GpuFusionRewritePass& parentPass,
                                SymbolTable& symbolTable) const;
};
}  // namespace

// Returns the number of groups per block for softmax. Each group of threads
// handles a row. A group has power-of-two size up to a warp. We use 256 threads
// per block and a group size that leaves less than half of the threads unused.
static int64_t getGroupsPerBlock(TensorType type) {
  int64_t reductionDim = type.getShape().back();
  for (int64_t numGroups = 8; numGroups <= 256; numGroups *= 2) {
    if (reductionDim * numGroups > 128) return numGroups;
  }
  return 8;
}

// Returns the number of elements each thread should handle for 'type'.
// The intention is that loads and stores are vectorized later on to this width
// to maximize memory throughput.
static int64_t getElementsPerThread(TensorType type) {
  // Don't vectorize if the number of elements cannot saturate the GPU.
  // Use a coarse heuristic because we don't know the target GPU here.
  const int64_t kNumFp32AlusOnV100 = 5376;
  if (type.getNumElements() < kNumFp32AlusOnV100) return 1;

  // Don't vectorize if element type is not int or float.
  if (!type.getElementType().isIntOrFloat()) return 1;

  // Vectorize so that loads and stores are 128 bits per thread.
  return 128 / type.getElementType().getIntOrFloatBitWidth();
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

// Annotates gpu.launch_func with attribute specifying written operands.
//
// func.func @fusion(%arg0, %arg1 {lmhlo.written}) {
//   gpu.launch_func args(%arg0, %arg1, %arg0)
//
// will add a `lmhlo.written = [false, true, false]` attribute.
//
// The 'written_operands' attribute is used later to retrieve which
// gpu.launch_func arguments are written vs. just read.
static void annotateLaunchFunc(func::FuncOp funcOp, RewriterBase& rewriter) {
  funcOp.walk([&](gpu::LaunchFuncOp op) {
    auto writtenOperands = llvm::to_vector(
        llvm::map_range(op.getKernelOperands(), [&](Value operand) -> bool {
          auto arg = operand.dyn_cast<BlockArgument>();
          if (!arg) return false;
          return funcOp.getArgAttr(arg.getArgNumber(),
                                   kWrittenOperandsAttrName) != nullptr;
        }));
    op->setAttr(kWrittenOperandsAttrName,
                rewriter.getBoolArrayAttr(writtenOperands));
  });
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
  // F8 types are not yet supported.
  // TODO(b/259609697): Support F8 types.
  if (shapedType.getElementType().isFloat8E5M2() ||
      shapedType.getElementType().isFloat8E4M3FN())
    return false;
  return true;
}

// Returns target where lowerable fusion ops are marked legal.
static ConversionTarget getRewritableTarget(MLIRContext* ctx) {
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
  target.addDynamicallyLegalOp<
      mhlo::AddOp, mhlo::AbsOp, mhlo::CbrtOp, mhlo::CeilOp, mhlo::CosineOp,
      mhlo::DivOp, mhlo::ExpOp, mhlo::Expm1Op, mhlo::FloorOp, mhlo::LogOp,
      mhlo::Log1pOp, mhlo::LogisticOp, mhlo::MulOp, mhlo::NegOp, mhlo::RoundOp,
#if !TENSORFLOW_USE_ROCM
      mhlo::RoundNearestEvenOp,
#endif
      mhlo::RsqrtOp, mhlo::SignOp, mhlo::SineOp, mhlo::SqrtOp, mhlo::SubtractOp,
      mhlo::TanhOp>([&](Operation* op) { return op->hasOneUse(); });
  return target;
}

// Returns the hlo-to-gpu pipeline options, or failure if the fusion cannot be
// rewritten.
static FailureOr<HloToGpuPipelineOptions> getPipelineOptions(
    lmhlo::FusionOp fusionOp, RewriterBase& rewriter) {
  if (fusionOp.getFusionResults().size() != 1)
    return rewriter.notifyMatchFailure(fusionOp, "expected single result");
  if (isa<bufferization::ToTensorOp>(fusionOp.getFusionRoots().front()))
    return rewriter.notifyMatchFailure(fusionOp, "expected non-empty fusion");

  auto resultType =
      fusionOp.getFusionResults().front().getType().cast<TensorType>();
  // If fusion type is tagged as softmax, use that.
  if (auto fusionType = fusionOp->getAttrOfType<StringAttr>("fusion_type");
      fusionType && fusionType.getValue() == "softmax_fusion") {
    HloToGpuPipelineOptions options;
    options.blockTileDim = {getGroupsPerBlock(resultType)};
    options.warpTileDim = {1};
    options.experimentalSoftmax = true;
    return options;
  }

  ConversionTarget rewritableTarget =
      getRewritableTarget(rewriter.getContext());

  // If fusion_op (including its region) is not legal by rewriteableTarget, we
  // expect lowering to GPU to fail or produce incorrect results.
  auto callback = [&](Operation* op) {
    if (rewritableTarget.isLegal(op)) return WalkResult::advance();
    (void)rewriter.notifyMatchFailure(op, "expected to be rewritable");
    return WalkResult::interrupt();
  };
  if (fusionOp.getRegion().walk(callback).wasInterrupted()) return failure();

  int64_t elementsPerThread = getElementsPerThread(resultType);
  constexpr int64_t kThreadsPerWarp = 32;
  int64_t elementsPerWarp = elementsPerThread * kThreadsPerWarp;
  int64_t elementsPerBlock =
      getThreadsPerBlock(resultType, elementsPerThread) * elementsPerThread;
  HloToGpuPipelineOptions options;
  options.blockTileDim = {elementsPerBlock};
  options.warpTileDim = {elementsPerWarp};
  options.threadTileDim = {elementsPerThread};
  return options;
}

// Rewrites `lmhlo.fusion` to `gpu.launch_func` for fusion regions that the
// HLO to GPU pipeline can handle.
LogicalResult GpuFusionRewritePass::rewriteFusionOp(
    lmhlo::FusionOp fusionOp, RewriterBase& rewriter,
    const HloToGpuPipelineOptions& options, GpuFusionRewritePass& parentPass,
    SymbolTable& symbolTable) const {
  rewriter.setInsertionPoint(fusionOp);

  // Collect values in fusion region defined above.
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(fusionOp->getRegions(), captures);

  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });
  // Convert statically shaped types to their 1D equivalent. This only works for
  // element wise fusions and will have to become a more sophisticated pass when
  // e.g. broadcasts are involved.
  if (!options.experimentalSoftmax) {
    converter.addConversion([](ShapedType type) {
      if (!type.hasStaticShape()) return type;
      return type.clone(type.getNumElements());
    });
    converter.addConversion([&](MemRefType type) {
      if (!type.hasStaticShape() || !type.getLayout().isIdentity()) return type;
      return MemRefType::get(type.getNumElements(), type.getElementType(),
                             MemRefLayoutAttrInterface(),
                             type.getMemorySpace());
    });
  }

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
  // Convert statically shaped types to their 1D equivalent.
  funcOp->walk([&](Operation* op) {
    for (auto result : op->getResults())
      result.setType(converter.convertType(result.getType()));
  });
  // Add attribute to written function arguments.
  for (const BlockArgument& arg : funcOp.getArguments()) {
    if (llvm::any_of(arg.getUsers(), [](Operation* op) {
          return isa<memref::TensorStoreOp>(op);
        })) {
      funcOp.setArgAttr(arg.getArgNumber(), kWrittenOperandsAttrName,
                        rewriter.getUnitAttr());
    }
  }

  // Create and run the HLO to GPU pass pipeline.
  // Note: passManager.enableIRPrinting() doesn't do anything on dynamic pass
  // pipelines. Printing needs to be enabled on the parent pass manager.
  PassManager passManager(rewriter.getContext());
  createHloToGpuPipeline(passManager, options.blockTileDim, options.warpTileDim,
                         options.threadTileDim, options.experimentalSoftmax);
  if (failed(parentPass.runPipeline(passManager, moduleOp)))
    return fusionOp->emitError() << "failed to run the hlo-to-gpu pipeline";

  // Clone the (single) gpu module with the device function.
  rewriter.setInsertionPoint(fusionOp->getParentOfType<func::FuncOp>());
  for (auto gpuModuleOp : moduleOp.getBodyRegion().getOps<gpu::GPUModuleOp>()) {
    StringAttr symbol =
        symbolTable.insert(rewriter.clone(*gpuModuleOp.getOperation()));
    if (failed(symbolTable.replaceAllSymbolUses(gpuModuleOp, symbol, funcOp)))
      return gpuModuleOp->emitError() << "failed to replace symbol";
  }
  // Add 'gpu.container_module' attribute to parent module.
  fusionOp->getParentOfType<ModuleOp>()->setAttr(
      gpu::GPUDialect::getContainerModuleAttrName(), rewriter.getUnitAttr());

  // Annotate gpu.launch_func loc and attribute specifying written operands.
  funcOp->walk([&](gpu::LaunchFuncOp op) { op->setLoc(loc); });
  annotateLaunchFunc(funcOp, rewriter);

  // Replace fusion op with host function region.
  rewriter.splitBlock(&funcOp.front(),
                      funcOp.front().getTerminator()->getIterator());
  rewriter.mergeBlockBefore(&funcOp.front(), fusionOp, captures.getArrayRef());

  rewriter.eraseOp(fusionOp);
  rewriter.eraseOp(moduleOp);

  return success();
}

void GpuFusionRewritePass::getDependentDialects(
    DialectRegistry& registry) const {
  // Collect the dependent dialects for both variants of the pipeline.
  OpPassManager passManager;
  for (bool experimentalSoftmax : {false, true})
    createHloToGpuPipeline(passManager, {}, {}, {}, experimentalSoftmax);
  passManager.getDependentDialects(registry);
}

void GpuFusionRewritePass::runOnOperation() {
  IRRewriter rewriter(&getContext());
  SymbolTable symbolTable(getOperation());

  auto callback = [&](lmhlo::FusionOp fusionOp) {
    FailureOr<HloToGpuPipelineOptions> options =
        getPipelineOptions(fusionOp, rewriter);
    if (failed(options)) return WalkResult::advance();
    return WalkResult(
        rewriteFusionOp(fusionOp, rewriter, *options, *this, symbolTable));
  };

  if (getOperation().walk(callback).wasInterrupted())
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> createGpuFusionRewritePass() {
  return std::make_unique<GpuFusionRewritePass>();
}

ArrayAttr getWrittenOperandsAttribute(Operation* op) {
  return op->getAttrOfType<ArrayAttr>(kWrittenOperandsAttrName);
}

}  // namespace mlir
