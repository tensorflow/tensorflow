/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/mlir_gpu/kernel_lowering.h"

#include "absl/memory/memory.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"  // from @llvm-project
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"  // from @llvm-project
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"  // from @llvm-project
#include "mlir/Dialect/GPU/Passes.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // from @llvm-project
#include "mlir/Dialect/Linalg/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/BufferPlacement.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/LoopUtils.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace mlir_gpu {
namespace {

using ::mlir::xla_lhlo::FusionOp;

// Replaces a FusionOp by the operations contained in its region.
struct FusionOpRemover
    : public mlir::PassWrapper<FusionOpRemover, ::mlir::FunctionPass> {
  void runOnFunction() override {
    getFunction().walk([&](FusionOp op) {
      mlir::OpBuilder builder(op);
      // FusionOp has a single region with a single block, so we can just walk
      // over it and clone operations to the outside.
      mlir::BlockAndValueMapping mapping;
      for (auto& nested_op : op.region().front().without_terminator()) {
        auto clone = builder.clone(nested_op, mapping);
        for (auto pair :
             llvm::zip(nested_op.getResults(), clone->getResults())) {
          mapping.map(std::get<0>(pair), std::get<1>(pair));
        }
      }
      op.erase();
    });
  }
};

// Simple pass that replaces a load that immediately follows a store to the
// same address with the stored value. This needs generalization.
struct StoreForwardingPass
    : mlir::PassWrapper<StoreForwardingPass, mlir::FunctionPass> {
  mlir::StoreOp findStore(mlir::Operation* op,
                          std::function<bool(mlir::StoreOp)> matches) {
    // Search from op upwards in the current block.
    mlir::Block* block = op->getBlock();
    auto startFromIt =
        std::find_if(block->rbegin(), block->rend(),
                     [op](mlir::Operation& other) { return &other == op; });
    for (auto storeOpIt = startFromIt; storeOpIt != block->rend();
         ++storeOpIt) {
      auto storeOp = llvm::dyn_cast<mlir::StoreOp>(&*(storeOpIt));
      if (!storeOp || !matches(storeOp)) {
        continue;
      }

      return storeOp;
    }
    // No store operation found. Continue search outside of the parallel
    // loop if block is in a parallel loop.
    if (auto parallelOp =
            llvm::dyn_cast<mlir::scf::ParallelOp>(block->getParentOp())) {
      return findStore(parallelOp.getOperation(), matches);
    }
    return {};
  }

  // Recursively search defining ops for AllocOp. Return either AllocOp if it is
  // found or nullptr.
  mlir::Operation* SearchAllocOp(mlir::Value memref) {
    mlir::Operation* defOp = memref.getDefiningOp();
    while (auto subviewOp = mlir::dyn_cast_or_null<mlir::SubViewOp>(defOp)) {
      defOp = subviewOp.source().getDefiningOp();
    }
    if (auto allocOp = mlir::dyn_cast_or_null<mlir::AllocOp>(defOp)) {
      return allocOp.getOperation();
    }
    return nullptr;
  }

  // Retrieves AllocOp from the cache or actually looks for it.
  mlir::Operation* GetAllocOp(
      mlir::Value memref,
      llvm::DenseMap<mlir::Value, mlir::Operation*>* memrefToAllocOp) {
    auto allocOpIt = memrefToAllocOp->find(memref);
    if (allocOpIt != memrefToAllocOp->end()) {
      return allocOpIt->second;
    }
    auto allocOp = SearchAllocOp(memref);
    memrefToAllocOp->insert({memref, allocOp});
    return allocOp;
  }

  void runOnFunction() override {
    llvm::DenseMap<mlir::Value, mlir::Operation*> memrefToAllocOp;

    getFunction().walk([&](mlir::LoadOp loadOp) {
      auto storeOp = findStore(loadOp, [&](mlir::StoreOp storeOp) {
        mlir::Operation* storeOpAlloc =
            GetAllocOp(storeOp.memref(), &memrefToAllocOp);
        mlir::Operation* loadOpAlloc =
            GetAllocOp(loadOp.memref(), &memrefToAllocOp);
        return storeOpAlloc && loadOpAlloc && (storeOpAlloc == loadOpAlloc);
      });
      if (!storeOp) {
        return;
      }
      auto storeIndices = storeOp.getIndices();
      auto loadIndices = loadOp.getIndices();
      if (!std::equal(storeIndices.begin(), storeIndices.end(),
                      loadIndices.begin(), loadIndices.end())) {
        return;
      }
      loadOp.replaceAllUsesWith(storeOp.getValueToStore());
      loadOp.erase();
    });
  }
};

// Simple pass that removes temporary buffers that are only written to but
// never read from or that are read but the read value is not used.
// Needs an analysis that proves that loads and stores are side-effect free
// (in bounds, no aliasing, etc.).
struct DeadTempBufferRemoval
    : mlir::PassWrapper<DeadTempBufferRemoval, ::mlir::FunctionPass> {
  bool operationConsideredDead(mlir::Operation* op) {
    for (auto result : op->getResults()) {
      if (!llvm::all_of(result.getUsers(), [&](mlir::Operation* op) {
            // Store and Dealloc is OK.
            if (llvm::isa<mlir::StoreOp>(op) ||
                llvm::isa<mlir::DeallocOp>(op)) {
              return true;
            }
            // Load without uses is also ok.
            if (auto loadOp = llvm::dyn_cast<mlir::LoadOp>(op)) {
              return loadOp.use_empty();
            }
            // Subview is ok if it is dead itself.
            if (llvm::isa<mlir::SubViewOp>(op)) {
              return operationConsideredDead(op);
            }
            return false;
          })) {
        return false;
      }
    }
    return true;
  }

  void recursiveErase(mlir::Operation* op,
                      llvm::SmallVectorImpl<mlir::Operation*>* erase_list) {
    for (auto result : op->getResults()) {
      for (auto user : llvm::make_early_inc_range(result.getUsers())) {
        recursiveErase(user, erase_list);
      }
    }
    erase_list->push_back(op);
  }

  void runOnFunction() override {
    llvm::SmallVector<mlir::Operation*, 8> dead_ops;
    getFunction().walk([&](mlir::AllocOp allocOp) {
      if (!operationConsideredDead(allocOp)) {
        return;
      }

      // TODO(herhut): There should be a generic helper for this.
      recursiveErase(allocOp, &dead_ops);
    });
    for (auto op : dead_ops) {
      op->erase();
    }
  }
};

// TODO(herhut): Move this to MLIR core.
struct MoveScalarComputationsIntoGpuLaunch
    : mlir::PassWrapper<MoveScalarComputationsIntoGpuLaunch,
                        mlir::FunctionPass> {
  static bool isInliningBeneficiary(mlir::Operation* op) {
    return llvm::isa<mlir::ConstantOp>(op) || llvm::isa<mlir::DimOp>(op) ||
           llvm::isa<mlir::SelectOp>(op) || llvm::isa<mlir::CmpIOp>(op);
  }

  static bool extractBeneficiaryOps(
      mlir::Operation* op, llvm::SmallVectorImpl<mlir::Operation*>* ops,
      llvm::SetVector<mlir::Value> args) {
    if (!isInliningBeneficiary(op)) {
      return false;
    }

    ops->push_back(op);
    for (auto operand : op->getOperands()) {
      // It is an existing arg, keep going.
      if (args.count(operand)) {
        continue;
      }
      mlir::Operation* definingOp = operand.getDefiningOp();
      if (!definingOp || !extractBeneficiaryOps(definingOp, ops, args)) {
        return false;
      }
    }
    return true;
  }

  static void inlineOperationsIntoLaunch(mlir::gpu::LaunchOp launch) {
    llvm::SetVector<mlir::Value> used_above;
    mlir::getUsedValuesDefinedAbove(launch.body(), used_above);
    mlir::BlockAndValueMapping inlined_map;
    for (mlir::Value v : used_above) {
      llvm::SmallVector<mlir::Operation*, 8> ops_to_move;
      mlir::Operation* definingOp = v.getDefiningOp();
      if (definingOp &&
          extractBeneficiaryOps(definingOp, &ops_to_move, used_above)) {
        mlir::OpBuilder b(launch.body());
        for (mlir::Operation* op : llvm::reverse(ops_to_move)) {
          auto result = b.clone(*op, inlined_map);
          for (auto pair : llvm::zip(op->getResults(), result->getResults())) {
            mlir::replaceAllUsesInRegionWith(std::get<0>(pair),
                                             std::get<1>(pair), launch.body());
          }
          inlined_map.map(op->getResults(), result->getResults());
        }
      }
    }
  }

  void runOnFunction() override {
    mlir::FuncOp fun = getFunction();
    fun.walk(
        [](mlir::gpu::LaunchOp launch) { inlineOperationsIntoLaunch(launch); });
  }
};

// TODO(herhut): Make this a proper thing.
struct FixKernelFunctionSignatures
    : mlir::PassWrapper<FixKernelFunctionSignatures, mlir::FunctionPass> {
  void runOnFunction() override {
    mlir::FuncOp func = getFunction();
    mlir::ModuleOp module = func.getParentOfType<mlir::ModuleOp>();
    getFunction().walk([&](mlir::gpu::LaunchFuncOp launchOp) {
      mlir::gpu::GPUFuncOp kernel =
          module.lookupSymbol<mlir::gpu::GPUFuncOp>(launchOp.kernel());
      // Compute a map from function arguments to kernel function operands.
      mlir::BlockAndValueMapping func_to_kernel;
      for (mlir::BlockArgument arg : func.getArguments()) {
        for (int i = 0, e = launchOp.getNumKernelOperands(); i < e; ++i) {
          if (launchOp.getKernelOperand(i) == arg) {
            func_to_kernel.map(arg, kernel.getArgument(i));
            break;
          }
        }
      }

      // Create a new kernel function with modified signature. We know that it
      // will have the same signature as the original function, so just reuse it
      // here.
      auto gpu_module = kernel.getParentOfType<mlir::gpu::GPUModuleOp>();
      mlir::OpBuilder kernel_builder(gpu_module.body());
      auto new_kernel = kernel_builder.create<mlir::gpu::GPUFuncOp>(
          kernel.getLoc(), kernel.getName(), func.getType());
      new_kernel.setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                         kernel_builder.getUnitAttr());

      // Create a map from old kernel argument to new one.
      mlir::BlockAndValueMapping old_kernel_to_new;
      for (int i = 0, e = kernel.getNumFuncArguments(); i < e; ++i) {
        mlir::Value func_arg = func.getArgument(i);
        mlir::Value new_kernel_arg = new_kernel.getArgument(i);
        mlir::Value old_kernel_arg = func_to_kernel.lookupOrNull(func_arg);
        if (!old_kernel_arg) {
          kernel.emitOpError()
              << "argument " << i
              << "to kernel is not an argument to the containing function";
          signalPassFailure();
          return;
        }
        old_kernel_to_new.map(old_kernel_arg, new_kernel_arg);
      }
      // Steal the body by appending the blocks and inserting a branch.
      kernel.body().cloneInto(&new_kernel.getBody(), old_kernel_to_new);
      kernel_builder.setInsertionPointToEnd(&new_kernel.body().front());
      kernel_builder.create<mlir::BranchOp>(
          new_kernel.getLoc(), &*std::next(new_kernel.body().begin()));
      // Now create a new launchOp calling the new kernel. We can just forward
      // the arguments of the function to the launch, as we fixed the
      // signature.
      mlir::OpBuilder launch_builder(launchOp);
      launch_builder.create<mlir::gpu::LaunchFuncOp>(
          launchOp.getLoc(), new_kernel, launchOp.getGridSizeOperandValues(),
          launchOp.getBlockSizeOperandValues(), func.getArguments());
      // Launch does not have results, so we can just erase it. And the kernel
      // also needs to go.
      launchOp.erase();
      kernel.erase();
    });
  }
};

// Extract_element(xla_hlo_scalars_to_dimension_tensor(v_i), i) -> v_i
//
// We need to direct fusion to the inner loops. This cannot be done with
// a passmanager alone ATM, as nested pass managers require operations to
// be closed from above.
struct MapParallelLoops
    : public mlir::PassWrapper<MapParallelLoops, mlir::FunctionPass> {
  void runOnFunction() override {
    mlir::greedilyMapParallelSCFToGPU(getFunction().getBody());
  }
};

// We need to direct fusion to the inner loops. This cannot be done with
// a passmanager alone ATM, as nested pass managers require operations to
// be closed from above.
struct FuseInnerParallelLoops
    : public mlir::PassWrapper<FuseInnerParallelLoops, mlir::FunctionPass> {
  void runOnFunction() override {
    getFunction().walk([](mlir::scf::ParallelOp op) {
      mlir::scf::naivelyFuseParallelOps(op.region());
    });
  }
};

// Collapse all loop dimension into the first one.
struct ParallelLoopCollapsingToFirstDim
    : public mlir::PassWrapper<ParallelLoopCollapsingToFirstDim,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    mlir::Operation* module = getOperation();

    module->walk([&](mlir::scf::ParallelOp op) {
      unsigned num_loops = op.getNumLoops();
      std::vector<unsigned> combinedLoops;
      combinedLoops.reserve(num_loops);
      for (unsigned i = 0; i < num_loops; ++i) {
        combinedLoops.push_back(i);
      }
      mlir::collapseParallelLoops(op, {combinedLoops});
    });
  }
};
}  // namespace

Status LowerLHLOToGPU(mlir::ModuleOp module,
                      llvm::ArrayRef<unsigned> tile_sizes,
                      llvm::ArrayRef<unsigned> unroll_factors,
                      bool collapseParallelLoops) {
  mlir::PassManager pm(module.getContext());
  applyPassManagerCLOptions(pm);

  // We have to anticipate later unrolling in tiling to make sure that we get
  // the requested tiling after unrolling. Compute the new tiling here if
  // needed.
  llvm::SmallVector<unsigned, 4> tiling_for_unrolling;
  llvm::SmallVector<int64_t, 4> as_int64;
  if (!unroll_factors.empty()) {
    tiling_for_unrolling.reserve(tile_sizes.size());
    for (auto pair : llvm::zip(tile_sizes, unroll_factors)) {
      tiling_for_unrolling.push_back(std::get<0>(pair) * std::get<1>(pair));
      as_int64.push_back(std::get<1>(pair));
    }
  } else {
    tiling_for_unrolling.append(tile_sizes.begin(), tile_sizes.end());
  }

  // Legalize from HLO to LHLO.
  pm.addPass(::mlir::xla_hlo::createLegalizeToLhloPass());
  // Moving `AllocOp`s and inserting missing `DeallocOp`s
  pm.addPass(::mlir::createBufferPlacementPass());
  // Next, we can strip the outer fusion operation.
  pm.addPass(absl::make_unique<FusionOpRemover>());
  // Remove unnecessary LHLO copies.
  pm.addPass(::mlir::xla_lhlo::createLhloCopyRemovalPass());
  // Transform LHLO operations to LinAlg.
  pm.addPass(::mlir::xla_lhlo::createLegalizeLhloToLinalgPass());
  // Fuse linalg operations.
  // TODO(herhut): Make tiling conigurable.
  pm.addPass(::mlir::xla_lhlo::createLhloFuseLinalg(/*use_parallel_loops=*/true,
                                                    tiling_for_unrolling));
  // Legalize reduce operations directly to GPU dialect.
  pm.addPass(::mlir::xla_lhlo::createLegalizeToGpuPass());
  // Transform the Linalg operations inside of the loop nest into parallel
  // loops.
  pm.addPass(::mlir::createConvertLinalgToParallelLoopsPass());
  // Canonicalize the code to simplify index computations. This is needed so
  // that loop bounds have the same value.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Fuse the inner-most loops.
  pm.addPass(absl::make_unique<FuseInnerParallelLoops>());
  // Run CSE to ensure that loads and stores to the same subview get
  // recognized as such.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Forward stores to buffers to loads.
  pm.addPass(absl::make_unique<StoreForwardingPass>());
  // Remove now unused temporary buffers.
  pm.addPass(absl::make_unique<DeadTempBufferRemoval>());
  if (!unroll_factors.empty()) {
    pm.addPass(::mlir::createParallelLoopTilingPass(as_int64));
  }
  // Project all loop dimensions to X if necessary.
  if (collapseParallelLoops) {
    pm.addPass(absl::make_unique<ParallelLoopCollapsingToFirstDim>());
  }
  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Greedily map the remaining loop to GPU hardware dimensions.
  pm.addPass(absl::make_unique<MapParallelLoops>());
  // Apply the mapping.
  pm.addPass(mlir::createParallelLoopToGpuPass());
  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Move scalar operations into the launch to ensure smaller signatures.
  pm.addPass(absl::make_unique<MoveScalarComputationsIntoGpuLaunch>());
  // Take launches to launches with kernels.
  pm.addPass(::mlir::createGpuKernelOutliningPass());
  // Make sure the kernel signature resembled the original function's
  // signature
  pm.addPass(absl::make_unique<FixKernelFunctionSignatures>());
  if (failed(pm.run(module))) {
    return InternalError("Lowering to GPU kernels failed.");
  }
  return Status::OK();
}

namespace {

/// A pass that does the final lowering to NVVM. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class LowerToNVVMPass
    : public ::mlir::PassWrapper<
          LowerToNVVMPass, ::mlir::OperationPass<::mlir::gpu::GPUModuleOp>> {
 public:
  void runOnOperation() override {
    ::mlir::gpu::GPUModuleOp m = getOperation();

    ::mlir::OwningRewritePatternList patterns;
    ::mlir::LLVMTypeConverter converter(m.getContext());
    ::mlir::populateStdToLLVMConversionPatterns(converter, patterns);
    // TODO(b/145824979) Remove linalg once sliceop is in std.
    ::mlir::populateLinalgToLLVMConversionPatterns(converter, patterns,
                                                   &getContext());
    ::mlir::populateGpuToNVVMConversionPatterns(converter, patterns);
    ::mlir::populateAffineToStdConversionPatterns(patterns, m.getContext());
    ::mlir::ConversionTarget target(getContext());
    target.addIllegalDialect<::mlir::gpu::GPUDialect>();
    target.addIllegalOp<::mlir::LLVM::ExpOp>();
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
    // TODO(csigg): Remove once we support replacing non-root ops.
    target.addLegalOp<::mlir::gpu::GPUModuleOp, ::mlir::gpu::ModuleEndOp,
                      ::mlir::gpu::YieldOp>();
    if (failed(mlir::applyFullConversion(m, target, patterns, &converter))) {
      signalPassFailure();
    }
  }
};

}  // namespace

Status LowerKernelBodiesToNVVM(mlir::ModuleOp module) {
  // We cannot verify as the signature of the kernel is rewritten.
  ::mlir::PassManager pm(module.getContext(), /*verifyPasses=*/false);
  applyPassManagerCLOptions(pm);

  // Rewrite kernel functions to LLVM IR.
  auto& kernelPm = pm.nest<::mlir::gpu::GPUModuleOp>();
  kernelPm.addPass(::mlir::createLowerToCFGPass());
  kernelPm.addPass(absl::make_unique<LowerToNVVMPass>());
  // Some basic cleanup.
  kernelPm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  kernelPm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  kernelPm.addPass(::mlir::createStripDebugInfoPass());

  if (failed(pm.run(module))) {
    return InternalError("Lowering to NVVM IR failed.");
  }
  return Status::OK();
}

StatusOr<mlir::ModuleOp> ExtractKernelModule(mlir::ModuleOp module) {
  auto kernelModule = ::mlir::ModuleOp::create(module.getLoc());
  // TODO(b/137624192): This also needs to resolve naming conflicts.
  module.walk([&kernelModule](mlir::gpu::GPUModuleOp nestedModule) {
    for (auto& fn : nestedModule.body().front()) {
      kernelModule.push_back(fn.clone());
    }
  });
  return kernelModule;
}
}  // namespace mlir_gpu
}  // namespace xla
