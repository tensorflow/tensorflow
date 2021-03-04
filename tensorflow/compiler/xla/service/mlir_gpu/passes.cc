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

#include "tensorflow/compiler/xla/service/mlir_gpu/passes.h"

#include "absl/memory/memory.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/GPU/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/GPU/ParallelLoopMapper.h"  // from @llvm-project
#include "mlir/Dialect/SCF/SCF.h"  // from @llvm-project
#include "mlir/Dialect/SCF/Transforms.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Transforms/LoopUtils.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"

namespace xla {
namespace mlir_gpu {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/xla/service/mlir_gpu/passes.h.inc"

struct FusionOpRemoverPass : FusionOpRemoverPassBase<FusionOpRemoverPass> {
  void runOnFunction() override {
    getFunction().walk([&](mlir::lmhlo::FusionOp op) {
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

template <typename EffectTy>
bool HasEffectsOnValue(mlir::Value value, mlir::Operation* op) {
  auto mem_effects_interface =
      mlir::dyn_cast_or_null<mlir::MemoryEffectOpInterface>(op);
  if (!mem_effects_interface) {
    return false;
  }
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 2> effects;
  mem_effects_interface.getEffects(effects);
  return llvm::any_of(effects,
                      [op](const mlir::MemoryEffects::EffectInstance& effect) {
                        return mlir::isa<EffectTy>(effect.getEffect());
                      });
}

struct StoreForwardingPass : StoreForwardingPassBase<StoreForwardingPass> {
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
    return HasEffectsOnValue<mlir::MemoryEffects::Allocate>(memref, defOp)
               ? defOp
               : nullptr;
  }

  // Retrieves AllocOp from the cache or actually looks for it.
  mlir::Operation* GetAllocOp(
      mlir::Value memref,
      llvm::DenseMap<mlir::Value, mlir::Operation*>* memrefToAllocOp) {
    auto allocOpIt = memrefToAllocOp->find(memref);
    if (allocOpIt != memrefToAllocOp->end()) {
      return allocOpIt->second;
    }
    mlir::Operation* allocOp = SearchAllocOp(memref);
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

struct DeadTempBufferRemovalPass
    : DeadTempBufferRemovalPassBase<DeadTempBufferRemovalPass> {
  bool operationConsideredDead(mlir::Operation* op) {
    for (auto result : op->getResults()) {
      if (!llvm::all_of(result.getUsers(), [&](mlir::Operation* op) {
            // Store and Dealloc is OK.
            if (llvm::isa<mlir::StoreOp, mlir::DeallocOp>(op)) {
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
    getFunction().walk([&](mlir::Operation* op) {
      if (op->getNumResults() != 1 ||
          !HasEffectsOnValue<mlir::MemoryEffects::Allocate>(op->getResult(0),
                                                            op)) {
        return;
      }
      if (!operationConsideredDead(op)) {
        return;
      }

      // TODO(herhut): There should be a generic helper for this.
      recursiveErase(op, &dead_ops);
    });
    for (auto op : dead_ops) {
      op->erase();
    }
  }
};

struct RewriteKernelSignaturePass
    : RewriteKernelSignaturePassBase<RewriteKernelSignaturePass> {
  void runOnFunction() override {
    mlir::FuncOp func = getFunction();
    mlir::ModuleOp module = func->getParentOfType<mlir::ModuleOp>();
    getFunction().walk([&](mlir::gpu::LaunchFuncOp launchOp) {
      mlir::gpu::GPUFuncOp kernel =
          module.lookupSymbol<mlir::gpu::GPUFuncOp>(launchOp.kernel());

      if (kernel.getNumFuncArguments() !=
          func.getNumArguments() + func.getNumResults()) {
        kernel.emitError()
            << "number of kernel arguments does not match number"
            << "of arguments and results of surrounding function";
        signalPassFailure();
        return;
      }
      if (!llvm::hasSingleElement(func)) {
        func.emitError() << "surrounding function has more than one block";
        signalPassFailure();
        return;
      }

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
      // Also add function results that are computed by the launch.
      mlir::Operation* returnOp = func.getBody().back().getTerminator();
      for (mlir::Value result : returnOp->getOperands()) {
        for (int i = 0, e = launchOp.getNumKernelOperands(); i < e; ++i) {
          if (launchOp.getKernelOperand(i) == result) {
            func_to_kernel.map(result, kernel.getArgument(i));
            break;
          }
        }
      }

      // Create a new kernel function with modified signature. It will have the
      // parameters and result types of the original funcion as its parameter
      // type and otherwise will be void.
      auto gpu_module = kernel->getParentOfType<mlir::gpu::GPUModuleOp>();
      mlir::OpBuilder kernel_builder(gpu_module.body());
      auto operand_types = llvm::to_vector<4>(llvm::concat<const mlir::Type>(
          func.getType().getInputs(), func.getType().getResults()));
      auto new_kernel = kernel_builder.create<mlir::gpu::GPUFuncOp>(
          kernel.getLoc(), kernel.getName(),
          kernel_builder.getFunctionType(operand_types, {}));
      new_kernel->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(),
                          kernel_builder.getUnitAttr());

      // Create a map from old kernel argument to new one.
      mlir::BlockAndValueMapping old_kernel_to_new;
      for (int i = 0, e = func.getNumArguments(); i < e; ++i) {
        mlir::Value func_arg = func.getArgument(i);
        mlir::Value new_kernel_arg = new_kernel.getArgument(i);
        mlir::Value old_kernel_arg = func_to_kernel.lookupOrNull(func_arg);
        if (!old_kernel_arg) {
          kernel.emitOpError()
              << "argument " << i
              << " to containing function is not an argument to the kernel";
          signalPassFailure();
          return;
        }
        old_kernel_to_new.map(old_kernel_arg, new_kernel_arg);
      }
      for (int i = 0, e = returnOp->getNumOperands(); i < e; ++i) {
        mlir::Value ret_op = returnOp->getOperand(i);
        mlir::Value new_kernel_arg =
            new_kernel.getArgument(func.getNumArguments() + i);
        mlir::Value old_kernel_arg = func_to_kernel.lookupOrNull(ret_op);
        if (!old_kernel_arg) {
          kernel.emitOpError()
              << "result " << i
              << " of containing function is not an argument to the kernel";
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
      // Now create a new launchOp calling the new kernel. We need to forward
      // the arguments of the surrounding function and operands to the return.
      mlir::SmallVector<mlir::Value, 4> new_operands;
      new_operands.reserve(new_kernel.getNumFuncArguments());
      new_operands.append(func.args_begin(), func.args_end());
      new_operands.append(returnOp->operand_begin(), returnOp->operand_end());
      mlir::OpBuilder launch_builder(launchOp);
      launch_builder.create<mlir::gpu::LaunchFuncOp>(
          launchOp.getLoc(), new_kernel, launchOp.getGridSizeOperandValues(),
          launchOp.getBlockSizeOperandValues(), new_operands);
      // Launch does not have results, so we can just erase it. And the kernel
      // also needs to go.
      launchOp.erase();
      kernel.erase();
    });
  }
};

struct MapParallelLoopsPass : MapParallelLoopsPassBase<MapParallelLoopsPass> {
  void runOnFunction() override {
    mlir::greedilyMapParallelSCFToGPU(getFunction().getBody());
  }
};

struct FuseInnerParallelLoopsPass
    : FuseInnerParallelLoopsPassBase<FuseInnerParallelLoopsPass> {
  void runOnFunction() override {
    getFunction().walk([](mlir::scf::ParallelOp op) {
      mlir::scf::naivelyFuseParallelOps(op.region());
    });
  }
};

struct ParallelLoopCollapsingToFirstDimPass
    : ParallelLoopCollapsingToFirstDimPassBase<
          ParallelLoopCollapsingToFirstDimPass> {
  void runOnFunction() override {
    getFunction().walk([&](mlir::scf::ParallelOp op) {
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

std::unique_ptr<mlir::FunctionPass> createFusionOpRemoverPass() {
  return absl::make_unique<FusionOpRemoverPass>();
}

std::unique_ptr<mlir::FunctionPass> createStoreForwardingPass() {
  return absl::make_unique<StoreForwardingPass>();
}

std::unique_ptr<mlir::FunctionPass> createDeadTempBufferRemovalPass() {
  return absl::make_unique<DeadTempBufferRemovalPass>();
}

std::unique_ptr<mlir::FunctionPass> createRewriteKernelSignaturePass() {
  return absl::make_unique<RewriteKernelSignaturePass>();
}

std::unique_ptr<mlir::FunctionPass> createFuseInnerParallelLoopsPass() {
  return absl::make_unique<FuseInnerParallelLoopsPass>();
}

std::unique_ptr<mlir::FunctionPass> createMapParallelLoopsPass() {
  return absl::make_unique<MapParallelLoopsPass>();
}

std::unique_ptr<mlir::FunctionPass>
createParallelLoopCollapsingToFirstDimPass() {
  return absl::make_unique<ParallelLoopCollapsingToFirstDimPass>();
}

}  // namespace mlir_gpu
}  // namespace xla
