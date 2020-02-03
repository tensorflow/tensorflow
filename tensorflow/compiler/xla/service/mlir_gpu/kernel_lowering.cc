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

#include <memory>

#include "absl/memory/memory.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"  // TF:llvm-project
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"  // TF:llvm-project
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"  // TF:llvm-project
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"  // TF:llvm-project
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"  // TF:llvm-project
#include "mlir/Dialect/GPU/GPUDialect.h"  // TF:llvm-project
#include "mlir/Dialect/GPU/Passes.h"  // TF:llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // TF:llvm-project
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // TF:llvm-project
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // TF:llvm-project
#include "mlir/Dialect/Linalg/Passes.h"  // TF:llvm-project
#include "mlir/Dialect/LoopOps/LoopOps.h"  // TF:llvm-project
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/IR/OperationSupport.h"  // TF:llvm-project
#include "mlir/IR/PatternMatch.h"  // TF:llvm-project
#include "mlir/IR/Region.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassManager.h"  // TF:llvm-project
#include "mlir/Transforms/DialectConversion.h"  // TF:llvm-project
#include "mlir/Transforms/Passes.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace mlir_gpu {
namespace {

using ::mlir::xla_lhlo::FusionOp;

// Following are some small transformations that are required to clean up code
// after lowering from linalg to loops.

// A simple pass that applies lowering of HLO to LHLO only within Fusion
// operations. This is needed, as FusionOp is not closed from above and hence
// nested pass managers can not be applied.
struct FusionToLhloConverter
    : public mlir::FunctionPass<FusionToLhloConverter> {
  void runOnFunction() override {
    auto& ctx = getContext();
    mlir::OwningRewritePatternList patterns;
    mlir::ConversionTarget target(ctx);
    target.addLegalDialect<::mlir::xla_lhlo::XlaLhloDialect>();
    ::mlir::xla_hlo::populateHLOToLHLOConversionPattern(&ctx, &patterns);

    getFunction().walk([&](FusionOp op) {
      if (failed(applyPartialConversion(op, target, patterns, nullptr))) {
        signalPassFailure();
      }
    });
    getFunction().walk([&](mlir::xla_lhlo::ReduceOp op) {
      if (failed(applyPartialConversion(op, target, patterns, nullptr))) {
        signalPassFailure();
      }
    });
  }
};

// Replaces a FusionOp by the operations contained in its region.
struct FusionOpRemover : public mlir::FunctionPass<FusionOpRemover> {
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

// Rewrite the single-trip loops we get out of linalg into just their bodies.
// TODO(herhut): Make this a general pattern.
struct SingleTripLoopRemoval
    : public mlir::FunctionPass<SingleTripLoopRemoval> {
  void runOnFunction() override {
    auto getConstantValue = [](mlir::Value value) -> llvm::Optional<int64_t> {
      auto definingOp = value.getDefiningOp();
      if (!definingOp) return llvm::None;
      auto constantOp = llvm::dyn_cast<mlir::ConstantOp>(definingOp);
      if (!constantOp) return llvm::None;
      auto integer = constantOp.getValue().dyn_cast<mlir::IntegerAttr>();
      if (!integer) return llvm::None;
      return integer.getInt();
    };
    getFunction().walk([&](mlir::loop::ForOp forOp) {
      auto lower = getConstantValue(forOp.lowerBound());
      auto upper = getConstantValue(forOp.upperBound());
      auto step = getConstantValue(forOp.step());
      if (!lower || !upper || !step) return;
      if ((lower.getValue() < upper.getValue()) &&
          (lower.getValue() + step.getValue() >= upper.getValue())) {
        // This loop has a single trip, so we can move the body in front.
        mlir::BlockAndValueMapping mapping;
        mlir::OpBuilder b(forOp);
        mapping.map(forOp.getInductionVar(), forOp.lowerBound());
        for (auto& nested_op : forOp.getBody()->without_terminator()) {
          auto clone = b.clone(nested_op, mapping);
          for (auto pair :
               llvm::zip(nested_op.getResults(), clone->getResults())) {
            mapping.map(std::get<0>(pair), std::get<1>(pair));
          }
        }
        forOp.erase();
      }
    });
  }
};

// Simple pass that replaces a load that immediately follows a store to the
// same address with the stored value. This needs generalization.
struct StoreForwardingPass : mlir::FunctionPass<StoreForwardingPass> {
  void runOnFunction() override {
    llvm::DenseMap<mlir::Value, mlir::Operation*> memrefToAllocOp;

    getFunction().walk([&](mlir::LoadOp loadOp) {
      auto* block = loadOp.getOperation()->getBlock();
      auto loadOpIt = std::find_if(block->rbegin(), block->rend(),
                                   [&loadOp](mlir::Operation& other) {
                                     return &other == loadOp.getOperation();
                                   });
      for (auto storeOpIt = loadOpIt; storeOpIt != block->rend(); ++storeOpIt) {
        auto storeOp = llvm::dyn_cast<mlir::StoreOp>(&*(storeOpIt));
        if (!storeOp) {
          continue;
        }
        mlir::Operation* storeOpAlloc =
            GetAllocOp(storeOp.memref(), &memrefToAllocOp);
        mlir::Operation* loadOpAlloc =
            GetAllocOp(loadOp.memref(), &memrefToAllocOp);
        if (!storeOpAlloc || !loadOpAlloc || storeOpAlloc != loadOpAlloc) {
          continue;
        }
        auto storeIndices = storeOp.getIndices();
        auto loadIndices = loadOp.getIndices();
        if (!std::equal(storeIndices.begin(), storeIndices.end(),
                        loadIndices.begin(), loadIndices.end())) {
          return;
        }
        loadOp.replaceAllUsesWith(storeOp.getValueToStore());
        loadOp.erase();
        return;
      }
    });
  };

  // Recursively checks defining ops until finds AllocOp. Return either AllocOp
  // if it is found or nullptr.
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
};

// Simple pass that removes temporary buffers that are only written to but
// never read from or that are read but the read value is not used.
// Needs an analysis that proves that loads and stores are side-effect free
// (in bounds, no aliasing, etc.).
struct DeadTempBufferRemoval : mlir::FunctionPass<DeadTempBufferRemoval> {
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

  void recursiveErase(mlir::Operation* op) {
    for (auto result : op->getResults()) {
      for (auto user : llvm::make_early_inc_range(result.getUsers())) {
        recursiveErase(user);
      }
    }
    op->erase();
  }

  void runOnFunction() override {
    getFunction().walk([&](mlir::AllocOp allocOp) {
      if (!operationConsideredDead(allocOp)) {
        return;
      }

      // TODO(herhut): There should be a generic helper for this.
      recursiveErase(allocOp);
    });
  }
};

void EnableIRPrinting(mlir::PassManager* passManager) {
  auto enable_if_vlog_is_on = [](mlir::Pass* pass, mlir::Operation* op) {
    return VLOG_IS_ON(1);
  };
  passManager->enableIRPrinting(/*shouldPrintBeforePass=*/{},
                                /*shouldPrintAfterPass=*/enable_if_vlog_is_on,
                                /*printModuleScope=*/false,
                                /*printAfterOnlyOnChange=*/true, llvm::dbgs());
  passManager->disableMultithreading();
}

}  // namespace

Status LowerLHLOToGPU(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  EnableIRPrinting(&pm);

  // First, lower bodies of fusion operations from hlo to lhlo.
  pm.addPass(absl::make_unique<FusionToLhloConverter>());
  // Next, we can strip the outer fusion operation.
  pm.addPass(absl::make_unique<FusionOpRemover>());
  // Transform lhlo operations to LinAlg.
  pm.addPass(::mlir::xla_lhlo::createLegalizeLhloToLinalgPass());
  // Fuse linalg operations. This will yield a single tiled loop nest where
  // the inner loops are single trip.
  pm.addPass(::mlir::xla_lhlo::createLhloFuseLinalg());
  // Legalize reduce operations directly to GPU dialect.
  pm.addPass(::mlir::xla_lhlo::createLegalizeToGpuPass());
  // Fuse linalg operations. This will yield a single tiled loop nest where
  // Go from linalg to normal loops.
  pm.addPass(::mlir::createConvertLinalgToLoopsPass());
  // Canonicalize the code to simplify index computations.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  // The innermost loops will be single-trip.
  pm.addPass(absl::make_unique<SingleTripLoopRemoval>());
  // Run CSE to ensure that loads and stores to the same subview get
  // recognized as such.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Forward stores to buffers to loads.
  pm.addPass(absl::make_unique<StoreForwardingPass>());
  // Remove now unused temporary buffers.
  pm.addPass(absl::make_unique<DeadTempBufferRemoval>());
  // Coalesce generated loops to have 1d loops.
  pm.addPass(::mlir::createLoopCoalescingPass());
  // Transform the now 1d loops to gpu launches.
  pm.addPass(::mlir::createSimpleLoopsToGPUPass(/*numBlockDims=*/0,
                                                /*numThreadDims=*/1));
  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Take launches to launches with kernels.
  pm.addPass(::mlir::createGpuKernelOutliningPass());

  if (failed(pm.run(module))) {
    return InternalError("Lowering to GPU kernels failed.");
  }
  return Status::OK();
}

namespace {

/// A pass that does the final lowering to NVVM. It collects all the patterns
/// that are currently required, currently mixing std, linalg and gpu.
class LowerToNVVMPass
    : public ::mlir::OperationPass<LowerToNVVMPass, ::mlir::gpu::GPUModuleOp> {
 public:
  void runOnOperation() override {
    ::mlir::gpu::GPUModuleOp m = getOperation();

    ::mlir::OwningRewritePatternList patterns;
    ::mlir::LinalgTypeConverter converter(m.getContext());
    ::mlir::populateStdToLLVMConversionPatterns(converter, patterns);
    // TODO(b/145824979) Remove linalg once sliceop is in std.
    ::mlir::populateLinalgToLLVMConversionPatterns(converter, patterns,
                                                   &getContext());
    ::mlir::populateGpuToNVVMConversionPatterns(converter, patterns);

    ::mlir::ConversionTarget target(getContext());
    target.addIllegalDialect<::mlir::gpu::GPUDialect>();
    target.addIllegalOp<::mlir::LLVM::ExpOp>();
    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
    // TODO(csigg): Remove once we support replacing non-root ops.
    target.addLegalOp<::mlir::gpu::GPUModuleOp, ::mlir::gpu::ModuleEndOp,
                      ::mlir::gpu::YieldOp>();
    if (failed(applyPartialConversion(m, target, patterns, &converter))) {
      signalPassFailure();
    }
  }
};

}  // anonymous namespace

Status LowerKernelBodiesToNVVM(mlir::ModuleOp module) {
  // We cannot verify as the signature of the kernel is rewritten.
  ::mlir::PassManager pm(module.getContext(), /*verifyPasses=*/false);
  EnableIRPrinting(&pm);

  // Rewrite kernel functions to LLVM IR.
  auto& kernelPm = pm.nest<::mlir::gpu::GPUModuleOp>();
  kernelPm.addPass(::mlir::createLowerToCFGPass());
  kernelPm.addPass(absl::make_unique<LowerToNVVMPass>());
  // Some basic cleanup.
  kernelPm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  kernelPm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());

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
