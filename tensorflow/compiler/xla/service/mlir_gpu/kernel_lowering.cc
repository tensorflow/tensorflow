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
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"  // TF:local_config_mlir
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"  // TF:local_config_mlir
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"  // TF:local_config_mlir
#include "mlir/Dialect/GPU/GPUDialect.h"  // TF:local_config_mlir
#include "mlir/Dialect/GPU/Passes.h"  // TF:local_config_mlir
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // TF:local_config_mlir
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"  // TF:local_config_mlir
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"  // TF:local_config_mlir
#include "mlir/Dialect/Linalg/Passes.h"  // TF:local_config_mlir
#include "mlir/Dialect/LoopOps/LoopOps.h"  // TF:local_config_mlir
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/BlockAndValueMapping.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/OperationSupport.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/IR/Region.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"  // TF:local_config_mlir
#include "mlir/Transforms/DialectConversion.h"  // TF:local_config_mlir
#include "mlir/Transforms/Passes.h"  // TF:local_config_mlir
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
    auto getConstantValue = [](mlir::Value* value) -> llvm::Optional<int64_t> {
      auto definingOp = value->getDefiningOp();
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
    getFunction().walk([&](mlir::LoadOp loadOp) {
      auto block = loadOp.getOperation()->getBlock();
      auto iterator = std::find_if(block->rbegin(), block->rend(),
                                   [&loadOp](mlir::Operation& other) {
                                     return &other == loadOp.getOperation();
                                   });
      if (++iterator == block->rend()) return;
      mlir::StoreOp storeOp = llvm::dyn_cast<mlir::StoreOp>(&*(iterator));
      if (!storeOp) return;
      // Check both store to the same value.
      if (storeOp.memref() != loadOp.memref()) return;
      auto storeIndices = storeOp.getIndices();
      auto loadIndices = loadOp.getIndices();
      if (!std::equal(storeIndices.begin(), storeIndices.end(),
                      loadIndices.begin(), loadIndices.end())) {
        return;
      }
      loadOp.replaceAllUsesWith(storeOp.getValueToStore());
      loadOp.erase();
    });
  };
};

// Simple pass that removes temporary buffers that are only written to but
// never read from or that are read but the read value is not used.
// Needs an analysis that proves that loads and stores are side-effect free
// (in bounds, no aliasing, etc.).
struct DeadTempBufferRemoval : mlir::FunctionPass<DeadTempBufferRemoval> {
  bool operationConsideredDead(mlir::Operation* op) {
    for (auto result : op->getResults()) {
      if (!llvm::all_of(result->getUsers(), [&](mlir::Operation* op) {
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
      for (auto user : llvm::make_early_inc_range(result->getUsers())) {
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

// Neat little helper pass to dump the IR inbetween passes.
struct DumpPass : public mlir::ModulePass<DumpPass> {
  void runOnModule() override {
#if DEBUG
    getModule().dump();
#endif
  }
};

}  // namespace

Status LowerLHLOToGPU(mlir::ModuleOp module) {
  mlir::PassManager pm(module.getContext());

  // First, lower bodies of fusion operations from hlo to lhlo.
  pm.addPass(absl::make_unique<FusionToLhloConverter>());
  // Next, we can strip the outer fusion operation.
  pm.addPass(absl::make_unique<FusionOpRemover>());
  // Transform lhlo operations to LinAlg.
  pm.addPass(::mlir::xla_lhlo::createLegalizeToLinalgPass());
  // Fuse linalg operations. This will yield a single tiled loop nest where
  // the inner loops are single trip.
  pm.addPass(::mlir::xla_lhlo::createLhloFuseLinalg());
  pm.addPass(absl::make_unique<DumpPass>());
  // Go from linalg to normal loops.
  pm.addPass(::mlir::linalg::createConvertLinalgToLoopsPass());
  pm.addPass(absl::make_unique<DumpPass>());
  // Canonicalize the code to simplify index computations.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addPass(absl::make_unique<DumpPass>());
  // The innermost loops will be single-trip.
  pm.addPass(absl::make_unique<SingleTripLoopRemoval>());
  pm.addPass(absl::make_unique<DumpPass>());
  // Run CSE to ensure that loads and stores to the same subview get
  // recognized as such.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  pm.addPass(absl::make_unique<DumpPass>());
  // Forward stores to buffers to loads.
  pm.addPass(absl::make_unique<StoreForwardingPass>());
  pm.addPass(absl::make_unique<DumpPass>());
  // Remove now unused temporary buffers.
  pm.addPass(absl::make_unique<DeadTempBufferRemoval>());
  pm.addPass(absl::make_unique<DumpPass>());
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

Status LowerKernelBodiesToNVVM(mlir::ModuleOp module) {
  // We cannot verify as the signature of the kernel is rewritten.
  ::mlir::PassManager pm(module.getContext(), /*verifyPasses=*/false);

  // Rewrite kernel functions to LLVM IR.
  auto& kernelPm = pm.nest<::mlir::ModuleOp>();
  kernelPm.addPass(::mlir::createLowerGpuOpsToNVVMOpsPass());
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
  module.walk([&kernelModule](mlir::ModuleOp nestedModule) {
    if (nestedModule.getAttrOfType<mlir::UnitAttr>(
            mlir::gpu::GPUDialect::getKernelModuleAttrName())) {
      for (auto& fn : nestedModule) {
        kernelModule.push_back(fn.clone());
      }
    }
  });
  return kernelModule;
}
}  // namespace mlir_gpu
}  // namespace xla
