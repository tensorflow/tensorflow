//===- KernelOutlining.cpp - Implementation of GPU kernel outlining -------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements the GPU dialect kernel outlining pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

template <typename OpTy>
static void createForAllDimensions(OpBuilder &builder, Location loc,
                                   SmallVectorImpl<Value *> &values) {
  for (StringRef dim : {"x", "y", "z"}) {
    Value *v = builder.create<OpTy>(loc, builder.getIndexType(),
                                    builder.getStringAttr(dim));
    values.push_back(v);
  }
}

// Add operations generating block/thread ids and grid/block dimensions at the
// beginning of the `body` region and replace uses of the respective function
// arguments.
static void injectGpuIndexOperations(Location loc, Region &body) {
  OpBuilder builder(loc->getContext());
  Block &firstBlock = body.front();
  builder.setInsertionPointToStart(&firstBlock);
  SmallVector<Value *, 12> indexOps;
  createForAllDimensions<gpu::BlockIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::ThreadIdOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::GridDimOp>(builder, loc, indexOps);
  createForAllDimensions<gpu::BlockDimOp>(builder, loc, indexOps);
  // Replace the leading 12 function args with the respective thread/block index
  // operations. Iterate backwards since args are erased and indices change.
  for (int i = 11; i >= 0; --i) {
    firstBlock.getArgument(i)->replaceAllUsesWith(indexOps[i]);
    firstBlock.eraseArgument(i);
  }
}

static bool isInliningBeneficiary(Operation *op) {
  return isa<ConstantOp>(op) || isa<DimOp>(op);
}

// Move arguments of the given kernel function into the function if this reduces
// the number of kernel arguments.
static gpu::LaunchFuncOp inlineBeneficiaryOps(gpu::GPUFuncOp kernelFunc,
                                              gpu::LaunchFuncOp launch) {
  OpBuilder kernelBuilder(kernelFunc.getBody());
  auto &firstBlock = kernelFunc.getBody().front();
  SmallVector<Value *, 8> newLaunchArgs;
  BlockAndValueMapping map;
  for (int i = 0, e = launch.getNumKernelOperands(); i < e; ++i) {
    map.map(launch.getKernelOperand(i), kernelFunc.getArgument(i));
  }
  for (int i = launch.getNumKernelOperands() - 1; i >= 0; --i) {
    auto operandOp = launch.getKernelOperand(i)->getDefiningOp();
    if (!operandOp || !isInliningBeneficiary(operandOp)) {
      newLaunchArgs.push_back(launch.getKernelOperand(i));
      continue;
    }
    // Only inline operations that do not create new arguments.
    if (!llvm::all_of(operandOp->getOperands(),
                      [map](Value *value) { return map.contains(value); })) {
      continue;
    }
    auto clone = kernelBuilder.clone(*operandOp, map);
    firstBlock.getArgument(i)->replaceAllUsesWith(clone->getResult(0));
    firstBlock.eraseArgument(i);
  }
  if (newLaunchArgs.size() == launch.getNumKernelOperands())
    return launch;

  std::reverse(newLaunchArgs.begin(), newLaunchArgs.end());
  OpBuilder LaunchBuilder(launch);
  SmallVector<Type, 8> newArgumentTypes;
  newArgumentTypes.reserve(firstBlock.getNumArguments());
  for (auto value : firstBlock.getArguments()) {
    newArgumentTypes.push_back(value->getType());
  }
  kernelFunc.setType(LaunchBuilder.getFunctionType(newArgumentTypes, {}));
  auto newLaunch = LaunchBuilder.create<gpu::LaunchFuncOp>(
      launch.getLoc(), kernelFunc, launch.getGridSizeOperandValues(),
      launch.getBlockSizeOperandValues(), newLaunchArgs);
  launch.erase();
  return newLaunch;
}

// Outline the `gpu.launch` operation body into a kernel function. Replace
// `gpu.return` operations by `std.return` in the generated function.
static gpu::GPUFuncOp outlineKernelFunc(gpu::LaunchOp launchOp) {
  Location loc = launchOp.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation.
  OpBuilder builder(launchOp.getContext());

  SmallVector<Type, 4> kernelOperandTypes(launchOp.getKernelOperandTypes());
  FunctionType type =
      FunctionType::get(kernelOperandTypes, {}, launchOp.getContext());
  std::string kernelFuncName =
      Twine(launchOp.getParentOfType<FuncOp>().getName(), "_kernel").str();
  auto outlinedFunc = builder.create<gpu::GPUFuncOp>(loc, kernelFuncName, type);
  outlinedFunc.setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                       builder.getUnitAttr());
  outlinedFunc.body().takeBody(launchOp.body());
  injectGpuIndexOperations(loc, outlinedFunc.body());
  return outlinedFunc;
}

// Replace `gpu.launch` operations with an `gpu.launch_func` operation launching
// `kernelFunc`. The kernel func contains the body of the `gpu.launch` with
// constant region arguments inlined.
static void convertToLaunchFuncOp(gpu::LaunchOp &launchOp,
                                  gpu::GPUFuncOp kernelFunc) {
  OpBuilder builder(launchOp);
  auto launchFuncOp = builder.create<gpu::LaunchFuncOp>(
      launchOp.getLoc(), kernelFunc, launchOp.getGridSizeOperandValues(),
      launchOp.getBlockSizeOperandValues(), launchOp.getKernelOperandValues());
  inlineBeneficiaryOps(kernelFunc, launchFuncOp);
  launchOp.erase();
}

namespace {

/// Pass that moves the kernel of each LaunchOp into its separate nested module.
///
/// This pass moves the kernel code of each LaunchOp into a function created
/// inside a nested module. It also creates an external function of the same
/// name in the parent module.
///
/// The kernel modules are intended to be compiled to a cubin blob independently
/// in a separate pass. The external functions can then be annotated with the
/// symbol of the cubin accessor function.
class GpuKernelOutliningPass : public ModulePass<GpuKernelOutliningPass> {
public:
  void runOnModule() override {
    SymbolTable symbolTable(getModule());
    bool modified = false;
    for (auto func : getModule().getOps<FuncOp>()) {
      // Insert just after the function.
      Block::iterator insertPt(func.getOperation()->getNextNode());
      func.walk([&](gpu::LaunchOp op) {
        gpu::GPUFuncOp outlinedFunc = outlineKernelFunc(op);

        // Create nested module and insert outlinedFunc. The module will
        // originally get the same name as the function, but may be renamed on
        // insertion into the parent module.
        auto kernelModule = createKernelModule(outlinedFunc, symbolTable);
        symbolTable.insert(kernelModule, insertPt);

        // Potentially changes signature, pulling in constants.
        convertToLaunchFuncOp(op, outlinedFunc);
        modified = true;
      });
    }

    // If any new module was inserted in this module, annotate this module as
    // a container module.
    if (modified)
      getModule().setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                          UnitAttr::get(&getContext()));
  }

private:
  // Returns a module containing kernelFunc and all callees (recursive).
  ModuleOp createKernelModule(gpu::GPUFuncOp kernelFunc,
                              const SymbolTable &parentSymbolTable) {
    auto context = getModule().getContext();
    Builder builder(context);
    auto kernelModule =
        ModuleOp::create(builder.getUnknownLoc(), kernelFunc.getName());
    kernelModule.setAttr(gpu::GPUDialect::getKernelModuleAttrName(),
                         builder.getUnitAttr());
    SymbolTable symbolTable(kernelModule);
    symbolTable.insert(kernelFunc);

    SmallVector<Operation *, 8> symbolDefWorklist = {kernelFunc};
    while (!symbolDefWorklist.empty()) {
      if (Optional<SymbolTable::UseRange> symbolUses =
              SymbolTable::getSymbolUses(symbolDefWorklist.pop_back_val())) {
        for (SymbolTable::SymbolUse symbolUse : *symbolUses) {
          StringRef symbolName =
              symbolUse.getSymbolRef().cast<FlatSymbolRefAttr>().getValue();
          if (symbolTable.lookup(symbolName))
            continue;

          Operation *symbolDefClone =
              parentSymbolTable.lookup(symbolName)->clone();
          symbolDefWorklist.push_back(symbolDefClone);
          symbolTable.insert(symbolDefClone);
        }
      }
    }

    return kernelModule;
  }
};

} // namespace

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createGpuKernelOutliningPass() {
  return std::make_unique<GpuKernelOutliningPass>();
}

static PassRegistration<GpuKernelOutliningPass>
    pass("gpu-kernel-outlining",
         "Outline gpu.launch bodies to kernel functions.");
