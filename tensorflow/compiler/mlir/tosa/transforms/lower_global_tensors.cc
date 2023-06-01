/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "mlir/Dialect/MLProgram/IR/MLProgram.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

#define PASS_NAME "tosa-lower-global-tensors"
#define DEBUG_TYPE PASS_NAME

namespace mlir::tosa {

#define GEN_PASS_DEF_LOWERGLOBALTENSORS
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

namespace {

class LowerGlobalTensorsPass
    : public impl::LowerGlobalTensorsBase<LowerGlobalTensorsPass> {
 public:
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::TFL::TensorFlowLiteDialect,
                    mlir::ml_program::MLProgramDialect>();
  }

  // Converts TFLite state operations to the MLProgram equivalent.
  void runOnOperation() override {
    auto* context = &getContext();
    auto moduleOp = getOperation();
    mlir::OpBuilder builder(moduleOp.getBodyRegion());

    DenseMap<StringRef, func::FuncOp> symNameToFunction;
    for (auto func : moduleOp.getOps<func::FuncOp>()) {
      symNameToFunction[func.getSymName()] = func;
    }

    DenseMap<StringRef, DenseElementsAttr> sharedNameToConstant;
    DenseMap<StringRef, LocationAttr> sharedNameToLoc;

    SmallVector<mlir::TFL::VarHandleOp, 6> handleOps;
    SmallVector<mlir::TFL::AssignVariableOp, 6> assignOps;
    SmallVector<mlir::TFL::ReadVariableOp, 6> readOps;
    for (auto it : symNameToFunction) {
      auto func = std::get<1>(it);
      // Look through the initialization functions and find the assigned values
      // for each handle, save out the constant value.
      for (auto init : func.getOps<mlir::TFL::CallOnceOp>()) {
        auto findInitFunc =
            symNameToFunction.find(init.getSessionInitFunction());
        if (findInitFunc == symNameToFunction.end()) {
          init.emitError("unable to find initialization function: " +
                         init.getSessionInitFunction());
          continue;
        }
        func::FuncOp initFunc = std::get<1>(*findInitFunc);
        for (auto assign : initFunc.getOps<mlir::TFL::AssignVariableOp>()) {
          auto handle = dyn_cast<mlir::TFL::VarHandleOp>(
              assign.getResourceId().getDefiningOp());
          if (!handle) continue;

          DenseElementsAttr constant;
          if (!matchPattern(assign.getValue(), m_Constant(&constant))) {
            // Quantized types we can not use the m_Constant matcher.
            if (auto constOp = dyn_cast<mlir::TFL::QConstOp>(
                    assign.getValue().getDefiningOp())) {
              constant = constOp.getValue().cast<DenseElementsAttr>();
            }
          }
          if (!constant) continue;

          auto name = handle.getSharedName();
          sharedNameToConstant[name] = constant;
          sharedNameToLoc[name] = handle.getLoc();
        }
      }

      // We also want to grab the list of operations to replace.
      for (auto& op : func.getOps()) {
        if (auto handle = dyn_cast<mlir::TFL::VarHandleOp>(op))
          handleOps.push_back(handle);
        if (auto assign = dyn_cast<mlir::TFL::AssignVariableOp>(op))
          assignOps.push_back(assign);
        if (auto read = dyn_cast<mlir::TFL::ReadVariableOp>(op))
          readOps.push_back(read);
      }
    }

    // TF::CallOnceOps are no longer needed as we have already extracted their
    // state.
    SmallVector<mlir::TFL::CallOnceOp> callOnceOps;
    for (auto func : moduleOp.getOps<func::FuncOp>()) {
      for (auto init : func.getOps<mlir::TFL::CallOnceOp>()) {
        callOnceOps.push_back(init);
      }
    }
    for (auto op : callOnceOps) op.erase();

    // Create the ml_program::GlobalOps to store our new global variables.
    DenseMap<StringRef, mlir::ml_program::GlobalOp> symbolRefMap;
    for (auto it : sharedNameToConstant) {
      auto name = std::get<0>(it);
      auto attribute = std::get<1>(it);
      auto locIt = sharedNameToLoc.find(name);
      LocationAttr loc = mlir::UnknownLoc();
      if (locIt != sharedNameToLoc.end()) {
        loc = std::get<1>(*locIt);
      }

      // TODO(suderman): Determine the global type based on all store
      // operations.
      auto global = builder.create<mlir::ml_program::GlobalOp>(
          loc, name, attribute.getType(), /*is_mutable=*/true, attribute,
          nullptr);
      global.setPrivate();

      symbolRefMap[name] = global;
    }

    // Replace the assign ops with a global store operation.
    for (auto assign : assignOps) {
      auto handle = dyn_cast<mlir::TFL::VarHandleOp>(
          assign.getResourceId().getDefiningOp());
      if (!handle) continue;

      Value value = assign.getValue();
      auto globalOpIt = symbolRefMap.find(handle.getSharedName());
      if (globalOpIt == symbolRefMap.end()) {
        assign->emitError(
            "unable to find corresponding GlobalOp for op's VarHandle");
        continue;
      }
      auto globalOp = std::get<1>(*globalOpIt);

      builder.setInsertionPoint(assign);
      if (globalOp.getType() != value.getType()) {
        value = builder
                    .create<UnrealizedConversionCastOp>(
                        assign.getLoc(), globalOp.getType(), value)
                    .getResult(0);
      }

      auto globalSymbolRef = SymbolRefAttr::get(context, globalOp.getSymName());
      builder.create<mlir::ml_program::GlobalStoreOp>(assign.getLoc(),
                                                      globalSymbolRef, value);
      assign.erase();
    }

    for (auto read : readOps) {
      auto handle = dyn_cast<mlir::TFL::VarHandleOp>(
          read.getResourceId().getDefiningOp());
      if (!handle) continue;

      auto globalOpIt = symbolRefMap.find(handle.getSharedName());
      if (globalOpIt == symbolRefMap.end()) continue;
      auto globalOp = std::get<1>(*globalOpIt);

      builder.setInsertionPoint(read);

      auto globalSymbolRef = SymbolRefAttr::get(context, globalOp.getSymName());
      Value load = builder.create<mlir::ml_program::GlobalLoadOp>(
          read.getLoc(), globalOp.getType(), globalSymbolRef);

      if (read.getType() != load.getType()) {
        load = builder
                   .create<UnrealizedConversionCastOp>(read.getLoc(),
                                                       read.getType(), load)
                   .getResult(0);
      }
      read.getResult().replaceAllUsesWith(load);
      read.erase();
    }

    for (auto handle : handleOps) {
      if (handle.getResult().use_empty()) {
        handle.erase();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createLowerGlobalTensorsPass() {
  return std::make_unique<LowerGlobalTensorsPass>();
}

}  // namespace mlir::tosa
