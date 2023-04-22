/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for injecting execution context to the entry
// function.
//
// Below is an example. Before Conversion:
//  ```
//   func @main(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) ->
//   memref<?x?xf32> {
//     %0 = memref.alloc(...)
//     "lmhlo.add"(%arg0, %arg1, %0) : (memref<?x?xf32>, memref<?x?xf32>,
//     memref<?x?xf32>) -> memref<?x?xf32> return %0 : memref<?x?xf32>
//   }
//  ```
// After conversion:
//  ```
//   func @main(%ctx: !disc_ral.context) {
//     %c0 = constant 0 : index
//     %c1 = constant 1 : index
//     "disc_ral.recv_input"(%ctx, %c0) : (!disc_ral.context, index) ->
//     memref<?x?xf32> "disc_ral.recv_input"(%ctx, %c1) : (!disc_ral.context,
//     index) -> memref<?x?xf32> %0 = memref.alloc(...) "lmhlo.add"(%arg0,
//     %arg1, %0) : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) ->
//     memref<?x?xf32> "disc_ral.send_output"(%ctx, %c0, %0) :
//     (!disc_ral.context, index, memref<?x?xf32>) -> ()
//   }
//  ```

// 1. rewrite entry function (supposed that no other function directly calls the
// entry function)
//    - function signature rewrite
//    - return-like ops rewrite.
// 2. Currently we suppose that functions except the entry function are inlined
// to the entry function. Thus, we don't rewrite all call ops and other
// functions a.t.m. Re-visit this assumption if necessary.

#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace disc_ral {

namespace {

struct RalInjectExecutionContextPass
    : public RalInjectExecutionContextPassBase<RalInjectExecutionContextPass> {
  explicit RalInjectExecutionContextPass(const std::string& entry_func_name)
      : RalInjectExecutionContextPassBase<RalInjectExecutionContextPass>::
            RalInjectExecutionContextPassBase() {
    this->entry_func_name_ = entry_func_name;
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<RalDialect>();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    FuncOp main = m.lookupSymbol<FuncOp>(entry_func_name_);
    if (!main) {
      m.emitError("entry func: " + entry_func_name_ + " not found");
      signalPassFailure();
    }

    Location loc = main.getLoc();
    FunctionType funcType = main.getType();
    OpBuilder b(&main.getBody());
    Block* entry_block = &main.getBody().front();
    Type ctx_type = RalExecutionContextType::get(b.getContext());

    // 1. Prepend context to the entry block arguments
    Value ctx = entry_block->insertArgument(0u, ctx_type);

    // 2. remap original arguments to recv_input ops
    for (auto&& en : llvm::enumerate(
             llvm::zip(funcType.getInputs(),
                       entry_block->getArguments().drop_front(1)))) {
      Value idx = b.create<ConstantIndexOp>(loc, en.index());
      Type argType = std::get<0>(en.value());
      Value oldArgument = std::get<1>(en.value());
      Value newInput = b.create<RecvInputOp>(loc, argType, ctx, idx);
      oldArgument.replaceAllUsesWith(newInput);
    }

    // 3. remap all return-like ops to send_output ops
    for (auto& block : main.getBody()) {
      if (block.empty()) continue;
      Operation& operation = block.back();
      if (!operation.hasTrait<OpTrait::ReturnLike>()) continue;
      b.setInsertionPoint(&operation);
      for (auto& en : llvm::enumerate(operation.getOperands())) {
        Value idx = b.create<ConstantIndexOp>(loc, en.index());
        b.create<SendOutputOp>(loc, ctx, idx, en.value());
      }
      operation.eraseOperands(0, operation.getNumOperands());
    }

    // 4. remove unused block arguments of entry block
    for (int i = 0, e = funcType.getInputs().size(); i < e; ++i) {
      // continue to remove the 1st (starting from zero) argument
      entry_block->eraseArgument(1);
    }

    // 5. set entry func to new type
    main.setType(b.getFunctionType({ctx_type}, {}));
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRalInjectExecutionContextPass(
    const std::string& entry_func_name) {
  return std::make_unique<RalInjectExecutionContextPass>(entry_func_name);
}

}  // namespace disc_ral
}  // namespace mlir
