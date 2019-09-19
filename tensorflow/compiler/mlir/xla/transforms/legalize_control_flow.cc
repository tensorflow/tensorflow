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

// This file implements logic for lowering XLA dialect to Standard dialect.

#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/BlockAndValueMapping.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

using mlir::PassRegistration;

namespace mlir {
namespace xla_hlo {
namespace {
struct LegalizeControlFlow : public mlir::FunctionPass<LegalizeControlFlow> {
  // Perform the lowering to MLIR control flow.
  void runOnFunction() override;
};

bool LowerWhileOp(mlir::xla_hlo::WhileOp while_op) {
  // Converts an xla while loop into control flow. This mostly generates the
  // right MLIR boilerplate for calling the body / condition functions, then
  // branching on their results appropriately. The operation should look similar
  // to below:
  //
  //   <prior operations>
  //   %0 = "xla_hlo.while"(%arg0) {body: @loop, cond: @cond}
  //   <post operations>
  auto* opInst = while_op.getOperation();
  mlir::OpBuilder builder(while_op);
  auto loc = while_op.getLoc();

  llvm::SmallVector<Value*, 4> operands;
  operands.reserve(while_op.getNumOperands());
  for (auto operand : while_op.getOperands()) {
    operands.push_back(operand);
  }

  // Break the block into four sections:
  // orig_block - operations before the while and the branch into looping check.
  // tail_block - operations after the while loop completes.
  // cond_block - check the looping condition, then conditionally branch into
  // the
  //             loop or, if condition is false, jump to the tail branch.
  // body_block - call the loop body, then jump back to the condition block.
  auto* orig_block = opInst->getBlock();
  auto* tail_block = orig_block->splitBlock(opInst);

  auto* cond_block = builder.createBlock(tail_block);
  auto* body_block = builder.createBlock(tail_block);

  // Setup the end of the original block:
  //     <prior operations>
  //     br ^cond(%arg0) // Jumps to the condition statement.
  builder.setInsertionPointToEnd(orig_block);
  builder.create<mlir::BranchOp>(loc, cond_block, operands);

  // Setup the condition block:
  //   ^cond(%0):
  //     %1 = call @cond(%0) : (...) -> tensor<i1> // Evaluate condition.
  //     %2 = extract_element %1[] : tensor<i1> // Extract the condition value.
  //     cond_br %2, ^body(%0), ^tail(%0) // Branch.
  builder.setInsertionPointToStart(cond_block);
  llvm::SmallVector<Value*, 4> cond_block_arguments;
  cond_block_arguments.reserve(while_op.getNumOperands());
  for (auto operand : while_op.getOperands()) {
    cond_block->addArgument(operand->getType());
    cond_block_arguments.push_back(cond_block->getArguments().back());
  }

  auto cond_op = builder.create<mlir::CallOp>(
      loc, while_op.cond(), builder.getTensorType({}, builder.getI1Type()),
      cond_block_arguments);
  auto cond_value =
      builder.create<mlir::ExtractElementOp>(loc, cond_op.getResult(0))
          .getResult();
  builder.create<mlir::CondBranchOp>(loc, cond_value, body_block,
                                     cond_block_arguments, tail_block,
                                     cond_block_arguments);

  // Create the body block:
  //   ^body(%3: tensor<i64>):
  //     %4 = call @body(%3)        // Call the body function to evaluate.
  //     br ^cond(%4 : tensor<i64>) // Continue to the loop condition.

  builder.setInsertionPointToStart(body_block);
  llvm::SmallVector<Value*, 4> body_block_arguments;
  body_block_arguments.reserve(while_op.getNumOperands());
  for (auto operand : while_op.getOperands()) {
    body_block->addArgument(operand->getType());
    body_block_arguments.push_back(body_block->getArguments().back());
  }

  SmallVector<Type, 4> body_result_types(while_op.getResultTypes());
  auto body_op = builder.create<mlir::CallOp>(
      loc, while_op.body(), body_result_types, body_block_arguments);
  llvm::SmallVector<Value*, 4> body_results;
  body_results.reserve(body_op.getNumResults());
  for (auto result : body_op.getResults()) {
    body_results.push_back(result);
  }
  builder.create<mlir::BranchOp>(loc, cond_block, body_results);

  // Setup the tail block:
  //   ^tail(%5):
  //     <post operations>
  llvm::SmallVector<Value*, 4> tail_block_arguments;
  tail_block_arguments.reserve(while_op.getNumOperands());

  // Erase the original while loop.
  for (int i = 0; i < while_op.getNumOperands(); i++) {
    tail_block->addArgument(while_op.getOperand(i)->getType());
    while_op.getResult(i)->replaceAllUsesWith(tail_block->getArgument(i));
  }
  opInst->erase();

  return false;
}

void LegalizeControlFlow::runOnFunction() {
  auto func = getFunction();
  llvm::SmallVector<WhileOp, 4> control_flow_ops;
  func.walk([&](WhileOp op) { control_flow_ops.push_back(op); });

  for (auto& op : control_flow_ops) {
    if (LowerWhileOp(op)) return signalPassFailure();
  }
}
}  // namespace
}  // namespace xla_hlo
}  // namespace mlir

std::unique_ptr<mlir::OpPassBase<mlir::FuncOp>>
mlir::xla_hlo::createLegalizeControlFlowPass() {
  return std::make_unique<LegalizeControlFlow>();
}

static PassRegistration<mlir::xla_hlo::LegalizeControlFlow> legalize_cf_pass(
    "xla-legalize-control-flow",
    "Legalize from XLA control flow to MLIR control flow");
