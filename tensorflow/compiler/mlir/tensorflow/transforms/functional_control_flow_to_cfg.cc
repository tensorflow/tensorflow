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

// This transformation pass transforms functional control flow operations in the
// standard TensorFlow dialect to MLIR Control Flow Graph (CFG) form.

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

struct FunctionalControlFlowToCFG
    : public FunctionPass<FunctionalControlFlowToCFG> {
  void runOnFunction() override;
};

// Lowers a general tensor argument that is used as a condition to a functional
// control flow op into an i1 value.
static Value LowerCondition(Location loc, Value value, OpBuilder* builder) {
  auto zero_d = builder->create<ToBoolOp>(loc, value);
  auto scalar = builder->create<ExtractElementOp>(loc, zero_d);
  return scalar.getResult();
}

// Calls the function `fn` with arguments provided by the given function and
// return the CallOp. Arguments are cast to the required type before calling
// the function.
//
// Requires the function to provide arguments for each of the `fn` operands
// that is compatible for tensor cast.
//
static Operation* CallFn(Location loc, const std::function<Value(int)>& get_arg,
                         FuncOp fn, OpBuilder* builder) {
  FunctionType fn_type = fn.getType();
  llvm::SmallVector<Value, 4> operands;
  int num_operands = fn_type.getNumInputs();
  operands.reserve(num_operands);
  for (int i = 0; i < num_operands; ++i) {
    Value val = get_arg(i);
    Type expected = fn_type.getInput(i);
    if (val.getType() != expected) {
      val =
          builder->create<TF::CastOp>(loc, expected, val,
                                      /*Truncate=*/builder->getBoolAttr(false));
    }
    operands.push_back(val);
  }
  return builder->create<CallOp>(loc, fn, operands).getOperation();
}

// Prepares for jump to the given block by introducing necessary tensor_cast
// operations and returning Values of types required by the block.
//
// Requires the function to provide values for each of the block arguments and
// they should be pair-wise compatible for tensor cast.
static llvm::SmallVector<Value, 4> PrepareValsForJump(
    Location loc, const std::function<Value(int)>& get_val, Block* block,
    OpBuilder* builder) {
  llvm::SmallVector<Value, 4> result;
  int num_vals = block->getNumArguments();
  result.reserve(num_vals);
  for (int i = 0; i < num_vals; ++i) {
    Value val = get_val(i);
    Type expected = block->getArgument(i).getType();
    if (val.getType() != expected) {
      val =
          builder->create<TF::CastOp>(loc, expected, val,
                                      /*Truncate=*/builder->getBoolAttr(false));
    }
    result.push_back(val);
  }
  return result;
}

// Jumps to the given block with arguments provided by the function. Arguments
// are cast to the required type before the jump.
//
// Requires the function to provide values for each of the block arguments and
// they should be pair-wise compatible for tensor cast.
static void JumpToBlock(Location loc, const std::function<Value(int)>& get_arg,
                        Block* block, OpBuilder* builder) {
  auto operands = PrepareValsForJump(loc, get_arg, block, builder);
  builder->create<BranchOp>(loc, block, operands);
}

// Replaces all uses of the operation results in this block with block
// arguments.
//
// Requires that the block has same number of arguments as number of results of
// the operation and either they have same types or are more generic types and
// it is possible to cast them to results' types.
//
static void ReplaceOpResultWithBlockArgs(Location loc, Operation* op,
                                         Block* block, OpBuilder* builder) {
  assert(op->getNumResults() == block->getNumArguments());
  for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
    Value arg = block->getArgument(i);
    Value result = op->getResult(i);
    if (arg.getType() != result.getType()) {
      arg =
          builder->create<TF::CastOp>(loc, result.getType(), arg,
                                      /*Truncate=*/builder->getBoolAttr(false));
    }
    result.replaceAllUsesWith(arg);
  }
}

// Given a functional IfOp, transforms the enclosing code to eliminate it
// completely from the IR, breaking it into operations to evaluate the condition
// as a bool, plus some branches.
//
// This returns true on failure.
//
static LogicalResult LowerIfOp(IfOp op) {
  Operation* op_inst = op.getOperation();
  Location loc = op_inst->getLoc();

  OpBuilder builder(op_inst);

  // Lower the condition to a boolean value (i1).
  Value cond_i1 = LowerCondition(loc, op.cond(), &builder);
  if (!cond_i1) return failure();

  auto module = op_inst->getParentOfType<ModuleOp>();
  auto then_fn = module.lookupSymbol<FuncOp>(op.then_branch());
  auto else_fn = module.lookupSymbol<FuncOp>(op.else_branch());

  // Split the basic block before the 'if'.  The new dest will be our merge
  // point.
  Block* orig_block = op_inst->getBlock();
  Block* merge_block = orig_block->splitBlock(op);

  // Add the block arguments to the merge point, and replace all uses of the
  // original operation results with them.
  for (Value value : op_inst->getResults())
    merge_block->addArgument(value.getType());
  ReplaceOpResultWithBlockArgs(loc, op_inst, merge_block, &builder);

  // Get arguments to the branches after dropping the condition which is the
  // first operand.
  auto get_operand = [&](int i) { return op_inst->getOperand(i + 1); };

  // Set up the 'then' block.
  Block* then_block = builder.createBlock(merge_block);
  Operation* call_op = CallFn(loc, get_operand, then_fn, &builder);

  auto get_then_result = [&](int i) { return call_op->getResult(i); };
  JumpToBlock(loc, get_then_result, merge_block, &builder);

  // Set up the 'else' block.
  Block* else_block = builder.createBlock(merge_block);
  call_op = CallFn(loc, get_operand, else_fn, &builder);

  auto get_else_result = [&](int i) { return call_op->getResult(i); };
  JumpToBlock(loc, get_else_result, merge_block, &builder);

  // Now that we have the then and else blocks, replace the terminator of the
  // orig_block with a conditional branch.
  builder.setInsertionPointToEnd(orig_block);
  builder.create<CondBranchOp>(loc, cond_i1, then_block,
                               llvm::ArrayRef<Value>(), else_block,
                               llvm::ArrayRef<Value>());

  // Finally, delete the op in question.
  op_inst->erase();
  return success();
}

// Given a functional WhileOp, transforms the enclosing code to eliminate it
// completely from the IR, breaking it into operations to execute the loop body
// repeatedly while the loop condition is true.
//
// This returns true on failure.
//
static LogicalResult LowerWhileOp(WhileOp op) {
  Operation* op_inst = op.getOperation();
  Location loc = op_inst->getLoc();

  OpBuilder builder(op_inst);

  auto module = op_inst->getParentOfType<ModuleOp>();
  auto cond_fn = module.lookupSymbol<FuncOp>(op.cond());
  auto body_fn = module.lookupSymbol<FuncOp>(op.body());

  // Split the block containing the While op into two blocks.  One containing
  // operations before the While op and other containing the rest.  Create two
  // new blocks to call condition and body functions.
  //
  // The final control flow graph would be as follows:
  //
  // ...
  // orig_block_head(...):
  //   ...
  //   br cond_block(...)
  // cond_block(...):
  //   %A = call @cond(...)
  //   cond br %A, body_block(...), orig_block_tail(...)
  // body_block(...):
  //   %B = call @body(...)
  //   br cond_block(...)
  // orig_block_tail(...):
  //   ...
  //
  Block* orig_block_head = op_inst->getBlock();
  Block* orig_block_tail = orig_block_head->splitBlock(op);
  Block* cond_block = builder.createBlock(orig_block_tail);
  Block* body_block = builder.createBlock(orig_block_tail);

  // Set argument types for the cond_block to be same as the types of the
  // condition function and argument types for the other two blocks to be same
  // as the input types of the body function. Note that it is always possible
  // for body_block and orig_block_tail to have arguments of the same types as
  // they have exactly one call-site and they are sharing the operands.
  for (Type type : cond_fn.getType().getInputs()) {
    cond_block->addArgument(type);
  }
  for (Type type : body_fn.getType().getInputs()) {
    body_block->addArgument(type);
    orig_block_tail->addArgument(type);
  }

  auto get_operand = [&](int i) { return op_inst->getOperand(i); };

  // Unconditionally branch from the original block to the block containing the
  // condition.
  builder.setInsertionPointToEnd(orig_block_head);
  JumpToBlock(loc, get_operand, cond_block, &builder);

  // Call condition function in the condition block and then branch to the body
  // block or remainder of the original block depending on condition function
  // result.
  builder.setInsertionPointToEnd(cond_block);

  auto get_cond_arg = [&](int i) { return cond_block->getArgument(i); };
  Operation* cond_call_op = CallFn(loc, get_cond_arg, cond_fn, &builder);

  assert(cond_call_op->getNumResults() == 1);
  Value condition = LowerCondition(loc, cond_call_op->getResult(0), &builder);
  auto br_operands =
      PrepareValsForJump(loc, get_cond_arg, body_block, &builder);
  builder.create<CondBranchOp>(loc, condition, body_block, br_operands,
                               orig_block_tail, br_operands);

  // Call body function in the body block and then unconditionally branch back
  // to the condition block.
  builder.setInsertionPointToEnd(body_block);
  auto get_body_arg = [&](int i) { return body_block->getArgument(i); };
  Operation* body_call_op = CallFn(loc, get_body_arg, body_fn, &builder);

  auto get_body_result = [&](int i) { return body_call_op->getResult(i); };
  JumpToBlock(loc, get_body_result, cond_block, &builder);

  // Replace use of the while loop results with block inputs in the remainder of
  // the original block and then delete the original While operation.
  builder.setInsertionPoint(&orig_block_tail->front());
  ReplaceOpResultWithBlockArgs(loc, op_inst, orig_block_tail, &builder);
  op_inst->erase();

  return success();
}

void FunctionalControlFlowToCFG::runOnFunction() {
  // Scan the function looking for these ops.
  for (Block& block : getFunction()) {
    for (Operation& op : block) {
      // If the operation is one of the control flow ops we know, lower it.
      // If we lower an operation, then the current basic block will be split,
      // and the operation will be removed, so we should continue looking at
      // subsequent blocks.
      //
      // TODO: Use PatternRewriter to eliminate these function control flow ops.

      if (IfOp if_op = llvm::dyn_cast<IfOp>(op)) {
        if (failed(LowerIfOp(if_op))) {
          return signalPassFailure();
        }
        break;
      }
      if (WhileOp while_op = llvm::dyn_cast<WhileOp>(op)) {
        if (failed(LowerWhileOp(while_op))) {
          return signalPassFailure();
        }
        break;
      }
    }
  }
}

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateTFFunctionalControlFlowToCFG() {
  return std::make_unique<FunctionalControlFlowToCFG>();
}

static PassRegistration<FunctionalControlFlowToCFG> pass(
    "tf-functional-control-flow-to-cfg",
    "Transform functional control flow Ops to MLIR Control Form Graph "
    "(CFG) form");

}  // namespace TF
}  // namespace mlir
