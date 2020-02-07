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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Identifier.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/SymbolTable.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Transforms/RegionUtils.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"

namespace mlir {
namespace TFL {
namespace {

// This pass outlines the cond/body region of the TFL WhileOp into functions and
// replaces the regions with calls to these outlined functions.
class WhileOutlinePass : public mlir::ModulePass<WhileOutlinePass> {
 public:
  explicit WhileOutlinePass() {}

 private:
  void runOnModule() override;

  // Outlines the regions of the WhileOp's cond and body and insert function
  // calls instead,
  void OutlineWhile(WhileOp while_op);

  // Get unique name by using the loc to name mapping.
  std::string GetName(Operation* op, StringRef suffix);

  tensorflow::OpOrArgLocNameMapper mapper_;
};

std::string WhileOutlinePass::GetName(Operation* op, StringRef suffix) {
  return (mapper_.GetUniqueName(op) + suffix).str();
}

void WhileOutlinePass::OutlineWhile(WhileOp while_op) {
  OpBuilder builder(&getContext());
  auto i1_tensor = RankedTensorType::get({}, builder.getI1Type());
  llvm::SetVector<Value> extern_values;
  getUsedValuesDefinedAbove(while_op.cond(), extern_values);
  getUsedValuesDefinedAbove(while_op.body(), extern_values);

  // Colect new types.
  SmallVector<Type, 4> types;
  types.reserve(extern_values.size() + while_op.getNumOperands());
  for (BlockArgument ba : while_op.cond().front().getArguments())
    types.push_back(ba.getType());
  for (Value operand : extern_values) types.push_back(operand.getType());

  // Create outline function from region. Optional pass extra arguments through
  // to yield.
  SymbolTable symbol_table(getModule());
  auto create_outline_func = [&](StringRef name, FunctionType type,
                                 Region& region, bool passthru_extra_args) {
    auto outlined_func = builder.create<FuncOp>(while_op.getLoc(), name, type,
                                                ArrayRef<NamedAttribute>{});
    outlined_func.getBody().takeBody(region);
    Block& block = outlined_func.getBody().front();
    llvm::SmallVector<Value, 4> new_args;
    new_args.reserve(extern_values.size());
    for (Value value : extern_values) {
      auto arg = block.addArgument(value.getType());
      replaceAllUsesInRegionWith(value, arg, outlined_func.getBody());
      new_args.push_back(arg);
    }

    // Replace yield op with return.
    Operation* yield_op = outlined_func.getBody().front().getTerminator();
    OpBuilder b(yield_op);
    llvm::SmallVector<Value, 4> args;
    args.reserve(yield_op->getNumOperands() + new_args.size());
    args.append(yield_op->operand_begin(), yield_op->operand_end());
    if (passthru_extra_args) args.append(new_args.begin(), new_args.end());
    b.create<ReturnOp>(yield_op->getLoc(), args);
    yield_op->erase();
    symbol_table.insert(outlined_func);
    return outlined_func;
  };

  // Replace region with call to outline function.
  auto replace_with_call = [&](FunctionType fn_type, StringRef name,
                               Region& region, bool passthru_extra_args) {
    // Skip if already only a call.
    if (region.front().getOperations().size() == 2 &&
        isa<mlir::CallOp>(region.front().front()))
      return;

    auto func = create_outline_func(name, fn_type, region, passthru_extra_args);
    OpBuilder b(region);
    // The body of the region is empty/has been outlined into the function.
    auto block = b.createBlock(&region);
    SmallVector<Value, 4> new_operands;
    new_operands.reserve(types.size());
    for (Type t : types) new_operands.push_back(block->addArgument(t));
    auto call = b.create<CallOp>(while_op.getLoc(), func, new_operands);
    b.create<YieldOp>(while_op.getLoc(), call.getResults());
  };

  replace_with_call(FunctionType::get(types, {i1_tensor}, &getContext()),
                    GetName(while_op.getOperation(), "_cond"), while_op.cond(),
                    false);
  replace_with_call(FunctionType::get(types, types, &getContext()),
                    GetName(while_op.getOperation(), "_body"), while_op.body(),
                    true);

  // If there are extern values used then the result type of the while has to
  // change, so replace with new while op.
  if (extern_values.empty()) return;

  Operation* op = while_op.getOperation();
  SmallVector<Value, 4> operands;
  SmallVector<Type, 4> new_types;
  operands.reserve(op->getNumOperands() + extern_values.size());
  new_types.reserve(operands.size());
  auto add_operand = [&](Value v) {
    operands.push_back(v);
    new_types.push_back(v.getType());
  };
  for (auto operand : op->getOperands()) add_operand(operand);
  for (auto operand : extern_values) add_operand(operand);

  Operation* new_op = OpBuilder(op).insert(Operation::create(
      op->getLoc(), op->getName(), new_types, operands, op->getAttrs(),
      /*successors=*/{}, /*numRegions=*/2,
      /*resizableOperandList=*/true));
  for (int i = 0; i < 2; ++i) new_op->getRegion(i).takeBody(op->getRegion(i));
  op->replaceAllUsesWith(new_op->getResults().take_front(op->getNumResults()));
  op->erase();
}

void WhileOutlinePass::runOnModule() {
  getModule().walk(
      [&](mlir::TFL::WhileOp while_op) { OutlineWhile(while_op); });
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect WhileOp outline pass.
std::unique_ptr<OpPassBase<ModuleOp>> CreateWhileOutlinePass() {
  return std::make_unique<WhileOutlinePass>();
}

static PassRegistration<WhileOutlinePass> pass(
    "tfl-while-loop-outline", "Hoist while op regions into functions");

}  // namespace TFL
}  // namespace mlir
