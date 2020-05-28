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

// This file implements logic for lowering TensorFlow dialect's control flow to
// the XLA dialect.

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/core/util/tensor_format.h"

using mlir::PassRegistration;

namespace mlir {
namespace xla_hlo {
namespace {
class LegalizeTFControlFlow
    : public PassWrapper<LegalizeTFControlFlow, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override;
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLegalizeTFControlFlowPass() {
  return std::make_unique<LegalizeTFControlFlow>();
}

namespace {

void Detuple(Value tuple, Operation::result_range replace, OpBuilder* builder) {
  // De-tuple the results of the xla hlo if result.
  for (auto result_it : llvm::enumerate(replace)) {
    auto get_tuple_value = builder->create<xla_hlo::GetTupleElementOp>(
        result_it.value().getLoc(), tuple, result_it.index());
    result_it.value().replaceAllUsesWith(get_tuple_value);
  }
}

// Imports the source region into the destination region. The XLA if
// operation only supports one argument per branch. Therefore any branch that
// requires additional arguments requires their values be tupled together. Then,
// to support multiple returns (as XLA only supports a single return value) the
// results of the if operation are tupled together.
void ImportXlaRegion(mlir::FuncOp func, Region* dest_region, Location loc,
                     bool tuple_return = true) {
  OpBuilder builder(dest_region);

  auto entry_block = builder.createBlock(dest_region);
  auto tuple_arg = entry_block->addArgument(
      builder.getTupleType(func.getType().getInputs()));
  llvm::SmallVector<Value, 4> detupled_args;
  detupled_args.reserve(func.getNumArguments());

  for (int64_t i = 0, s = func.getNumArguments(); i < s; i++) {
    auto extract = builder.create<GetTupleElementOp>(loc, tuple_arg, i);
    detupled_args.push_back(extract);
  }

  auto result = builder.create<CallOp>(loc, func, detupled_args).getResults();
  if (!tuple_return) {
    builder.create<xla_hlo::ReturnOp>(loc, result);
  } else {
    auto tuple_op = builder.create<TupleOp>(loc, result);
    builder.create<xla_hlo::ReturnOp>(loc, tuple_op.getResult());
  }
}

void LowerIf(TF::IfOp op, ModuleOp module) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // XLA prefers tuple arguments for control flow due to XLA not supporting
  // multiple return values.
  SmallVector<Value, 3> inputs(op.input());
  auto tuple_input = builder.create<xla_hlo::TupleOp>(loc, inputs);

  // Create the new if op with tuple inputs.
  auto result_type = builder.getTupleType(op.getResultTypes());
  auto if_op = builder.create<xla_hlo::IfOp>(loc, result_type, op.cond(),
                                             tuple_input, tuple_input);

  // Import the regions for both the true and false cases. These regions
  // must be updated to tuple the return results together and use the xla hlo
  // return op.
  auto then_branch = module.lookupSymbol<mlir::FuncOp>(op.then_branch());
  auto else_branch = module.lookupSymbol<mlir::FuncOp>(op.else_branch());
  ImportXlaRegion(then_branch, &if_op.true_branch(), loc);
  ImportXlaRegion(else_branch, &if_op.false_branch(), loc);

  // De-tuple the results of the xla hlo if result.
  Detuple(if_op.getResult(), op.getResults(), &builder);
  op.erase();
}

void LowerCase(TF::CaseOp op, ModuleOp module) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // XLA requires one argument per branch so we create a tuple of inputs to pass
  // to each branch.
  SmallVector<Value, 4> inputs(op.input());
  auto tuple_input = builder.create<xla_hlo::TupleOp>(loc, inputs);

  // Create replica of input tuple for each branch
  SmallVector<Value, 4> n_tuple_inputs(op.branches().size(), tuple_input);

  // Create the new case op with tuple inputs.
  auto case_op = builder.create<xla_hlo::CaseOp>(
      loc, op.getResultTypes(), op.branch_index(), n_tuple_inputs,
      op.branches().size());

  // Import the regions for all branches.
  for (unsigned i = 0; i < op.branches().size(); ++i) {
    mlir::FuncOp branch_func = module.lookupSymbol<mlir::FuncOp>(
        op.branches()[i].cast<SymbolRefAttr>());
    ImportXlaRegion(branch_func, &case_op.branches()[i], loc,
                    /*tuple_return=*/false);
  }

  op.replaceAllUsesWith(case_op.getResults());
  op.erase();
}

void LowerWhile(TF::WhileOp op, ModuleOp module) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // XLA prefers tuple arguments for control flow due to XLA not supporting
  // multiple return values.
  SmallVector<Value, 3> inputs(op.input());
  builder.setInsertionPoint(op);
  Value tuple_input = builder.create<xla_hlo::TupleOp>(loc, inputs);

  // Create the new while op with tuple inputs.
  auto while_op = builder.create<xla_hlo::WhileOp>(
      loc, builder.getTupleType(op.getResultTypes()), tuple_input);

  // Import the regions for both the cond and body. These regions must be
  // updated to tuple the return results together and use the xla hlo return op.
  auto body_branch = module.lookupSymbol<mlir::FuncOp>(op.body());
  auto cond_branch = module.lookupSymbol<mlir::FuncOp>(op.cond());

  ImportXlaRegion(body_branch, &while_op.body(), loc);
  ImportXlaRegion(cond_branch, &while_op.cond(), loc, /*tuple_return=*/false);

  // De-tuple the results of the xla hlo while.
  Detuple(while_op.getResult(), op.getResults(), &builder);
  op.erase();
}
}  // namespace

void LegalizeTFControlFlow::runOnOperation() {
  auto module = getOperation();

  module.walk([&](Operation* op) {
    if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
      LowerWhile(while_op, module);
      return;
    }
    if (auto if_op = dyn_cast<TF::IfOp>(op)) {
      LowerIf(if_op, module);
      return;
    }
    if (auto case_op = dyn_cast<TF::CaseOp>(op)) {
      LowerCase(case_op, module);
      return;
    }
  });
}
}  // namespace xla_hlo
}  // namespace mlir

static PassRegistration<mlir::xla_hlo::LegalizeTFControlFlow> cfpass(
    "xla-legalize-tf-control-flow",
    "Legalize TensorFlow control flow to the XLA dialect");
