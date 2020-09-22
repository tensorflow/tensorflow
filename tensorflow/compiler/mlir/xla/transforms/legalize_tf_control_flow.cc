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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Module.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/util/tensor_format.h"

using mlir::PassRegistration;

namespace mlir {
namespace mhlo {
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

void Detuple(Value tuple, ValueRange replace, OpBuilder* builder) {
  // De-tuple the results of the xla hlo if result.
  for (auto result_it : llvm::enumerate(replace)) {
    auto get_tuple_value = builder->create<mhlo::GetTupleElementOp>(
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
    builder.create<mhlo::ReturnOp>(loc, result);
  } else {
    auto tuple_op = builder.create<TupleOp>(loc, result);
    builder.create<mhlo::ReturnOp>(loc, tuple_op.getResult());
  }
}

void LowerIf(TF::IfOp op, ModuleOp module) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // XLA prefers tuple arguments for control flow due to XLA not supporting
  // multiple return values.
  SmallVector<Value, 3> inputs(op.input());
  auto tuple_input = builder.create<mhlo::TupleOp>(loc, inputs);

  // Create the new if op with tuple inputs.
  auto result_type = builder.getTupleType(op.getResultTypes());
  auto if_op = builder.create<mhlo::IfOp>(loc, result_type, op.cond(),
                                          tuple_input, tuple_input);

  // Import the regions for both the true and false cases. These regions
  // must be updated to tuple the return results together and use the xla hlo
  // return op.
  ImportXlaRegion(op.then_function(), &if_op.true_branch(), loc);
  ImportXlaRegion(op.else_function(), &if_op.false_branch(), loc);

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
  auto tuple_input = builder.create<mhlo::TupleOp>(loc, inputs);

  // Create replica of input tuple for each branch
  SmallVector<Value, 4> n_tuple_inputs(op.num_branches(), tuple_input);

  // Create the new case op with tuple inputs.
  auto case_op =
      builder.create<mhlo::CaseOp>(loc, op.getResultTypes(), op.branch_index(),
                                   n_tuple_inputs, op.branches().size());

  // Import the regions for all branches.
  for (unsigned i = 0; i < op.num_branches(); ++i) {
    mlir::FuncOp branch_func = op.branch_function(i);
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
  Value tuple_input = builder.create<mhlo::TupleOp>(loc, inputs);

  // Create the new while op with tuple inputs.
  auto while_op = builder.create<mhlo::WhileOp>(
      loc, builder.getTupleType(op.getResultTypes()), tuple_input);

  // Import the regions for both the cond and body. These regions must be
  // updated to tuple the return results together and use the xla hlo return op.
  ImportXlaRegion(op.body_function(), &while_op.body(), loc);
  ImportXlaRegion(op.cond_function(), &while_op.cond(), loc,
                  /*tuple_return=*/false);

  // De-tuple the results of the xla hlo while.
  Detuple(while_op.getResult(), op.getResults(), &builder);
  op.erase();
}

// Replaces all block arguments of a block with a single block arg of Tuple
// type `tuple_type`. Single block arguments are removed and remapped to
// get_tuple_element(tuple_arg, index).
void ReplaceBlockArgs(Block* block, Type tuple_type, OpBuilder* builder) {
  auto tuple_arg = block->addArgument(tuple_type);
  Detuple(tuple_arg, block->getArguments().drop_back(1), builder);
  for (int i = block->getNumArguments() - 2; i >= 0; --i)
    block->eraseArgument(i);
}

// Finds and replaces implicitly captured value uses with tuple block argument.
// get_tuple_element's are created to extract specific values. Values from
// get_tuple_element's are returned in the order of `implicit_inputs`.
llvm::SmallVector<Value, 4> ReplaceImplicitInputs(
    Block* block, int offset, ArrayRef<Value> implicit_inputs,
    OpBuilder* builder) {
  llvm::SmallVector<Value, 4> implicit_input_elements;
  implicit_input_elements.reserve(implicit_inputs.size());

  Region* region = block->getParent();
  assert(block->getNumArguments() == 1);

  BlockArgument tuple_arg = block->getArgument(0);
  for (auto& implicit_input : llvm::enumerate(implicit_inputs)) {
    Value implicit_input_value = implicit_input.value();
    auto get_tuple_element = builder->create<mhlo::GetTupleElementOp>(
        implicit_input_value.getLoc(), tuple_arg,
        implicit_input.index() + offset);
    implicit_input_elements.emplace_back(get_tuple_element.getResult());
    for (auto& use :
         llvm::make_early_inc_range(implicit_input_value.getUses())) {
      if (!region->isAncestor(use.getOwner()->getParentRegion())) continue;
      use.set(get_tuple_element.getResult());
    }
  }

  return implicit_input_elements;
}

// Replaces block terminator (tf.Yield) with `mhlo.return`. Additional results
// can be returned if `extra_results` is not empty. If `tuple_return` is
// set, a tuple of the return values will be set as the terminator operand.
void ReplaceTerminator(Block* block, ArrayRef<Value> extra_results,
                       OpBuilder* builder, bool tuple_return = true) {
  Operation* terminator = block->getTerminator();
  assert(isa<TF::YieldOp>(terminator));
  Location loc = terminator->getLoc();

  builder->setInsertionPoint(terminator);
  auto results = llvm::to_vector<4>(terminator->getOperands());
  results.append(extra_results.begin(), extra_results.end());
  if (tuple_return) {
    auto tuple_results = builder->create<mhlo::TupleOp>(loc, results);
    builder->create<mhlo::ReturnOp>(loc, tuple_results.getResult());
  } else {
    builder->create<mhlo::ReturnOp>(loc, results);
  }

  terminator->erase();
}

void LowerWhileRegion(TF::WhileRegionOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // XLA prefers tuple arguments for control flow due to XLA not supporting
  // multiple return values.
  SmallVector<Value, 3> inputs(op.input());
  const int inputs_size = inputs.size();
  llvm::SetVector<Value> implicit_inputs;
  getUsedValuesDefinedAbove(op.getOperation()->getRegions(), implicit_inputs);
  inputs.append(implicit_inputs.begin(), implicit_inputs.end());

  builder.setInsertionPoint(op);
  Value tuple_input = builder.create<mhlo::TupleOp>(loc, inputs);

  // Create the new while op with tuple inputs. Implicit inputs are also
  // returned.
  auto while_result_types = llvm::to_vector<4>(op.getResultTypes());
  while_result_types.reserve(while_result_types.size() +
                             implicit_inputs.size());
  for (const auto& implicit_input : implicit_inputs)
    while_result_types.emplace_back(implicit_input.getType());
  auto while_op = builder.create<mhlo::WhileOp>(
      loc, builder.getTupleType(while_result_types), tuple_input);

  // Rewrite cond and associated block arguments and terminator.
  Region& cond = while_op.cond();
  cond.takeBody(op.cond());
  Block& cond_block = cond.front();
  builder.setInsertionPointToStart(&cond_block);
  ReplaceBlockArgs(&cond_block, tuple_input.getType(), &builder);
  ReplaceImplicitInputs(&cond_block, inputs_size, implicit_inputs.getArrayRef(),
                        &builder);
  // Cond always returns a single result of bool type.
  ReplaceTerminator(&cond_block, /*extra_results=*/{}, &builder,
                    /*tuple_return=*/false);

  // Rewrite body and associated block arguments and terminator.
  Region& body = while_op.body();
  body.takeBody(op.body());
  Block& body_block = body.front();
  builder.setInsertionPointToStart(&body_block);
  ReplaceBlockArgs(&body_block, tuple_input.getType(), &builder);
  // Capture implicit inputs that were added as a tuple block arguments. These
  // are to be returned by the body in addition to explicit inputs.
  auto implicit_input_elements = ReplaceImplicitInputs(
      &body_block, inputs_size, implicit_inputs.getArrayRef(), &builder);
  ReplaceTerminator(&body_block, implicit_input_elements, &builder);

  // De-tuple the results of the xla hlo while.
  builder.setInsertionPoint(op);
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
    if (auto while_region_op = dyn_cast<TF::WhileRegionOp>(op)) {
      LowerWhileRegion(while_region_op);
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
}  // namespace mhlo
}  // namespace mlir

static PassRegistration<mlir::mhlo::LegalizeTFControlFlow> cfpass(
    "xla-legalize-tf-control-flow",
    "Legalize TensorFlow control flow to the XLA dialect");
