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
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_legalize_tf_passes_detail.h"

using mlir::PassRegistration;

namespace mlir {
namespace mhlo {
namespace {
class LegalizeTFControlFlow
    : public LegalizeTFControlFlowBase<LegalizeTFControlFlow> {
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

void LowerIf(TF::IfOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // XLA prefers tuple arguments for control flow due to XLA not supporting
  // multiple return values.
  SmallVector<Value, 3> inputs(op.input());
  auto tuple_input = builder.create<mhlo::TupleOp>(loc, inputs);

  // Create the new `mhlo.if` op with tuple inputs.
  auto result_type = builder.getTupleType(op.getResultTypes());
  auto if_op = builder.create<mhlo::IfOp>(loc, result_type, op.cond(),
                                          tuple_input, tuple_input);

  // Import the regions for both the true and false cases. These regions
  // must be updated to tuple the return results together and use the xla hlo
  // return op.
  ImportXlaRegion(op.then_function(), &if_op.true_branch(), loc);
  ImportXlaRegion(op.else_function(), &if_op.false_branch(), loc);

  // De-tuple the results of the `mhlo.if`.
  Detuple(if_op.getResult(), op.getResults(), &builder);
  op.erase();
}

void LowerCase(TF::CaseOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // XLA requires one argument per branch so we create a tuple of inputs to pass
  // to each branch.
  SmallVector<Value, 4> inputs(op.input());
  auto tuple_input = builder.create<mhlo::TupleOp>(loc, inputs);

  // Create replica of input tuple for each branch
  SmallVector<Value, 4> n_tuple_inputs(op.num_branches(), tuple_input);

  // Create the new `mhlo.case` op with tuple inputs.
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

void LowerWhile(TF::WhileOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // XLA prefers tuple arguments for control flow due to XLA not supporting
  // multiple return values.
  SmallVector<Value, 3> inputs(op.input());
  builder.setInsertionPoint(op);
  Value tuple_input = builder.create<mhlo::TupleOp>(loc, inputs);

  // Create the new `mhlo.while` op with tuple inputs.
  auto while_op = builder.create<mhlo::WhileOp>(
      loc, builder.getTupleType(op.getResultTypes()), tuple_input);

  // Import the regions for both the cond and body. These regions must be
  // updated to tuple the return results together and use the xla hlo return op.
  ImportXlaRegion(op.body_function(), &while_op.body(), loc);
  ImportXlaRegion(op.cond_function(), &while_op.cond(), loc,
                  /*tuple_return=*/false);

  // De-tuple the results of the `mhlo.while` if needed.
  if (while_op.getNumResults() == 1 && while_op.getType(0).isa<TupleType>())
    Detuple(while_op.getResult(0), op.getResults(), &builder);
  else
    op->replaceAllUsesWith(while_op);
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

// Replaces implicitly captured value uses with tuple block argument.
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

// Finds and replaces implicitly captured value uses with tuple block argument.
// A tuple of implicitly captured values is also created and returned, for use
// as an operand to the associated mhlo control flow op.
Value TupleImplicitInputs(Region& region, Location loc, OpBuilder* builder) {
  llvm::SetVector<Value> implicit_inputs;
  getUsedValuesDefinedAbove(region, region, implicit_inputs);
  llvm::ArrayRef<Value> implicit_inputs_ref = implicit_inputs.getArrayRef();
  Value tuple_input = builder->create<mhlo::TupleOp>(loc, implicit_inputs_ref);
  Block& block = region.front();
  // `tf.CaseRegion`/`tf.IfRegion` are expected to have no block arguments and
  // instead all inputs used by their branch regions are implicitly captured
  // from above.
  assert(block.getNumArguments() == 0);
  block.addArgument(tuple_input.getType());
  builder->setInsertionPointToStart(&block);
  ReplaceImplicitInputs(&block, /*offset=*/0, implicit_inputs_ref, builder);
  return tuple_input;
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

void LowerIfRegion(TF::IfRegionOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  // Tuple implicit inputs per region and update terminators to return tuples.
  builder.setInsertionPoint(op);
  Value then_input = TupleImplicitInputs(op.then_branch(), loc, &builder);
  ReplaceTerminator(&op.then_branch().front(), /*extra_results=*/{}, &builder);

  builder.setInsertionPoint(op);
  Value else_input = TupleImplicitInputs(op.else_branch(), loc, &builder);
  ReplaceTerminator(&op.else_branch().front(), /*extra_results=*/{}, &builder);

  // Create the new `mhlo.if` op with tuple inputs and take ownership of regions
  // from `tf.IfRegion` op.
  builder.setInsertionPoint(op);
  auto result_type = builder.getTupleType(op.getResultTypes());
  auto if_op = builder.create<mhlo::IfOp>(loc, result_type, op.cond(),
                                          then_input, else_input);
  if_op.true_branch().takeBody(op.then_branch());
  if_op.false_branch().takeBody(op.else_branch());

  // De-tuple the results of the `mhlo.if`.
  Detuple(if_op.getResult(), op.getResults(), &builder);
  op.erase();
}

void LowerCaseRegion(TF::CaseRegionOp op) {
  Location loc = op.getLoc();
  OpBuilder builder(op);

  llvm::SmallVector<Value, 4> branch_inputs;
  branch_inputs.reserve(op.branches().size());
  // Tuple implicit inputs per region and update terminators.
  for (Region& region : op.branches()) {
    builder.setInsertionPoint(op);
    Value branch_input = TupleImplicitInputs(region, loc, &builder);
    branch_inputs.emplace_back(branch_input);
    ReplaceTerminator(&region.front(), /*extra_results=*/{}, &builder,
                      /*tuple_return=*/false);
  }

  // Create the new `mhlo.case` op with tuple inputs and take ownership of
  // regions from `tf.CaseRegion` op.
  builder.setInsertionPoint(op);
  auto case_op =
      builder.create<mhlo::CaseOp>(loc, op.getResultTypes(), op.branch_index(),
                                   branch_inputs, branch_inputs.size());
  for (auto region : llvm::zip(case_op.branches(), op.branches()))
    std::get<0>(region).takeBody(std::get<1>(region));

  op.replaceAllUsesWith(case_op.getResults());
  op.erase();
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

  // Create the new `mhlo.while` op with tuple inputs. Implicit inputs are also
  // returned.
  auto while_result_types = llvm::to_vector<4>(op.getResultTypes());
  while_result_types.reserve(while_result_types.size() +
                             implicit_inputs.size());
  for (const auto& implicit_input : implicit_inputs)
    while_result_types.emplace_back(implicit_input.getType());
  auto while_op = builder.create<mhlo::WhileOp>(
      loc, builder.getTupleType(while_result_types), tuple_input);

  // Rewrite cond and associated block arguments and terminator. Ownership of
  // cond region is transfered over from `tf.WhileRegion` to `mhlo.while`.
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

  // Rewrite body and associated block arguments and terminator. Ownership of
  // body region is transfered over from `tf.WhileRegion` to `mhlo.while`.
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

  // De-tuple the results of the `mhlo.while`.
  builder.setInsertionPoint(op);
  if (while_op.getNumResults() == 1 && while_op.getType(0).isa<TupleType>())
    Detuple(while_op.getResult(0), op.getResults(), &builder);
  else
    op->replaceAllUsesWith(while_op);
  op.erase();
}
}  // namespace

void LegalizeTFControlFlow::runOnOperation() {
  getOperation().walk([&](Operation* op) {
    if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
      LowerWhile(while_op);
      return;
    }
    if (auto while_region_op = dyn_cast<TF::WhileRegionOp>(op)) {
      LowerWhileRegion(while_region_op);
      return;
    }
    if (auto if_op = dyn_cast<TF::IfOp>(op)) {
      LowerIf(if_op);
      return;
    }
    if (auto if_region_op = dyn_cast<TF::IfRegionOp>(op)) {
      LowerIfRegion(if_region_op);
      return;
    }
    if (auto case_op = dyn_cast<TF::CaseOp>(op)) {
      LowerCase(case_op);
      return;
    }
    if (auto case_region_op = dyn_cast<TF::CaseRegionOp>(op)) {
      LowerCaseRegion(case_region_op);
      return;
    }
  });
}
}  // namespace mhlo
}  // namespace mlir
