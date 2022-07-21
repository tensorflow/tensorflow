/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <iterator>
#include <memory>
#include <tuple>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

using mlir::func::FuncOp;

namespace mlir {
namespace TF {

namespace {

struct DecomposeReduceDatasetPass
    : public DecomposeReduceDatasetPassBase<DecomposeReduceDatasetPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<tf_device::TensorFlowDeviceDialect>();
  }

  void runOnOperation() override;
};

// Create the AnonymousIterator for `reduce_dataset` with `dataset_types` using
// `builder`.
AnonymousIteratorV3Op CreateIterator(OpBuilder builder,
                                     llvm::ArrayRef<Type> dataset_types,
                                     ReduceDatasetOp reduce_dataset) {
  llvm::SmallVector<Attribute, 2> shape_attrs;
  llvm::SmallVector<Attribute, 2> type_attrs;
  for (Type type : dataset_types) {
    shape_attrs.push_back(
        TF::ShapeAttr::get(builder.getContext(), type.cast<ShapedType>()));
    type_attrs.push_back(TypeAttr::get(getElementTypeOrSelf(type)));
  }

  auto anonymous_iterator = builder.create<AnonymousIteratorV3Op>(
      reduce_dataset.getLoc(),
      RankedTensorType::get({}, builder.getType<ResourceType>()),
      /*output_types=*/builder.getArrayAttr(type_attrs),
      /*shape_types=*/builder.getArrayAttr(shape_attrs));
  builder.create<MakeIteratorOp>(reduce_dataset.getLoc(),
                                 reduce_dataset.input_dataset(),
                                 anonymous_iterator.getResult());
  return anonymous_iterator;
}

// Create a WhileRegionOp turning `reduce_dataset` into a dataset iteration with
// reduce_fn call.
WhileRegionOp CreateDatasetWhile(OpBuilder builder,
                                 ReduceDatasetOp reduce_dataset) {
  auto const_true = builder.create<TF::ConstOp>(
      reduce_dataset.getLoc(),
      DenseIntElementsAttr::get(
          RankedTensorType::get(/*shape=*/{}, builder.getI1Type()), true));

  SmallVector<Value, 4> while_input_values;
  SmallVector<Type, 4> while_input_types;
  while_input_values.push_back(const_true.getResult());
  while_input_types.push_back(const_true.getResult().getType());
  for (int i = 1; i < reduce_dataset.getNumOperands(); ++i) {
    while_input_values.push_back(reduce_dataset.getOperand(i));
    while_input_types.push_back(reduce_dataset.getOperand(i).getType());
  }

  auto dataset_while = builder.create<TF::WhileRegionOp>(
      reduce_dataset.getLoc(), while_input_types, /*input=*/while_input_values,
      /*parallel_iterations=*/10, false,
      /*shape_invariant=*/false);

  // `_lower_using_switch_merge` is the default for While ops created
  // in TensorFlow and allows lowering to V1 control flow for loop
  // parallelization.
  dataset_while->setAttr("_lower_using_switch_merge",
                         builder.getBoolAttr(true));

  return dataset_while;
}

// Populate the cond of `dataset_while`.  The cond body just returns the
// condition of whether to continue to next iteration.
void PopulateDatasetWhileCond(OpBuilder builder, WhileRegionOp dataset_while,
                              Location loc) {
  auto& cond_region = dataset_while.cond();
  Block* cond_block = builder.createBlock(&cond_region);
  auto while_input_types = dataset_while.getOperandTypes();
  cond_block->addArguments(
      while_input_types, SmallVector<Location>(while_input_types.size(), loc));
  builder.create<YieldOp>(loc, cond_block->getArgument(0));
}

// Create an IfRegionOp with a predicate from `optional_has_value`.  If true, it
// uses `get_next` to get the next value and calls `reduce_func`.  `body_args`
// is used as pass through of state values for else branch.  `dataset_types` is
// used for constructing the CallOp for `reduce_func`.
IfRegionOp CreateOptionalDatasetIf(
    OpBuilder builder, ReduceDatasetOp reduce_dataset, FuncOp reduce_func,
    IteratorGetNextAsOptionalOp get_next, OptionalHasValueOp optional_has_value,
    ArrayRef<Value> body_args, ArrayRef<Type> dataset_types) {
  const Location loc = reduce_dataset.getLoc();
  // If returns are the state variables.
  SmallVector<Type, 4> if_return_types;
  const int state_size =
      reduce_dataset->getAttrOfType<ArrayAttr>("Tstate").size();
  for (int i = 1; i < state_size + 1; i++) {
    if_return_types.push_back(reduce_dataset.getOperand(i).getType());
  }

  auto dataset_if = builder.create<TF::IfRegionOp>(
      loc, if_return_types, optional_has_value.getResult(), false,
      /*_then_func_name=*/nullptr,
      /*_else_func_name=*/nullptr);
  // `_lower_using_switch_merge` allows lowering to V1 control flow for loop
  // parallelization.
  dataset_if->setAttr("_lower_using_switch_merge", builder.getBoolAttr(true));
  // Empty else branch, if there is no more data, do nothing.
  auto& else_branch = dataset_if.else_branch();
  else_branch.push_back(new Block);
  builder.setInsertionPointToEnd(&else_branch.front());
  // Return only the state variables from the body arguments.
  SmallVector<Value, 4> else_returns;
  for (int i = 1; i < state_size + 1; i++) {
    else_returns.push_back(body_args[i]);
  }
  builder.create<TF::YieldOp>(loc,
                              /*operands=*/else_returns);

  // Then branch gets the data and calls the reduce_function.
  auto& then_branch = dataset_if.then_branch();
  then_branch.push_back(new Block);
  builder.setInsertionPointToEnd(&then_branch.front());
  // Add iterator operational data access inside if.
  auto get_value = builder.create<TF::OptionalGetValueOp>(loc, dataset_types,
                                                          get_next.getResult());
  SmallVector<Value, 4> reduce_fn_args;

  // Function arguments are state values, dataset values, and then passthrough
  // arguments.
  // First argument to body is the while loop condition and state values start
  // at index=1.
  for (int i = 1; i < state_size + 1; ++i) {
    reduce_fn_args.push_back(body_args[i]);
  }
  for (Value value : get_value.getResults()) {
    reduce_fn_args.push_back(value);
  }
  for (int i = state_size + 1; i < body_args.size(); ++i) {
    reduce_fn_args.push_back(body_args[i]);
  }

  auto reduce_call =
      builder.create<mlir::func::CallOp>(loc, reduce_func, reduce_fn_args);

  reduce_call->setAttr(
      TF::kCompileDeviceTypeAttr,
      reduce_dataset->getAttrOfType<StringAttr>(TF::kCompileDeviceTypeAttr));

  SmallVector<Value, 4> if_returns;

  builder.create<TF::YieldOp>(loc,
                              /*operands=*/reduce_call.getResults());
  return dataset_if;
}

// Populates WhileRegionOp body which is replacing `reduce_dataset`.  Iterates
// `anonymous_iterator` with `dataset_types` and optional calls `reduce_func`.
void PopulateDatasetWhileBody(OpBuilder builder, ReduceDatasetOp reduce_dataset,
                              FuncOp reduce_func, WhileRegionOp dataset_while,
                              AnonymousIteratorV3Op anonymous_iterator,
                              ArrayRef<Type> dataset_types) {
  const Location loc = reduce_dataset.getLoc();
  auto while_input_types = dataset_while.getOperandTypes();
  auto& body_region = dataset_while.body();
  Block* body_block = builder.createBlock(&body_region);
  auto body_arguments = body_block->addArguments(
      while_input_types, SmallVector<Location>(while_input_types.size(), loc));
  auto get_next = builder.create<IteratorGetNextAsOptionalOp>(
      loc, RankedTensorType::get({}, builder.getType<VariantType>()),
      anonymous_iterator.getResult(), anonymous_iterator.output_types(),
      anonymous_iterator.output_shapes());
  auto optional_has_value = builder.create<OptionalHasValueOp>(
      loc, RankedTensorType::get({}, builder.getI1Type()),
      get_next.getResult());

  SmallVector<Value, 4> body_args;
  for (Value value : body_arguments) {
    body_args.push_back(value);
  }

  IfRegionOp dataset_if =
      CreateOptionalDatasetIf(builder, reduce_dataset, reduce_func, get_next,
                              optional_has_value, body_args, dataset_types);

  builder.setInsertionPointToEnd(body_block);
  // The body returns consist of the loop condition (whether the next iterator
  // has a value), the state returned by the IfRegionOp, and the pass through
  // values.
  SmallVector<Value, 4> body_returns;
  body_returns.push_back(optional_has_value.getResult());

  const int state_size =
      reduce_dataset->getAttrOfType<ArrayAttr>("Tstate").size();
  for (int i = 0; i < state_size; ++i) {
    body_returns.push_back(dataset_if.getResult(i));
  }
  // Copy the arguments but skip the states and the loop condition
  // which are updated in while body.
  for (int i = state_size + 1; i < body_args.size(); ++i) {
    body_returns.push_back(body_args[i]);
  }
  builder.create<TF::YieldOp>(loc,
                              /*operands=*/body_returns);
}

// Decomposes any ReduceDatasetOps in `function` into a dataset iteration and a
// call to the reduce function in the ReduceDatasetOp.
LogicalResult DecomposeReduceDatasetInFunction(FuncOp function) {
  if (!llvm::hasSingleElement(function))
    return function.emitOpError("Expecting a single block function");

  auto decompose_result = function.walk([&](ReduceDatasetOp reduce_dataset) {
    if (!reduce_dataset->hasAttrOfType<StringAttr>(TF::kCompileDeviceTypeAttr))
      return WalkResult::advance();
    OpBuilder builder(reduce_dataset);
    Location loc = reduce_dataset.getLoc();

    // Get reduce function signature for dataset iteration types.
    // Note: lookupSymbol is a linear lookup which means the overall
    // complexity = # ReduceDataset ops x # of functions in module.
    func::FuncOp reduce_func =
        function->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
            reduce_dataset.f());

    // The reduce function arguments consist of three part in this order:
    // 1. Reduction state inputs.
    // 2. Dataset inputs.
    // 3. Captures inputs.
    // The number of dataset inputs can be indirectly determined to be
    // total_number_of_inputs - state_inputs - captured_inputs.
    auto func_inputs = reduce_func.getFunctionType().getInputs();
    const int func_input_size = func_inputs.size();
    const int argument_size =
        reduce_dataset->getAttrOfType<ArrayAttr>("Targuments").size();
    const int state_size =
        reduce_dataset->getAttrOfType<ArrayAttr>("Tstate").size();
    const int dataset_input_size = func_input_size - state_size - argument_size;

    SmallVector<Type, 2> dataset_types;
    for (int i = 0; i < dataset_input_size; ++i) {
      dataset_types.push_back(func_inputs[state_size + i]);
    }

    // Create dataset iterator and iterate dataset in while loop which calls
    // reduce_fn.
    AnonymousIteratorV3Op anonymous_iterator =
        CreateIterator(builder, dataset_types, reduce_dataset);
    WhileRegionOp dataset_while = CreateDatasetWhile(builder, reduce_dataset);
    PopulateDatasetWhileCond(builder, dataset_while, loc);
    PopulateDatasetWhileBody(builder, reduce_dataset, reduce_func,
                             dataset_while, anonymous_iterator, dataset_types);

    // Updates usage and erases rewritten reduce_dataset op.
    reduce_dataset.getResult(0).replaceAllUsesWith(dataset_while.getResult(1));
    reduce_dataset.erase();

    return WalkResult::advance();
  });

  return failure(decompose_result.wasInterrupted());
}

void DecomposeReduceDatasetPass::runOnOperation() {
  if (failed(DecomposeReduceDatasetInFunction(getOperation()))) {
    return signalPassFailure();
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateDecomposeReduceDatasetPass() {
  return std::make_unique<DecomposeReduceDatasetPass>();
}

}  // namespace TF
}  // namespace mlir
