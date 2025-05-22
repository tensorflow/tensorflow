/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_freeze_utils.h"

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/CallInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/framework/types.pb.h"

namespace mlir {
namespace tf_saved_model {

namespace {
// Attribute name that specifies the input shapes of a function.
constexpr StringRef kTfInputShapesAttr = "tf._input_shapes";

// Updates terminator op arguments of 'func' after removing arguments
// specified in 'arguments_to_erase'.
template <typename T>
void UpdateTerminatorArguments(T& func,
                               const ArrayRef<unsigned> arguments_to_erase,
                               llvm::BitVector& erase_indices) {
  auto terminator = func.front().getTerminator();
  int num_operands = terminator->getNumOperands();
  erase_indices.resize(num_operands);
  for (auto arg_index : arguments_to_erase) {
    auto argument = func.getArgument(arg_index);
    for (auto& use : argument.getUses()) {
      if (llvm::isa<func::ReturnOp, TF::YieldOp>(use.getOwner())) {
        erase_indices.set(use.getOperandNumber());
      }
    }
    func.getArgument(arg_index).dropAllUses();
  }
  if (llvm::isa<func::ReturnOp, TF::YieldOp>(func.front().getTerminator())) {
    terminator->eraseOperands(erase_indices);
  }
}

// Erases function arguments indexed at `args_to_erase`. Also applies the
// changes to any relevant function attributes accordingly.
LogicalResult EraseFuncOpArguments(func::FuncOp func_op,
                                   const ArrayRef<unsigned> args_to_erase) {
  BitVector args_to_erase_bit_vector(func_op.getNumArguments());
  for (const unsigned i : args_to_erase) args_to_erase_bit_vector.set(i);

  if (failed(func_op.eraseArguments(args_to_erase_bit_vector))) {
    return failure();
  }

  // Erases entries in "tf._input_shapes" attribute of `func_op` that correspond
  // to the erased arguments.
  if (auto input_shapes_attr =
          func_op->getAttrOfType<ArrayAttr>(kTfInputShapesAttr);
      input_shapes_attr) {
    // Construct a new array of input shapes excluding the input shapes of the
    // erased arguments.
    SmallVector<Attribute> updated_input_shapes_attr;
    for (const unsigned i : args_to_erase_bit_vector.flip().set_bits()) {
      updated_input_shapes_attr.emplace_back(input_shapes_attr[i]);
    }

    // Replaces the attribute with the updated "#tf_type.shape" array.
    // Builder builder(func_op.getContext());
    func_op->setAttr(
        kTfInputShapesAttr,
        ArrayAttr::get(func_op.getContext(), updated_input_shapes_attr));
  }
  return success();
}

// Updates 'while_op' signatures based on which arguments should be removed
// in 'arguments_to_erase'.
template <typename T, typename U>
T GetUpdatedWhileOp(T while_op, const U& argument_types,
                    const llvm::SmallVector<unsigned, 4>& arguments_to_erase) {
  OpBuilder builder(while_op);
  llvm::SmallVector<Type, 4> new_operand_types;
  llvm::SmallVector<Value> new_operands;
  auto operands = while_op->getOperands();
  const int num_operands = while_op->getNumOperands();
  llvm::BitVector skip_indices(num_operands);
  for (int i : arguments_to_erase) skip_indices.set(i);
  for (int i = 0; i < num_operands; ++i) {
    if (!skip_indices.test(i)) {
      new_operand_types.emplace_back(argument_types[i]);
      new_operands.emplace_back(operands[i]);
    }
  }
  auto new_while_op = builder.create<T>(while_op->getLoc(), new_operand_types,
                                        new_operands, while_op->getAttrs());
  int new_index = 0;
  for (int i = 0; i < num_operands; ++i) {
    if (!skip_indices.test(i)) {
      while_op->getResult(i).replaceAllUsesWith(
          new_while_op->getResult(new_index++));
    }
  }
  return new_while_op;
}

// Creates a constant op that holds 'tensor_elements'.
TF::ConstOp GetConstOpFromElementsAttr(ElementsAttr tensor_elements,
                                       OpBuilder builder, Location loc) {
  return builder.create<TF::ConstOp>(loc, tensor_elements);
}

// Replace usage of 'read_variable_op' with 'value'.
void PropagateUsage(TF::ReadVariableOp read_variable_op, ElementsAttr value) {
  OpBuilder builder(read_variable_op);
  read_variable_op->getResult(0).replaceAllUsesWith(
      GetConstOpFromElementsAttr(value, builder, read_variable_op->getLoc()));
}

// Propagates a resource usage across the graph where
// 'user_op' uses a resource and is passed to this op at 'argument_index'.
// This resource should be replaced by 'value'.
// Output params:
// - work_list: Is updated with new regions to process that is called
//   by 'user_op';
// - arguments_to_erase: Captures updates to the graph - which arguments
//   to remove from the op;
LogicalResult PropagateUsage(
    Operation* user_op, int argument_index, ElementsAttr value,
    llvm::SmallVector<std::pair<Region*, int>, 4>* work_list,
    llvm::MapVector<Operation*, llvm::SmallVector<unsigned int, 4>>*
        arguments_to_erase) {
  if (auto read_variable_op = dyn_cast<TF::ReadVariableOp>(user_op)) {
    (*arguments_to_erase)[read_variable_op];
    PropagateUsage(read_variable_op, value);
  } else if (auto call = dyn_cast<CallOpInterface>(user_op)) {
    (*arguments_to_erase)[call].push_back(argument_index);
    if (auto func = dyn_cast<func::FuncOp>(call.resolveCallable())) {
      (*arguments_to_erase)[func].push_back(argument_index);
      work_list->push_back(std::make_pair(&func.getRegion(), argument_index));
    }
  } else if (auto if_op = dyn_cast<TF::IfOp>(user_op)) {
    (*arguments_to_erase)[if_op].push_back(argument_index);
    for (auto callee : {if_op.then_function(), if_op.else_function()}) {
      (*arguments_to_erase)[callee].push_back(argument_index - 1);
      work_list->push_back(
          std::make_pair(&callee.getBody(), argument_index - 1));
    }
  } else if (auto if_op = dyn_cast<TF::IfRegionOp>(user_op)) {
    (*arguments_to_erase)[if_op].push_back(argument_index);
    for (auto callee : {&if_op.getThenBranch(), &if_op.getElseBranch()}) {
      work_list->push_back(std::make_pair(callee, argument_index));
    }
  } else if (auto while_op = dyn_cast<TF::WhileOp>(user_op)) {
    (*arguments_to_erase)[while_op].push_back(argument_index);
    for (auto callee : {while_op.cond_function(), while_op.body_function()}) {
      (*arguments_to_erase)[callee].push_back(argument_index);
      work_list->push_back(std::make_pair(&callee.getBody(), argument_index));
    }
  } else if (auto while_op = dyn_cast<TF::WhileRegionOp>(user_op)) {
    (*arguments_to_erase)[while_op].push_back(argument_index);
    for (auto callee : {&while_op.getCond(), &while_op.getBody()}) {
      work_list->push_back(std::make_pair(callee, argument_index));
    }
  } else if (auto batch_func_op = dyn_cast<TF::BatchFunctionOp>(user_op)) {
    (*arguments_to_erase)[batch_func_op].push_back(argument_index);
    // Add the called function to the work list.
    func::FuncOp func_op = batch_func_op.func();
    (*arguments_to_erase)[func_op].push_back(argument_index);
    work_list->push_back({&func_op.getRegion(), argument_index});
  } else {
    // Return and yield ops are the only ops that use the resource outside of
    // the above ops.
    if (!(isa<func::ReturnOp>(user_op) || isa<TF::YieldOp>(user_op))) {
      user_op->emitError() << "could not rewrite use of immutable bound input";
      return failure();
    }
  }

  return success();
}

// An override that takes region.
LogicalResult PropagateUsage(
    Region* region, ElementsAttr value, int argument_index,
    llvm::SmallVector<std::pair<Region*, int>, 4>* work_list,
    llvm::MapVector<Operation*, llvm::SmallVector<unsigned int, 4>>*
        arguments_to_erase) {
  auto arg = region->getArgument(argument_index);
  for (auto& usage : arg.getUses()) {
    auto* user_op = usage.getOwner();
    int operand_index = usage.getOperandNumber();
    if (failed(PropagateUsage(user_op, operand_index, value, work_list,
                              arguments_to_erase))) {
      return failure();
    }
  }

  return success();
}
}  // namespace

LogicalResult EraseObsoleteResourceUses(
    llvm::MapVector<Operation*, llvm::SmallVector<unsigned int, 4>>
        arguments_to_erase) {
  // All updates to different ops are captured in 'arguments_to_erase'.
  // Now loop on them and based on each item type update accordingly.
  for (auto& [user_op, args_to_erase] : arguments_to_erase) {
    if (auto func = dyn_cast<func::FuncOp>(user_op)) {
      // To update a function we will need to:
      // 1) Remove the unused arguments from the function itself.
      //    1-2) Remove func attributes corresponding to the removed arguments.
      // 2) Remove any returns that are not needed from the function terminator
      //    op in the function.
      // 3) Update function result to match the terminator.
      llvm::BitVector result_indices_to_erase;
      UpdateTerminatorArguments(func, args_to_erase, result_indices_to_erase);
      if (failed(EraseFuncOpArguments(func, args_to_erase))) {
        return failure();
      }

      if (failed(func.eraseResults(result_indices_to_erase))) {
        return failure();
      }
    } else if (auto read_var = dyn_cast<TF::ReadVariableOp>(user_op)) {
      // Read variables was already replaced by constant op. Just remove the op.
      read_var->erase();
    } else if (auto while_op = dyn_cast<TF::WhileOp>(user_op)) {
      GetUpdatedWhileOp<TF::WhileOp>(
          while_op, while_op.cond_function().getArgumentTypes(), args_to_erase);
      while_op->erase();
    } else if (auto while_op = dyn_cast<TF::WhileRegionOp>(user_op)) {
      auto new_while_op = GetUpdatedWhileOp(
          while_op, while_op.getCond().getArgumentTypes(), args_to_erase);
      new_while_op.getCond().takeBody(while_op.getCond());
      new_while_op.getBody().takeBody(while_op.getBody());
      llvm::BitVector erase_indices;
      UpdateTerminatorArguments(new_while_op.getBody(), args_to_erase,
                                erase_indices);
      llvm::BitVector body_bit_vector(
          new_while_op.getBody().front().getNumArguments());
      for (auto i : args_to_erase) body_bit_vector.set(i);
      new_while_op.getBody().front().eraseArguments(body_bit_vector);
      llvm::BitVector cond_bit_vector(
          new_while_op.getCond().front().getNumArguments());
      for (auto i : args_to_erase) cond_bit_vector.set(i);
      new_while_op.getCond().front().eraseArguments(cond_bit_vector);
      while_op->erase();
    } else if (auto batch_func_op = dyn_cast<TF::BatchFunctionOp>(user_op)) {
      llvm::BitVector erase_indices(user_op->getNumOperands());
      for (auto operand_index : args_to_erase) {
        erase_indices.set(operand_index);
      }
      batch_func_op.eraseArguments(erase_indices);
    } else {
      llvm::BitVector erase_indices(user_op->getNumOperands());
      for (auto operand_index : args_to_erase) {
        erase_indices.set(operand_index);
      }
      user_op->eraseOperands(erase_indices);
    }
  }

  return success();
}

// Traces usage of 'var_handle_op' or 'resources' and replaces it's usage with
// constant value 'value'. All op operands updates are captured in
// 'arguments_to_erase'.
LogicalResult ReplaceVarWithConstant(
    mlir::Value::use_range uses, ElementsAttr value,
    llvm::MapVector<Operation*, llvm::SmallVector<unsigned int, 4>>*
        arguments_to_erase) {
  llvm::SmallVector<std::pair<Region*, int>, 4> work_list;
  for (auto& usage : uses) {
    auto* user_op = usage.getOwner();
    int operand_index = usage.getOperandNumber();
    if (failed(PropagateUsage(user_op, operand_index, value, &work_list,
                              arguments_to_erase))) {
      return failure();
    }
  }

  // Container to mark visited regions to avoid infinite loop.
  llvm::DenseSet<std::pair<Region*, int>> visited;
  while (!work_list.empty()) {
    auto work_item = work_list.pop_back_val();
    if (visited.contains(work_item)) continue;
    if (failed(PropagateUsage(work_item.first, value, work_item.second,
                              &work_list, arguments_to_erase))) {
      return failure();
    }
    visited.insert(work_item);
  }

  return success();
}

}  // namespace tf_saved_model
}  // namespace mlir
