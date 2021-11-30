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

#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"

namespace mlir {

namespace {

namespace cutil = TF::collection_ops_util;

struct StackOpsDecompositionPass
    : public TF::StackOpsDecompositionPassBase<StackOpsDecompositionPass> {
  void runOnOperation() final;
};

// Returns the type of the local variable for the stack size.
Type GetSizeVarType(OpBuilder builder) {
  auto size_type = cutil::GetSizeType(builder);
  return RankedTensorType::get(
      {}, TF::ResourceType::get(ArrayRef<TensorType>{size_type},
                                builder.getContext()));
}

// Returns the aliasing argument number of a fucntion return value if it simply
// forwards the argument. Otherwise, returns -1.
int64_t FindAliasedInput(FuncOp func, int64_t return_index) {
  Value return_val = func.front().getTerminator()->getOperand(return_index);
  auto maybe_arg = return_val.dyn_cast<BlockArgument>();
  if (!maybe_arg) return -1;
  return maybe_arg.getArgNumber();
}

// Changes the function signature that has stacks in the arguments. A stack
// argument will be turned into a variable type if arg_to_stack_type returns
// such a type, and a new argument will be added to the end of the argument
// list for the size variable.
//
// If stack_var_to_size_var is not nullptr, it will  be used to store the
// mapping from the stack-variable argument to the size-variable argument.
//
// If handle_new_size_vars is provided, it will be invoked on the list of new
// size variables before finally changing the function type.
void ModifyFunctionSignature(
    FuncOp func, llvm::SmallDenseMap<Value, Value>* stack_var_to_size_var,
    llvm::function_ref<llvm::Optional<Type>(int64_t)> arg_to_stack_type,
    llvm::function_ref<void(ArrayRef<BlockArgument>)> handle_new_size_vars =
        nullptr) {
  auto new_input_types = llvm::to_vector<8>(func.getType().getInputs());
  auto size_var_type = GetSizeVarType(OpBuilder(func));
  int64_t original_arg_count = new_input_types.size();
  for (int64_t i = 0; i < original_arg_count; ++i) {
    auto stack_type = arg_to_stack_type(i);
    if (!stack_type.hasValue()) continue;
    func.getArgument(i).setType(*stack_type);
    new_input_types[i] = *stack_type;
    auto size_arg = func.front().addArgument(size_var_type);
    new_input_types.push_back(size_arg.getType());
    if (stack_var_to_size_var) {
      (*stack_var_to_size_var)[func.getArgument(i)] = size_arg;
    }
  }
  if (handle_new_size_vars) {
    handle_new_size_vars(func.getArguments().drop_front(original_arg_count));
  }
  func.setType(
      FunctionType::get(func.getContext(), new_input_types,
                        func.front().getTerminator()->getOperandTypes()));
}

// Contains cached information for decomposed callee functions for (stateful)
// partitioned call ops.
struct PartitionedCallStackOpsInfo {
  bool signature_change;
  FuncOp decomposed_callee;
  llvm::SmallDenseMap<int64_t, int64_t> stack_var_arg_to_size_arg;
};

LogicalResult DecomposeStackOpsInternal(
    Block*, ModuleOp, llvm::SmallDenseMap<Value, Value>*,
    llvm::StringMap<PartitionedCallStackOpsInfo>*);

// Handles stack usage by a tf.While. It will convert the body and conditional
// function signatures, and performs stack ops decomposition on them.
LogicalResult HandleWhileOp(
    TF::WhileOp while_op, ModuleOp module,
    const llvm::SmallDenseMap<Value, Value>& data_var_to_size_var,
    llvm::StringMap<PartitionedCallStackOpsInfo>*
        decomposed_partitioned_call_callees) {
  auto body = while_op.body_function();
  llvm::SmallDenseMap<Value, Value> body_map;
  auto find_arg_stack_type = [&](int64_t index) -> llvm::Optional<Type> {
    auto it = data_var_to_size_var.find(while_op.getOperand(index));
    if (it == data_var_to_size_var.end()) return llvm::None;
    return it->getFirst().getType();
  };
  auto add_size_vars_to_return = [&](ArrayRef<BlockArgument> new_args) {
    if (new_args.empty()) return;
    auto body_ret = body.front().getTerminator();
    auto new_body_returns = llvm::to_vector<8>(body_ret->getOperands());
    for (auto arg : new_args) new_body_returns.push_back(arg);
    OpBuilder(body_ret).create<ReturnOp>(body_ret->getLoc(), new_body_returns);
    body_ret->erase();
  };
  // Handle body.
  ModifyFunctionSignature(body, &body_map, find_arg_stack_type,
                          add_size_vars_to_return);
  const bool signature_change = !body_map.empty();
  if (failed(DecomposeStackOpsInternal(&body.front(), module, &body_map,
                                       decomposed_partitioned_call_callees))) {
    return failure();
  }
  // Cond should not change stacks in the arguments, so use an empty map.
  auto cond = while_op.cond_function();
  ModifyFunctionSignature(cond, nullptr, find_arg_stack_type);
  llvm::SmallDenseMap<Value, Value> empty_map;
  if (failed(DecomposeStackOpsInternal(&cond.front(), module, &empty_map,
                                       decomposed_partitioned_call_callees))) {
    return failure();
  }
  if (!signature_change) return success();
  // Create the new while op.
  auto new_while_operands = llvm::to_vector<8>(while_op.getOperands());
  OpBuilder builder(while_op);
  assert(while_op.getNumOperands() == while_op.getNumResults());
  for (int64_t i = 0; i < while_op.getNumResults(); ++i) {
    auto it = data_var_to_size_var.find(while_op.getOperand(i));
    if (it == data_var_to_size_var.end()) continue;
    new_while_operands.push_back(it->getSecond());
  }
  auto new_while =
      builder.create<TF::WhileOp>(while_op.getLoc(), body.getType().getInputs(),
                                  new_while_operands, while_op->getAttrs());
  for (int64_t i = 0; i < while_op.getNumResults(); ++i) {
    if (!getElementTypeOrSelf(while_op.getOperand(i).getType())
             .isa<TF::ResourceType>()) {
      continue;
    }
    int64_t aliased_input = FindAliasedInput(body, i);
    if (aliased_input == i) {
      // Replace aliased stack output uses with input.
      while_op.getResult(i).replaceAllUsesWith(while_op.getOperand(i));
    }
  }
  while_op.replaceAllUsesWith(
      new_while.getResults().take_front(while_op.getNumResults()));
  while_op.erase();
  return success();
}

// Handles stack usage by a tf.If. It will convert the branch function
// signatures, and performs stack ops decomposition on them.
LogicalResult HandleIfOp(
    TF::IfOp if_op, ModuleOp module,
    const llvm::SmallDenseMap<Value, Value>& data_var_to_size_var,
    llvm::StringMap<PartitionedCallStackOpsInfo>*
        decomposed_partitioned_call_callees) {
  auto then_func = if_op.then_function();
  auto else_func = if_op.else_function();
  llvm::SmallDenseMap<Value, Value> then_map;
  llvm::SmallDenseMap<Value, Value> else_map;

  auto find_arg_stack_type = [&](int64_t index) -> llvm::Optional<Type> {
    auto it = data_var_to_size_var.find(if_op.getOperand(index + 1));
    if (it == data_var_to_size_var.end()) return llvm::None;
    return it->getFirst().getType();
  };
  ModifyFunctionSignature(then_func, &then_map, find_arg_stack_type);
  ModifyFunctionSignature(else_func, &else_map, find_arg_stack_type);
  const bool signature_change = !then_map.empty() || !else_map.empty();
  if (failed(DecomposeStackOpsInternal(&then_func.front(), module, &then_map,
                                       decomposed_partitioned_call_callees)) ||
      failed(DecomposeStackOpsInternal(&else_func.front(), module, &else_map,
                                       decomposed_partitioned_call_callees))) {
    return failure();
  }
  if (!signature_change) return success();
  auto new_if_operands = llvm::to_vector<8>(if_op.getOperands());
  for (auto operand : if_op.getOperands()) {
    auto it = data_var_to_size_var.find(operand);
    if (it == data_var_to_size_var.end()) continue;
    new_if_operands.push_back(it->getSecond());
  }
  auto new_if = OpBuilder(if_op).create<TF::IfOp>(
      if_op.getLoc(), then_func.getType().getResults(), new_if_operands,
      if_op->getAttrs());
  for (auto result : if_op.getResults()) {
    if (!getElementTypeOrSelf(result.getType()).isa<TF::ResourceType>()) {
      continue;
    }
    int64_t then_aliased_input =
        FindAliasedInput(then_func, result.getResultNumber());
    int64_t else_aliased_input =
        FindAliasedInput(else_func, result.getResultNumber());
    if (then_aliased_input >= 0 && then_aliased_input == else_aliased_input) {
      // Replace aliased stack output uses with input.
      result.replaceAllUsesWith(if_op.getOperand(then_aliased_input + 1));
    }
  }
  if_op.replaceAllUsesWith(new_if);
  if_op.erase();
  return success();
}

// Handles stack usage by a tf.StatefulPartitionedCall or a tf.PartitionedCall.
// It will first check if the callee was previously handled, and try to reuse
// that result if so. Otherwise, it will clone and convert the callee function,
// and performs stack ops decomposition on it.
template <typename CallOp>
LogicalResult HandlePartitionedCallOp(
    CallOp call, FuncOp callee, ModuleOp module,
    const llvm::SmallDenseMap<Value, Value>& data_var_to_size_var,
    llvm::StringMap<PartitionedCallStackOpsInfo>*
        decomposed_partitioned_call_callees) {
  auto emplace_res = decomposed_partitioned_call_callees->try_emplace(
      callee.getName(), PartitionedCallStackOpsInfo());
  auto& info = emplace_res.first->second;
  // Recreate the call op with info.
  auto recreate_caller = [&] {
    auto new_operands = llvm::to_vector<8>(call.getOperands());
    for (int64_t i = 0; i < call.getNumOperands(); ++i) {
      auto arg_it = info.stack_var_arg_to_size_arg.find(i);
      if (arg_it == info.stack_var_arg_to_size_arg.end()) continue;
      auto it = data_var_to_size_var.find(call.getOperand(i));
      if (it == data_var_to_size_var.end()) {
        call.emitOpError("unknown stack");
        return failure();
      }
      assert(arg_it->second == new_operands.size());
      new_operands.push_back(it->getSecond());
    }
    OpBuilder builder(call);
    auto new_call = builder.create<CallOp>(
        call.getLoc(), info.decomposed_callee.getType().getResults(),
        new_operands, call->getAttrs());
    new_call->setAttr(
        "f", SymbolRefAttr::get(
                 builder.getContext(),
                 const_cast<FuncOp&>(info.decomposed_callee).getName()));
    for (int64_t i = 0; i < call.getNumResults(); ++i) {
      auto result = call.getResult(i);
      if (!getElementTypeOrSelf(result.getType())
               .template isa<TF::ResourceType>()) {
        continue;
      }
      int64_t aliased_input = FindAliasedInput(info.decomposed_callee, i);
      if (aliased_input >= 0) {
        // Replace aliased stack output uses with input.
        result.replaceAllUsesWith(call.getOperand(aliased_input));
      }
    }
    call.replaceAllUsesWith(new_call);
    call.erase();
    return success();
  };
  if (!emplace_res.second) {
    // This callee was handled before.
    if (!info.signature_change) return success();
    return recreate_caller();
  }
  llvm::SmallDenseMap<Value, Value> callee_map;
  FuncOp lowered_callee = callee;
  if (!callee.isPrivate()) {
    // Clone non-private callee in case of signature change.
    lowered_callee = callee.clone();
    lowered_callee.setPrivate();
  }
  auto find_arg_stack_type = [&](int64_t index) -> llvm::Optional<Type> {
    auto it = data_var_to_size_var.find(call.getOperand(index));
    if (it == data_var_to_size_var.end()) return llvm::None;
    return it->getFirst().getType();
  };
  ModifyFunctionSignature(lowered_callee, &callee_map, find_arg_stack_type);
  info.signature_change = !callee_map.empty();
  if (!info.signature_change) {
    // Signature is not modified. We do not need the clone.
    if (lowered_callee != callee) {
      lowered_callee.erase();
    }
  } else {
    info.decomposed_callee = lowered_callee;
    for (auto& entry : callee_map) {
      info.stack_var_arg_to_size_arg
          [entry.getFirst().cast<BlockArgument>().getArgNumber()] =
          entry.getSecond().cast<BlockArgument>().getArgNumber();
    }
    if (lowered_callee != callee) {
      // Add the clone with a new name.
      lowered_callee.setName(StringAttr::get(
          callee->getContext(),
          llvm::formatv("{0}_stack_decomposed", callee.getName()).str()));
      SymbolTable(module).insert(lowered_callee);
      callee = lowered_callee;
    }
  }
  if (failed(DecomposeStackOpsInternal(&callee.front(), module, &callee_map,
                                       decomposed_partitioned_call_callees))) {
    return failure();
  }
  if (info.signature_change) return recreate_caller();
  return success();
}

LogicalResult HandleStackV2Op(
    TF::StackV2Op stack, ModuleOp module,
    llvm::SmallDenseMap<Value, Value>* data_var_to_size_var) {
  // Create a buffer variable and a size variable to replace the stack.
  auto elem_type = cutil::GetElementTypeFromAccess(
      stack.handle(), module, [](Operation* user) -> llvm::Optional<Type> {
        auto push = llvm::dyn_cast<TF::StackPushV2Op>(user);
        if (!push) return llvm::None;
        return push.elem().getType();
      });
  if (!elem_type.hasValue()) {
    return stack.emitOpError("cannot infer element shape of stack");
  }
  OpBuilder builder(stack);
  Value buffer;
  if (failed(cutil::CreateInitBufferValue(
          elem_type->getShape(), stack.max_size(), stack,
          elem_type->getElementType(), builder, &buffer))) {
    return failure();
  }
  auto size_var_type = GetSizeVarType(builder);
  auto var_type = RankedTensorType::get(
      {}, TF::ResourceType::get(
              ArrayRef<TensorType>{buffer.getType().cast<TensorType>()},
              stack.getContext()));
  auto local_var = builder.create<TF::MlirLocalVarOp>(
      stack.getLoc(), ArrayRef<Type>{var_type}, ArrayRef<Value>{});
  auto local_size_var = builder.create<TF::MlirLocalVarOp>(
      stack.getLoc(), ArrayRef<Type>{size_var_type}, ArrayRef<Value>{});
  // Zero-initialize the local vars.
  cutil::WriteLocalVariable(local_size_var,
                            cutil::GetR1Const({0LL}, builder, stack.getLoc()),
                            builder, stack.getLoc());
  cutil::WriteLocalVariable(local_var, buffer, builder, stack.getLoc());
  stack.handle().replaceAllUsesWith(local_var);
  (*data_var_to_size_var)[local_var] = local_size_var;
  stack.erase();
  return success();
}

LogicalResult HandleStackPushV2Op(
    TF::StackPushV2Op push,
    llvm::SmallDenseMap<Value, Value>* data_var_to_size_var) {
  auto it = data_var_to_size_var->find(push.handle());
  if (it == data_var_to_size_var->end()) {
    return push.emitOpError("unknown stack");
  }
  // Push output simply forward the input element.
  push.replaceAllUsesWith(push.elem());
  OpBuilder builder(push);
  // Read the current buffer and size.
  auto stack_val =
      cutil::ReadLocalVariable(push.handle(), builder, push.getLoc());
  auto index =
      cutil::ReadLocalVariable(it->getSecond(), builder, push.getLoc());
  stack_val =
      cutil::SetElement(index, stack_val, push.elem(), builder, push.getLoc());
  // Assign the new buffer and size.
  cutil::WriteLocalVariable(push.handle(), stack_val, builder, push.getLoc());
  index = builder.create<TF::AddV2Op>(
      push.getLoc(), ArrayRef<Type>{index.getType()},
      ArrayRef<Value>{index, cutil::GetR1Const({1}, builder, push.getLoc())});
  cutil::WriteLocalVariable(it->getSecond(), index, builder, push.getLoc());
  push.erase();
  return success();
}

LogicalResult HandleStackPopV2Op(
    TF::StackPopV2Op pop,
    llvm::SmallDenseMap<Value, Value>* data_var_to_size_var) {
  auto it = data_var_to_size_var->find(pop.handle());
  if (it == data_var_to_size_var->end()) {
    return pop.emitOpError("unknown stack");
  }
  OpBuilder builder(pop);
  // Read the current buffer and size.
  auto stack_val =
      cutil::ReadLocalVariable(pop.handle(), builder, pop.getLoc());
  auto size = cutil::ReadLocalVariable(it->getSecond(), builder, pop.getLoc());
  auto new_size = builder.create<TF::SubOp>(
      pop.getLoc(), ArrayRef<Type>{size.getType()},
      ArrayRef<Value>{size, cutil::GetR1Const({1}, builder, pop.getLoc())});
  auto pop_val = cutil::GetElement(new_size, stack_val, builder, pop.getLoc());
  pop.replaceAllUsesWith(pop_val);
  // Update the size.
  cutil::WriteLocalVariable(it->getSecond(), new_size, builder, pop.getLoc());
  pop.erase();
  return success();
}

LogicalResult HandleRegionControlFlowOps(
    Operation& op, ModuleOp module,
    llvm::SmallDenseMap<Value, Value>* data_var_to_size_var,
    llvm::StringMap<PartitionedCallStackOpsInfo>*
        decomposed_partitioned_call_callees) {
  for (OpOperand& operand : op.getOpOperands()) {
    if (getElementTypeOrSelf(operand.get().getType()).isa<TF::ResourceType>()) {
      return op.emitOpError()
             << "found unexpected type " << operand.get().getType()
             << " of operand #" << operand.getOperandNumber()
             << ", resource type operands are expected to have been "
                "canonicalized away for region based control flow ops";
    }
  }
  for (OpResult result : op.getResults()) {
    if (getElementTypeOrSelf(result.getType()).isa<TF::ResourceType>()) {
      return op.emitOpError()
             << "found unexpected type " << result.getType() << " of result #"
             << result.getResultNumber()
             << ", resource type results are expected to have been "
                "canonicalized away for region based control flow ops";
    }
  }
  for (Region& region : op.getRegions()) {
    if (failed(DecomposeStackOpsInternal(&region.front(), module,
                                         data_var_to_size_var,
                                         decomposed_partitioned_call_callees)))
      return failure();
  }
  return success();
}

// Decomposes stack ops on a region and recursively decomposes called functions.
// data_var_to_size_var: a mapping from stacks' buffer local variables to size
// local variables.
// decomposed_partitioned_call_callees: cache for partitioned call ops' callee
// function handling.
LogicalResult DecomposeStackOpsInternal(
    Block* block, ModuleOp module,
    llvm::SmallDenseMap<Value, Value>* data_var_to_size_var,
    llvm::StringMap<PartitionedCallStackOpsInfo>*
        decomposed_partitioned_call_callees) {
  for (auto& op : llvm::make_early_inc_range(block->getOperations())) {
    if (llvm::isa<TF::IdentityOp, TF::IdentityNOp>(&op)) {
      // Removes identity nodes in the block. The device computation does not
      // need such nodes to carry information.
      op.replaceAllUsesWith(op.getOperands());
      op.erase();
    } else if (auto stack = llvm::dyn_cast<TF::StackV2Op>(&op)) {
      if (failed(HandleStackV2Op(stack, module, data_var_to_size_var))) {
        return failure();
      }
    } else if (auto push = llvm::dyn_cast<TF::StackPushV2Op>(&op)) {
      if (failed(HandleStackPushV2Op(push, data_var_to_size_var))) {
        return failure();
      }
    } else if (auto pop = llvm::dyn_cast<TF::StackPopV2Op>(&op)) {
      if (failed(HandleStackPopV2Op(pop, data_var_to_size_var))) {
        return failure();
      }
    } else if (auto close = llvm::dyn_cast<TF::StackCloseV2Op>(&op)) {
      data_var_to_size_var->erase(close.handle());
      close.erase();
    } else if (auto while_op = llvm::dyn_cast<TF::WhileOp>(&op)) {
      if (failed(HandleWhileOp(while_op, module, *data_var_to_size_var,
                               decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto if_op = llvm::dyn_cast<TF::IfOp>(&op)) {
      if (failed(HandleIfOp(if_op, module, *data_var_to_size_var,
                            decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (llvm::isa<TF::WhileRegionOp>(op) ||
               llvm::isa<TF::IfRegionOp>(op) ||
               llvm::isa<TF::CaseRegionOp>(op)) {
      if (failed(
              HandleRegionControlFlowOps(op, module, data_var_to_size_var,
                                         decomposed_partitioned_call_callees)))
        return failure();
    } else if (auto pcall = llvm::dyn_cast<TF::PartitionedCallOp>(&op)) {
      if (!pcall.func()) {
        return pcall.emitOpError(
            "stack decomposition does not support call with nested references");
      }
      if (failed(HandlePartitionedCallOp(
              pcall, pcall.func(), module, *data_var_to_size_var,
              decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto spcall =
                   llvm::dyn_cast<TF::StatefulPartitionedCallOp>(&op)) {
      if (failed(HandlePartitionedCallOp(
              spcall, spcall.func(), module, *data_var_to_size_var,
              decomposed_partitioned_call_callees))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult DecomposeStackOps(Block* block, ModuleOp module) {
  llvm::SmallDenseMap<Value, Value> data_var_to_size_var;
  llvm::StringMap<PartitionedCallStackOpsInfo>
      decomposed_partitioned_call_callees;
  return DecomposeStackOpsInternal(block, module, &data_var_to_size_var,
                                   &decomposed_partitioned_call_callees);
}

void StackOpsDecompositionPass::runOnOperation() {
  auto module = getOperation();
  auto main = module.lookupSymbol<FuncOp>("main");
  if (!main) return;
  if (failed(DecomposeStackOps(&main.front(), module))) {
    signalPassFailure();
  }
}

}  // namespace

namespace TF {
std::unique_ptr<OperationPass<ModuleOp>> CreateStackOpsDecompositionPass() {
  return std::make_unique<StackOpsDecompositionPass>();
}

}  // namespace TF
}  // namespace mlir
