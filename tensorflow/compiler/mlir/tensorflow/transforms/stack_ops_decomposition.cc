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
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"  // TF:llvm-project
#include "mlir/IR/Module.h"  // TF:llvm-project
#include "mlir/IR/StandardTypes.h"  // TF:llvm-project
#include "mlir/IR/SymbolTable.h"  // TF:llvm-project
#include "mlir/IR/TypeUtilities.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Support/LLVM.h"  // TF:llvm-project
#include "mlir/Support/LogicalResult.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/types.h"

namespace mlir {

namespace {

// A pass that converts stack operations to tensor operations and read/assign
// ops on local variables. A later resource lifting pass can further remove the
// local variables.
//
// This pass requires that the full shape of the stack can be inferred: 1) the
// maximum size needs to be a constant and 2) a push op can be found with a
// known shape, and all push ops need to have the same shape.
//
// A stack creation op "tf.StackV2" will be turned in to two zero-initialized
// variables, for the buffer and current size. Each push will be turned into
//   %old_val = "tf.ReadVariableOp"(%buffer)
//   %old_size = "tf.ReadVariableOp"(%size)
//   %offsets = "tf.ConcatV2"(%old_size, %other_dims_0s, %const0)
//   %new_val = "tf.XlaDynamicUpdateSlice"(%old_val, %push_val, %offsets)
//   "tf.AssignVariableOp"(%buffer, %new_val)
//   %new_size = "tf.AddV2"(%old_size, %const1)
//   "tf.AssignVariableOp"(%size, %new_size)
//
// and each pop will be turned into
//
//   %old_val = "tf.ReadVariableOp"(%buffer)
//   %old_size = "tf.ReadVariableOp"(%size)
//   %new_size = "tf.Sub"(%old_size, %const1)
//   %offsets = "tf.ConcatV2"(%old_size, %other_dims_0s, %const0)
//   %slice = "tf.Slice"(%old_val, %offsets, %slice_size_const)
//   %pop_result = "tf.Reshape"(%slice, %elem_size_const)
//   "tf.AssignVariableOp"(%size, %new_size)
//
// The pass also works across control flow and functional calls.
struct StackOpsDecompositionPass
    : public ModulePass<StackOpsDecompositionPass> {
  void runOnModule() override;
};

// Creates a ReadVariableOp on a local variable.
Value ReadLocalVariable(Value local_var, OpBuilder builder, Location loc) {
  return builder
      .create<TF::ReadVariableOp>(
          loc,
          ArrayRef<Type>{getElementTypeOrSelf(local_var.getType())
                             .cast<TF::ResourceType>()
                             .getSubtypes()[0]},
          ArrayRef<Value>{local_var}, ArrayRef<NamedAttribute>{})
      .value();
}

// Creates an AssignVariableOp on a local variable.
TF::AssignVariableOp WriteLocalVariable(Value local_var, Value value,
                                        OpBuilder builder, Location loc) {
  return builder.create<TF::AssignVariableOp>(loc, ArrayRef<Type>{},
                                              ArrayRef<Value>{local_var, value},
                                              ArrayRef<NamedAttribute>{});
}

// Creates an i32 scalar tf.Const.
TF::ConstOp CreateScalarConst(int value, OpBuilder builder, Location loc) {
  tensorflow::Tensor scalar_tensor(tensorflow::DT_INT32, {});
  scalar_tensor.scalar<tensorflow::int32>()() = value;
  return builder.create<TF::ConstOp>(
      loc, tensorflow::ConvertTensor(scalar_tensor, &builder).ValueOrDie());
}

// Creates an i32 vector tf.Const.
TF::ConstOp GetR1Const(ArrayRef<int64_t> r1, OpBuilder builder, Location loc) {
  tensorflow::Tensor shape_tensor(tensorflow::DT_INT32,
                                  {static_cast<int64_t>(r1.size())});
  for (int i = 0; i < r1.size(); ++i) {
    shape_tensor.vec<tensorflow::int32>()(i) = r1[i];
  }
  return builder.create<TF::ConstOp>(
      loc, tensorflow::ConvertTensor(shape_tensor, &builder).ValueOrDie());
}

// Creates a rank-1 op that represents the offsets of the stack element in the
// stack buffer.
Value GetIndicesForStackElement(Value index, Value stack_value,
                                OpBuilder builder, Location loc) {
  auto stack_type = stack_value.getType().cast<RankedTensorType>();
  if (stack_type.getShape().size() == 1) return index;
  llvm::SmallVector<int64_t, 8> zeros(stack_type.getShape().size() - 1, 0);
  auto zeros_tensor = GetR1Const(zeros, builder, loc);
  return builder.create<TF::ConcatV2Op>(
      loc,
      ArrayRef<Type>{RankedTensorType::get(
          {static_cast<int64_t>(stack_type.getShape().size())},
          getElementTypeOrSelf(index.getType()))},
      ArrayRef<Value>{index, zeros_tensor, CreateScalarConst(0, builder, loc)},
      ArrayRef<NamedAttribute>{});
}

// Returns the type of the local variable for the stack size. It is a
// tensor<1xi32>, and we use R1 instead of a scalar because it is easier to
// concat it with other offsets.
Type GetSizeVarType(OpBuilder builder) {
  auto size_type = RankedTensorType::get({1}, builder.getIntegerType(32));
  return RankedTensorType::get(
      {}, TF::ResourceType::get(ArrayRef<TensorType>{size_type},
                                builder.getContext()));
}

// Creates the buffer and size local variables for a stack.
std::pair<Value, Value> CreateVariablesForStack(TensorType stack_tensor_type,
                                                TF::StackV2Op stack) {
  OpBuilder builder(stack);
  auto size_var_type = GetSizeVarType(builder);
  auto var_type = RankedTensorType::get(
      {}, TF::ResourceType::get(ArrayRef<TensorType>{stack_tensor_type},
                                stack.getContext()));
  auto local_var = builder.create<TF::MlirLocalVarOp>(
      stack.getLoc(), ArrayRef<Type>{var_type}, ArrayRef<Value>{},
      ArrayRef<NamedAttribute>{});
  auto local_size_var = builder.create<TF::MlirLocalVarOp>(
      stack.getLoc(), ArrayRef<Type>{size_var_type}, ArrayRef<Value>{},
      ArrayRef<NamedAttribute>{});

  // Zero-initialize the local vars.
  WriteLocalVariable(local_size_var, GetR1Const({0LL}, builder, stack.getLoc()),
                     builder, stack.getLoc());
  auto zero = CreateScalarConst(0, builder, stack.getLoc()).output();
  if (getElementTypeOrSelf(zero.getType()) !=
      stack_tensor_type.getElementType()) {
    zero = builder.create<TF::CastOp>(
        stack.getLoc(),
        ArrayRef<Type>{
            RankedTensorType::get({}, stack_tensor_type.getElementType())},
        ArrayRef<Value>{zero}, ArrayRef<NamedAttribute>{});
  }
  auto broadcast = builder.create<TF::BroadcastToOp>(
      stack.getLoc(), ArrayRef<Type>{stack_tensor_type},
      ArrayRef<Value>{zero, GetR1Const(stack_tensor_type.getShape(), builder,
                                       stack.getLoc())},
      ArrayRef<NamedAttribute>{});
  WriteLocalVariable(local_var, broadcast, builder, stack.getLoc());
  return {local_var, local_size_var};
}

// Tries to infer the stack element type with full shape based on its uses.
llvm::Optional<RankedTensorType> GetStackElementType(Value stack,
                                                     ModuleOp module) {
  for (auto& use : stack.getUses()) {
    if (auto push = llvm::dyn_cast<TF::StackPushV2Op>(use.getOwner())) {
      auto elem_type = push.elem().getType().dyn_cast<RankedTensorType>();
      if (elem_type && elem_type.hasStaticShape()) {
        return elem_type;
      }
    } else if (auto while_op = llvm::dyn_cast<TF::WhileOp>(use.getOwner())) {
      auto body = module.lookupSymbol<FuncOp>(while_op.body());
      assert(body);
      auto type_from_body =
          GetStackElementType(body.getArgument(use.getOperandNumber()), module);
      if (type_from_body.hasValue()) return type_from_body;
    } else if (auto if_op = llvm::dyn_cast<TF::IfOp>(use.getOwner())) {
      auto then_branch = module.lookupSymbol<FuncOp>(if_op.then_branch());
      auto else_branch = module.lookupSymbol<FuncOp>(if_op.else_branch());
      assert(then_branch && else_branch);
      auto type_from_then = GetStackElementType(
          then_branch.getArgument(use.getOperandNumber() - 1), module);
      if (type_from_then.hasValue()) return type_from_then;
      auto type_from_else = GetStackElementType(
          else_branch.getArgument(use.getOperandNumber() - 1), module);
      if (type_from_else.hasValue()) return type_from_else;
    } else if (auto pcall =
                   llvm::dyn_cast<TF::PartitionedCallOp>(use.getOwner())) {
      if (!pcall.f().isa<FlatSymbolRefAttr>()) continue;
      auto callee = module.lookupSymbol<FuncOp>(pcall.f().getRootReference());
      assert(callee);
      auto type_from_callee = GetStackElementType(
          callee.getArgument(use.getOperandNumber()), module);
      if (type_from_callee.hasValue()) return type_from_callee;
    } else if (auto spcall = llvm::dyn_cast<TF::StatefulPartitionedCallOp>(
                   use.getOwner())) {
      auto callee = module.lookupSymbol<FuncOp>(spcall.f());
      assert(callee);
      auto type_from_callee = GetStackElementType(
          callee.getArgument(use.getOperandNumber()), module);
      if (type_from_callee.hasValue()) return type_from_callee;
    } else if (llvm::isa<TF::IdentityOp>(use.getOwner()) ||
               llvm::isa<TF::IdentityNOp>(use.getOwner())) {
      auto type_from_alias = GetStackElementType(
          use.getOwner()->getResult(use.getOperandNumber()), module);
      if (type_from_alias.hasValue()) return type_from_alias;
    }
  }
  return llvm::None;
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
  func.setType(FunctionType::get(
      new_input_types,
      llvm::to_vector<8>(func.front().getTerminator()->getOperandTypes()),
      func.getContext()));
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
    llvm::SmallDenseMap<FuncOp, PartitionedCallStackOpsInfo>*);

// Handles stack usage by a tf.While. It will convert the body and conditional
// function signatures, and performs stack ops decomposition on them.
LogicalResult HandleWhileOp(
    TF::WhileOp while_op, ModuleOp module,
    const llvm::SmallDenseMap<Value, Value>& data_var_to_size_var,
    llvm::SmallDenseMap<FuncOp, PartitionedCallStackOpsInfo>*
        decomposed_partitioned_call_callees) {
  auto body = module.lookupSymbol<FuncOp>(while_op.body());
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
  auto cond = module.lookupSymbol<FuncOp>(while_op.cond());
  ModifyFunctionSignature(cond, nullptr, find_arg_stack_type);
  llvm::SmallDenseMap<Value, Value> empty_map;
  if (failed(DecomposeStackOpsInternal(&cond.front(), module, &empty_map,
                                       decomposed_partitioned_call_callees))) {
    return failure();
  }
  if (!signature_change) return success();
  // Create the new while op.
  auto new_while_operands = llvm::to_vector<8>(while_op.getOperands());
  auto new_output_shapes =
      llvm::to_vector<8>(while_op.output_shapes().getValue());
  OpBuilder builder(while_op);
  assert(while_op.getNumOperands() == while_op.getNumResults());
  for (int64_t i = 0; i < while_op.getNumResults(); ++i) {
    auto it = data_var_to_size_var.find(while_op.getOperand(i));
    if (it == data_var_to_size_var.end()) continue;
    new_while_operands.push_back(it->getSecond());
    if (!new_output_shapes.empty()) {
      // Size is a scalar shape.
      tensorflow::TensorShapeProto shape_proto;
      new_output_shapes.push_back(builder.getStringAttr(
          tensorflow::mangling_util::MangleShape(shape_proto)));
    }
  }
  auto new_while =
      builder.create<TF::WhileOp>(while_op.getLoc(), body.getType().getInputs(),
                                  new_while_operands, while_op.getAttrs());
  new_while.setAttr("output_shapes", builder.getArrayAttr(new_output_shapes));
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
    llvm::SmallDenseMap<FuncOp, PartitionedCallStackOpsInfo>*
        decomposed_partitioned_call_callees) {
  auto then_branch = module.lookupSymbol<FuncOp>(if_op.then_branch());
  auto else_branch = module.lookupSymbol<FuncOp>(if_op.else_branch());
  llvm::SmallDenseMap<Value, Value> then_map;
  llvm::SmallDenseMap<Value, Value> else_map;

  auto find_arg_stack_type = [&](int64_t index) -> llvm::Optional<Type> {
    auto it = data_var_to_size_var.find(if_op.getOperand(index + 1));
    if (it == data_var_to_size_var.end()) return llvm::None;
    return it->getFirst().getType();
  };
  ModifyFunctionSignature(then_branch, &then_map, find_arg_stack_type);
  ModifyFunctionSignature(else_branch, &else_map, find_arg_stack_type);
  const bool signature_change = !then_map.empty() || !else_map.empty();
  if (failed(DecomposeStackOpsInternal(&then_branch.front(), module, &then_map,
                                       decomposed_partitioned_call_callees)) ||
      failed(DecomposeStackOpsInternal(&else_branch.front(), module, &else_map,
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
      if_op.getLoc(), then_branch.getType().getResults(), new_if_operands,
      if_op.getAttrs());
  for (auto result : if_op.getResults()) {
    if (!getElementTypeOrSelf(result.getType()).isa<TF::ResourceType>()) {
      continue;
    }
    int64_t then_aliased_input =
        FindAliasedInput(then_branch, result.getResultNumber());
    int64_t else_aliased_input =
        FindAliasedInput(else_branch, result.getResultNumber());
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
    llvm::SmallDenseMap<FuncOp, PartitionedCallStackOpsInfo>*
        decomposed_partitioned_call_callees) {
  auto emplace_res = decomposed_partitioned_call_callees->try_emplace(
      callee, PartitionedCallStackOpsInfo());
  auto& info = emplace_res.first->getSecond();
  // Recreate the call op with info.
  auto recreate_caller = [&] {
    auto new_operands = llvm::to_vector<8>(call.getOperands());
    for (int64_t i = 0; i < call.getNumOperands(); ++i) {
      auto arg_it = info.stack_var_arg_to_size_arg.find(i);
      if (arg_it == info.stack_var_arg_to_size_arg.end()) continue;
      auto it = data_var_to_size_var.find(call.getOperand(i));
      if (it == data_var_to_size_var.end()) {
        call.emitOpError("Unknown stack.");
        return failure();
      }
      assert(arg_it->second == new_operands.size());
      new_operands.push_back(it->getSecond());
    }
    OpBuilder builder(call);
    auto new_call = builder.create<CallOp>(
        call.getLoc(), info.decomposed_callee.getType().getResults(),
        new_operands, call.getAttrs());
    new_call.setAttr(
        "f", builder.getSymbolRefAttr(
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
  auto callee_clone = callee.clone();
  auto find_arg_stack_type = [&](int64_t index) -> llvm::Optional<Type> {
    auto it = data_var_to_size_var.find(call.getOperand(index));
    if (it == data_var_to_size_var.end()) return llvm::None;
    return it->getFirst().getType();
  };
  ModifyFunctionSignature(callee_clone, &callee_map, find_arg_stack_type);
  if (callee_map.empty()) {
    // Signature is not modified. We do not need the clone.
    info.signature_change = false;
    callee_clone.erase();
  } else {
    info.signature_change = true;
    info.decomposed_callee = callee_clone;
    for (auto& entry : callee_map) {
      info.stack_var_arg_to_size_arg
          [entry.getFirst().cast<BlockArgument>().getArgNumber()] =
          entry.getSecond().cast<BlockArgument>().getArgNumber();
    }
    // Add the clone with a new name.
    auto name_base = llvm::join(
        std::vector<std::string>{callee.getName().str(), "stack_decomposed"},
        "_");
    auto name = name_base;
    {
      int64_t counter = 0;
      while (module.lookupSymbol(name)) {
        name = llvm::formatv("{0}_{1}", name_base, counter++).str();
      }
    }
    callee_clone.setName(name);
    SymbolTable(module).insert(callee_clone);
    callee = callee_clone;
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
  auto elem_type = GetStackElementType(stack.handle(), module);
  if (!elem_type.hasValue()) {
    return stack.emitOpError("cannot infer element shape of stack.");
  }
  auto size_op = stack.max_size().getDefiningOp();
  if (!size_op || !llvm::isa<TF::ConstOp>(size_op)) {
    return stack.emitOpError("max size of stack is not a constant.");
  }
  int64_t max_size =
      (*llvm::cast<TF::ConstOp>(size_op).value().getValues<APInt>().begin())
          .getSExtValue();
  llvm::SmallVector<int64_t, 8> stack_shape;
  stack_shape.push_back(max_size);
  for (int64_t dim : elem_type->getShape()) stack_shape.push_back(dim);
  auto stack_tensor_type =
      RankedTensorType::get(stack_shape, elem_type->getElementType());
  Value local_var;
  Value local_size_var;
  std::tie(local_var, local_size_var) =
      CreateVariablesForStack(stack_tensor_type, stack);
  stack.replaceAllUsesWith(local_var);
  (*data_var_to_size_var)[local_var] = local_size_var;
  stack.erase();
  return success();
}

LogicalResult HandleStackPushV2Op(
    TF::StackPushV2Op push,
    llvm::SmallDenseMap<Value, Value>* data_var_to_size_var) {
  auto it = data_var_to_size_var->find(push.handle());
  if (it == data_var_to_size_var->end()) {
    return push.emitOpError("unknown stack.");
  }
  // Push output simply forward the input element.
  push.replaceAllUsesWith(push.elem());
  OpBuilder builder(push);
  // Read the current buffer and size.
  auto stack_val = ReadLocalVariable(push.handle(), builder, push.getLoc());
  auto index = ReadLocalVariable(it->getSecond(), builder, push.getLoc());
  auto stack_buffer_type = stack_val.getType().cast<RankedTensorType>();
  auto slice_shape = llvm::to_vector<8>(stack_buffer_type.getShape());
  slice_shape[0] = 1;
  // Caculate the updated buffer.
  auto update_slice = builder.create<TF::ReshapeOp>(
      push.getLoc(),
      ArrayRef<Type>{RankedTensorType::get(slice_shape,
                                           stack_buffer_type.getElementType())},
      ArrayRef<Value>{push.elem(),
                      GetR1Const(slice_shape, builder, push.getLoc())},
      ArrayRef<NamedAttribute>{});
  stack_val =
      builder
          .create<TF::XlaDynamicUpdateSliceOp>(
              push.getLoc(), ArrayRef<Type>{stack_val.getType()},
              ArrayRef<Value>{stack_val, update_slice,
                              GetIndicesForStackElement(
                                  index, stack_val, builder, push.getLoc())},
              ArrayRef<NamedAttribute>{})
          .output();
  // Assign the new buffer and size.
  WriteLocalVariable(push.handle(), stack_val, builder, push.getLoc());
  index = builder.create<TF::AddV2Op>(
      push.getLoc(), ArrayRef<Type>{index.getType()},
      ArrayRef<Value>{index, GetR1Const({1}, builder, push.getLoc())},
      ArrayRef<NamedAttribute>{});
  WriteLocalVariable(it->getSecond(), index, builder, push.getLoc());
  push.erase();
  return success();
}

LogicalResult HandleStackPopV2Op(
    TF::StackPopV2Op pop,
    llvm::SmallDenseMap<Value, Value>* data_var_to_size_var) {
  auto it = data_var_to_size_var->find(pop.handle());
  if (it == data_var_to_size_var->end()) {
    return pop.emitOpError("unknown stack.");
  }
  OpBuilder builder(pop);
  // Read the current buffer and size.
  auto stack_val = ReadLocalVariable(pop.handle(), builder, pop.getLoc());
  auto size = ReadLocalVariable(it->getSecond(), builder, pop.getLoc());
  auto new_size = builder.create<TF::SubOp>(
      pop.getLoc(), ArrayRef<Type>{size.getType()},
      ArrayRef<Value>{size, GetR1Const({1}, builder, pop.getLoc())},
      ArrayRef<NamedAttribute>{});
  auto stack_val_type = stack_val.getType().cast<RankedTensorType>();
  auto elem_type = RankedTensorType::get(stack_val_type.getShape().drop_front(),
                                         stack_val_type.getElementType());
  // Slice the buffer to get the element.
  llvm::SmallVector<int64_t, 8> slice_size;
  slice_size.push_back(1);
  for (int64_t dim : elem_type.getShape()) slice_size.push_back(dim);
  auto size_const = GetR1Const(slice_size, builder, pop.getLoc());
  auto slice_type =
      RankedTensorType::get(slice_size, stack_val_type.getElementType());
  auto slice = builder.create<TF::SliceOp>(
      pop.getLoc(), ArrayRef<Type>{slice_type},
      ArrayRef<Value>{
          stack_val,
          GetIndicesForStackElement(new_size, stack_val, builder, pop.getLoc()),
          size_const},
      ArrayRef<NamedAttribute>{});
  auto pop_val = builder.create<TF::ReshapeOp>(
      pop.getLoc(), ArrayRef<Type>{elem_type},
      ArrayRef<Value>{slice,
                      GetR1Const(elem_type.getShape(), builder, pop.getLoc())},
      ArrayRef<NamedAttribute>{});
  pop.replaceAllUsesWith(pop_val.output());
  // Update the size.
  WriteLocalVariable(it->getSecond(), new_size, builder, pop.getLoc());
  pop.erase();
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
    llvm::SmallDenseMap<FuncOp, PartitionedCallStackOpsInfo>*
        decomposed_partitioned_call_callees) {
  for (auto& op : llvm::make_early_inc_range(block->getOperations())) {
    if (llvm::isa<TF::IdentityOp>(&op) || llvm::isa<TF::IdentityNOp>(&op)) {
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
    } else if (auto pcall = llvm::dyn_cast<TF::PartitionedCallOp>(&op)) {
      if (!pcall.f().isa<FlatSymbolRefAttr>()) {
        return pcall.emitOpError(
            "Stack decomposition does not support call with nested "
            "references.");
      }
      if (failed(HandlePartitionedCallOp(
              pcall, module.lookupSymbol<FuncOp>(pcall.f().getRootReference()),
              module, *data_var_to_size_var,
              decomposed_partitioned_call_callees))) {
        return failure();
      }
    } else if (auto spcall =
                   llvm::dyn_cast<TF::StatefulPartitionedCallOp>(&op)) {
      if (failed(HandlePartitionedCallOp(
              spcall, module.lookupSymbol<FuncOp>(spcall.f()), module,
              *data_var_to_size_var, decomposed_partitioned_call_callees))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult DecomposeStackOps(Block* block, ModuleOp module) {
  llvm::SmallDenseMap<Value, Value> data_var_to_size_var;
  llvm::SmallDenseMap<FuncOp, PartitionedCallStackOpsInfo>
      decomposed_partitioned_call_callees;
  return DecomposeStackOpsInternal(block, module, &data_var_to_size_var,
                                   &decomposed_partitioned_call_callees);
}

void StackOpsDecompositionPass::runOnModule() {
  auto module = getModule();
  auto main = module.lookupSymbol<FuncOp>("main");
  if (!main) return;
  if (failed(DecomposeStackOps(&main.front(), module))) {
    signalPassFailure();
  }
}

static PassRegistration<StackOpsDecompositionPass> pass(
    "tf-stack-ops-decomposition",
    "Decompose stack operations into local variable operations. Needs static "
    "shapes.");

}  // namespace

namespace TF {
std::unique_ptr<OpPassBase<ModuleOp>> CreateStackOpsDecompositionPass() {
  return std::make_unique<StackOpsDecompositionPass>();
}

}  // namespace TF
}  // namespace mlir
