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
#include "tensorflow/compiler/mlir/quantization/tensorflow/utils/lift_as_function_call_utils.h"

#include <queue>
#include <stack>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {

// Checks if the op is inside a lifted function.
bool IsInLiftedFunc(Operation *op) {
  return op->getParentOfType<func::FuncOp>()->hasAttr(kFusedFunctionAttr);
}

// Inserts the function to the symbol table of the module thread-safely.
StringAttr InsertToSymbolTable(Operation *module, Operation *function,
                               const std::string &func_name) {
  static tensorflow::mutex *mtx = new tensorflow::mutex();
  tensorflow::mutex_lock lock(*mtx);

  SymbolTable symbol_table(module);
  std::string unique_name = func_name;
  int32_t uniquing_counter = 0;
  while (symbol_table.lookup(unique_name) != nullptr) {
    ++uniquing_counter;
    unique_name = func_name + "_" + std::to_string(uniquing_counter);
  }
  function->setAttr("sym_name",
                    StringAttr::get(module->getContext(), unique_name));
  return symbol_table.insert(function);
}

ValueRange createFusedFnCall(OpBuilder builder, Location location,
                             StringRef func_name, TypeRange output_types,
                             ValueRange args) {
  TF::PartitionedCallOp call_op = builder.create<TF::PartitionedCallOp>(
      location, output_types, args,
      FlatSymbolRefAttr::get(builder.getStringAttr(func_name)),
      /*config=*/"", /*config_proto=*/"", /*executor_type=*/"");
  call_op->setAttr(
      kQuantTraitAttrName,
      builder.getStringAttr(llvm::StringRef(
          std::string(QuantTraitValues[QuantizationTrait::FullyQuantizable]))));

  return call_op.getOutput();
}

// Finds ops in the paths from arguments to results. The ops is listed in an
// order that the former ops shouldn't have any dependencies on the later ones.
llvm::SmallVector<Operation *> FindOpsFromArgumentsToResults(
    const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results) {
  std::queue<Value> value_queue;
  for (Value result : results) {
    value_queue.push(result);
  }
  absl::flat_hash_set<mlir::detail::ValueImpl *> argument_set;
  for (Value argument : arguments) {
    argument_set.insert(argument.getImpl());
  }

  // Searching for ops from results to arguments. Duplicate ops in the op stack
  // are intentional in order to make sure the op on the top of the stack
  // doesn't depends on any ops below it.
  std::stack<Operation *> op_stack;
  while (!value_queue.empty()) {
    Value current_value = value_queue.front();
    value_queue.pop();

    Operation *defining_node = current_value.getDefiningOp();
    if (defining_node == nullptr) continue;
    op_stack.push(defining_node);
    for (const auto &arg : defining_node->getOperands()) {
      if (!argument_set.contains(arg.getImpl())) {
        value_queue.push(arg);
      }
    }
  }

  // Remove duplicate ops from the op stack.
  llvm::SmallVector<Operation *> sorted_ops;
  absl::flat_hash_set<Operation *> unique_ops;
  while (!op_stack.empty()) {
    Operation *current_op = op_stack.top();
    op_stack.pop();
    if (unique_ops.contains(current_op)) continue;
    sorted_ops.push_back(current_op);
    unique_ops.insert(current_op);
  }
  return sorted_ops;
}

// Finds the name of each attribute in `attributes` and set the attr_map
// attribute which maps an attribute identifier to its attribute name. The
// identifier is the order of that attribute in `attributes`. This map
// is then used to set attributes in the quantized functions in the
// QuantizeCompositeFunctionsPass.
// For example, for tf.MatMul with `attributes` = {{"transpose_a", false},
// {"transpose_b", false}}, the generated attr_map is
// "0:transpose_a,1:transpose_b", where 0 and 1 are the respective attribute
// identifiers.
// This function returns success if all attributes could be found.
LogicalResult SetAttributeMap(
    MLIRContext *context, const llvm::SmallVector<NamedAttribute> &attributes,
    const llvm::SmallVector<Operation *> &ops) {
  // A map to find which operation an attribute belongs to.
  // The key for this map uses the entire NamedAttribute object, i.e. the
  // {attribute_name, attribute_value} pair.
  llvm::SmallDenseMap<NamedAttribute, Operation *> attr_to_op_map;
  for (Operation *op : ops) {
    for (const auto &named_attr : op->getAttrs()) {
      attr_to_op_map.insert({named_attr, op});
    }
  }

  for (int idx : llvm::seq<int>(0, attributes.size())) {
    const NamedAttribute &attribute = attributes[idx];

    // Skip the following steps if the attribute value is `NullAttribute`.
    if (const auto string_attr =
            attribute.getValue().dyn_cast_or_null<StringAttr>();
        string_attr != nullptr &&
        string_attr.getValue().equals(kNullAttributeValue)) {
      continue;
    }

    if (attr_to_op_map.count(attribute) == 0) {
      mlir::emitError(UnknownLoc::get(context),
                      "Could not find attribute: " + attribute.getName().str());
      return failure();
    }

    Operation *owner_op = attr_to_op_map[attribute];

    std::string new_attr_map_str{};
    if (owner_op->hasAttr(kAttrMapAttribute)) {
      new_attr_map_str =
          owner_op->getAttrOfType<StringAttr>(kAttrMapAttribute).str();
      absl::StrAppend(&new_attr_map_str, ",");
    }

    // Append "<identifier>:<attribute_name>". Ex) "0:transpose_a".
    const std::string identifier = std::to_string(idx);
    const mlir::StringAttr attribute_name = attribute.getName();
    absl::StrAppend(&new_attr_map_str, identifier, ":", attribute_name.str());
    owner_op->setAttr(kAttrMapAttribute,
                      StringAttr::get(context, new_attr_map_str));
  }
  return success();
}

// Creates a function to wrap the section between arguments and results.
llvm::SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder builder, Location location, StringRef func_name,
    const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results,
    const llvm::SmallVector<NamedAttribute> &attributes) {
  MLIRContext *context = builder.getContext();
  if (results.empty()) {
    mlir::emitError(UnknownLoc::get(context), "No result values specified");
    return {};
  }
  Operation *result_op = results[0].getDefiningOp();
  auto module = result_op->getParentOfType<ModuleOp>();

  // Create a private function and copy all ops between arguments and results.
  auto current_func = result_op->getParentOfType<func::FuncOp>();
  auto guard = OpBuilder::InsertionGuard(builder);
  builder.setInsertionPointAfter(current_func);
  TypeRange arg_types{ValueRange{arguments}};
  TypeRange result_types{ValueRange{results}};
  auto func_type = FunctionType::get(context, arg_types, result_types);

  llvm::SmallVector<Location> arg_locs;
  for (const auto &arg : arguments) {
    arg_locs.push_back(arg.getLoc());
  }
  auto wrap_func = builder.create<func::FuncOp>(location, func_name, func_type);
  wrap_func.setVisibility(SymbolTable::Visibility::Private);
  wrap_func->setAttr(kFusedFunctionAttr, builder.getUnitAttr());
  builder.createBlock(&wrap_func.getBody(), wrap_func.begin(), arg_types,
                      arg_locs);

  IRMapping mapping;
  for (int32_t i : llvm::seq<int32_t>(0, arguments.size())) {
    mapping.map(arguments[i], wrap_func.getArgument(i));
  }

  auto cloning_ops = FindOpsFromArgumentsToResults(arguments, results);
  if (failed(SetAttributeMap(context, attributes, cloning_ops))) {
    current_func.emitError() << "Some attributes couldn't be found.";
  }
  for (Operation *op : cloning_ops) {
    builder.clone(*op, mapping);
  }

  llvm::SmallVector<Value> return_values;
  for (Value result : results) {
    return_values.push_back(mapping.lookupOrNull(result));
  }
  builder.create<mlir::func::ReturnOp>(location, return_values);

  // Create a function call to the newly created function.
  StringAttr new_func_name =
      InsertToSymbolTable(module, wrap_func, func_name.str());
  builder.setInsertionPointAfter(result_op);
  ValueRange new_results = createFusedFnCall(
      builder, location, new_func_name.getValue(), result_types, arguments);
  return llvm::SmallVector<Value, 4>(new_results.begin(), new_results.end());
}

llvm::SmallVector<Value, 4> LiftAsFunctionCall(
    OpBuilder builder, Location location, StringRef func_name,
    const llvm::SmallVector<Value> &arguments,
    const llvm::SmallVector<Value> &results) {
  llvm::SmallVector<NamedAttribute> attributes;
  return LiftAsFunctionCall(builder, location, func_name, arguments, results,
                            attributes);
}

llvm::SmallVector<Value> AppendToVector(
    const llvm::SmallVector<Value> &arguments, Value append) {
  llvm::SmallVector<Value> ret(arguments);
  ret.push_back(append);
  return ret;
}

}  // namespace quant
}  // namespace mlir
