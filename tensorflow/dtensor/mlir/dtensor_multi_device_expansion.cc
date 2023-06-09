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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORMULTIDEVICEEXPANSION
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";
constexpr char kEntryFuncAttr[] = "tf.entry_function";
constexpr char kMainFuncName[] = "main";
constexpr int kDeviceIDArgumentNumber = 0;

// This is a map from argument numbers and meshes to per-device values.
// Most arguments will only be expanded on one mesh (the one given by its
// "tf._layout" attribute); however, the device id may be expanded across
// multiple meshes. For example, when main functions have both cpu and tpu
// mesh partitioned calls.
using ExpandedArgumentMap =
    absl::flat_hash_map<int,
                        absl::flat_hash_map<Mesh, std::vector<mlir::Value>>>;

using ExpandedResultsMap = absl::flat_hash_map<int, std::vector<mlir::Value>>;

mlir::BlockArgument InsertArgumentForDevice(mlir::OpBuilder& builder,
                                            mlir::func::FuncOp func,
                                            mlir::Type arg_type,
                                            const std::string& device) {
  const int arg_index = func.getNumArguments();

  std::vector<mlir::NamedAttribute> named_attrs = {builder.getNamedAttr(
      kFuncDeviceAttr, builder.getStringAttr(llvm::StringRef(device)))};

  llvm::ArrayRef<mlir::NamedAttribute> named_array_ref(named_attrs);
  mlir::DictionaryAttr dic_attr = builder.getDictionaryAttr(named_array_ref);
  func.insertArgument(arg_index, arg_type, dic_attr, func.getLoc());

  return func.getArgument(arg_index);
}

// Returns the user of all the ops in the span iff it is a single return op.
// Otherwise, returns nullptr; for example, if there are multiple return ops.
template <typename Operation>
mlir::func::ReturnOp GetReturnOpFromUsers(absl::Span<Operation> ops) {
  mlir::func::ReturnOp return_op;

  for (Operation op : ops) {
    for (mlir::Operation* user : op->getUsers()) {
      // TODO(twelve): Determine whether we should follow identity ops.
      if (mlir::func::ReturnOp op =
              llvm::dyn_cast_or_null<mlir::func::ReturnOp>(user)) {
        if (return_op) {
          if (return_op != op) {
            return nullptr;
          }
        } else {
          return_op = op;
        }
      } else {
        return nullptr;
      }
    }
  }

  return return_op;
}

// Returns the devices for a given mesh.
absl::Span<const std::string> GetDevices(const Mesh& mesh) {
  const std::vector<std::string>& devices = mesh.global_devices();
  if (devices.empty()) {
    return mesh.local_devices();
  } else {
    return devices;
  }
}

StatusOr<absl::Span<mlir::Value>> GetExpandedArguments(
    mlir::OpBuilder& builder, mlir::func::FuncOp target_func,
    ExpandedArgumentMap& expanded_arguments, mlir::BlockArgument argument,
    const Mesh* target_mesh = nullptr);

// Extracts the operation's layouts, then expands it across them.
template <typename Operation>
mlir::LogicalResult ExpandOperation(mlir::func::FuncOp target_func,
                                    mlir::func::ReturnOp return_op,
                                    ExpandedArgumentMap& expanded_arguments,
                                    ExpandedResultsMap& expanded_results,
                                    Operation op) {
  const StatusOr<std::optional<Mesh>> mesh = ExtractDeviceMeshFromOp(op);
  if (!(mesh.ok() && *mesh)) {
    op->emitOpError("Failed to retrieve op mesh or layout.");
    return mlir::failure();
  } else if ((*mesh)->IsSingleDevice()) {
    op->emitOpError("Unimplemented, single-device expansion support.");
    return mlir::failure();
  }

  mlir::OpBuilder builder(target_func.getBody());
  const absl::Span<const std::string> devices = GetDevices(**mesh);
  const std::size_t num_devices = devices.size();
  const Mesh* target_mesh = &(**mesh);

  llvm::SmallVector<Operation> replications;
  for (std::size_t i = 0; i < num_devices; ++i) {
    llvm::SmallVector<mlir::Value, 8> operands;
    for (const mlir::Value& operand : op->getOperands()) {
      if (const auto arg = operand.dyn_cast_or_null<mlir::BlockArgument>()) {
        const StatusOr<absl::Span<mlir::Value>> new_args = GetExpandedArguments(
            builder, target_func, expanded_arguments, arg, target_mesh);
        if (!new_args.ok()) {
          op->emitOpError(tsl::NullTerminatedMessage(new_args.status()));
          return mlir::failure();
        } else if (new_args->empty()) {
          operands.push_back(operand);
        } else {
          operands.push_back((*new_args)[i]);
        }
      } else {
        operands.push_back(operand);
      }
    }

    auto new_op = builder.create<Operation>(
        op->getLoc(), op->getResultTypes(), operands, op.getFAttr(),
        /*config=*/builder.getStringAttr(""),
        /*config_proto=*/builder.getStringAttr(""),
        /*executor_type=*/builder.getStringAttr(""));

    new_op->setAttr(kDeviceAttr, builder.getStringAttr(devices[i]));

    replications.emplace_back(new_op);
  }

  if (return_op) {
    mlir::Operation::operand_range operands = return_op->getOperands();
    for (const auto [i, operand] : llvm::enumerate(operands)) {
      if (op == operand.getDefiningOp()) {
        const mlir::Operation::result_range results = op->getResults();
        const mlir::Operation::result_range::iterator search =
            llvm::find(results, operand);
        const std::size_t result_number = search - results.begin();
        for (const Operation& replication : replications) {
          expanded_results[i].emplace_back(
              replication->getResult(result_number));
        }
      }
    }
  }

  return mlir::success();
}

// Generates a `NamedAttr` whose value is a comma-separated list of
// elements, given by applying the format function to each integer in
// the range 0..len(types).
template <typename Format>
mlir::NamedAttribute GetNamesAttr(mlir::OpBuilder& builder, const char* name,
                                  llvm::ArrayRef<mlir::Type> types,
                                  const Format& fmt) {
  llvm::SmallVector<string, 8> names;
  for (int i = 0; i < types.size(); ++i) {
    names.push_back(fmt(i));
  }
  return builder.getNamedAttr(name,
                              builder.getStringAttr(absl::StrJoin(names, ",")));
}

// Updates the entry function's attributes to reflect its inputs/outputs.
void UpdateEntryFuncAttr(mlir::OpBuilder& builder, mlir::func::FuncOp func) {
  const mlir::FunctionType func_type = func.getFunctionType();
  const mlir::NamedAttribute inputs =
      GetNamesAttr(builder, "inputs", func_type.getInputs(),
                   [](int i) { return absl::StrFormat("input_%d", i); });
  const mlir::NamedAttribute outputs =
      GetNamesAttr(builder, "outputs", func_type.getResults(),
                   [](int i) { return absl::StrFormat("output_%d", i); });
  llvm::SmallVector<mlir::NamedAttribute, 2> named_attrs = {inputs, outputs};
  llvm::ArrayRef<mlir::NamedAttribute> named_array_ref(named_attrs);
  mlir::DictionaryAttr dic_attr = builder.getDictionaryAttr(named_array_ref);
  func->setAttr(kEntryFuncAttr, dic_attr);
}

StatusOr<absl::Span<mlir::Value>> GetExpandedArguments(
    mlir::OpBuilder& builder, mlir::func::FuncOp target_func,
    ExpandedArgumentMap& expanded_arguments, mlir::BlockArgument arg,
    const Mesh* target_mesh) {
  std::optional<Mesh> mesh;
  unsigned int argument_number = arg.getArgNumber();
  if (argument_number == kDeviceIDArgumentNumber) {
    if (target_mesh) {
      mesh = *target_mesh;
    }
  } else {
    TF_ASSIGN_OR_RETURN(const std::optional<Layout> layout,
                        ExtractLayoutFromOperand(arg));
    if (layout) {
      mesh = layout->mesh();
    }
  }
  if (mesh.has_value()) {
    std::vector<mlir::Value>& replications =
        expanded_arguments[argument_number][*mesh];
    if (replications.empty()) {
      const absl::Span<const std::string> devices = GetDevices(*mesh);
      const std::size_t num_devices = devices.size();
      replications.reserve(num_devices);
      if (argument_number == kDeviceIDArgumentNumber) {
        for (int i = 0; i < num_devices; ++i) {
          const auto value_attr = mlir::DenseIntElementsAttr::get<int>(
              mlir::RankedTensorType::get({0}, builder.getI32Type()), {i});
          replications.emplace_back(
              builder.create<mlir::TF::ConstOp>(arg.getLoc(), value_attr));
        }
      } else {
        mlir::TensorType tensor_type =
            arg.getType().dyn_cast_or_null<mlir::TensorType>();
        if (!tensor_type) {
          return errors::InvalidArgument("Could not determine tensor type.");
        }
        for (int i = 0; i < num_devices; ++i) {
          replications.emplace_back(InsertArgumentForDevice(
              builder, target_func, tensor_type, devices[i]));
        }
      }
    }
    return absl::Span<mlir::Value>(replications);
  } else {
    return absl::Span<mlir::Value>();  // no per-device arguments necessary
  }
}

template <typename Results>
mlir::FunctionType GetFunctionType(mlir::OpBuilder& builder,
                                   mlir::func::FuncOp func, Results results) {
  std::vector<mlir::Type> input_types, result_types;
  for (mlir::BlockArgument input : func.getArguments()) {
    input_types.emplace_back(input.getType());
  }
  for (const auto result : results) {
    result_types.emplace_back(result.getType());
  }
  return builder.getFunctionType(input_types, result_types);
}

// Build a new main function that calls the multi-device/translated function.
mlir::LogicalResult BuildOuterMainFunc(
    mlir::ModuleOp module, mlir::func::FuncOp old_main_func,
    mlir::func::FuncOp translated_func, mlir::func::ReturnOp return_op,
    absl::Span<mlir::TF::StatefulPartitionedCallOp> call_ops) {
  llvm::SmallVector<mlir::Attribute, 4> output_layouts;
  for (mlir::TF::StatefulPartitionedCallOp call_op : call_ops) {
    // Then extract all their output layouts.
    mlir::ArrayAttr layouts =
        call_op->getAttr(kLayoutAttr).dyn_cast_or_null<mlir::ArrayAttr>();
    if (!layouts) {
      call_op.emitOpError() << "Could not find op's layouts.";
      return mlir::failure();
    }
    // Here, we assume that the output layouts and the results are in the same
    // ordering--this property should be guaranteed as long as all the results
    // have been expanded (produced by ExpandOperation).
    output_layouts.insert(output_layouts.end(), layouts.begin(), layouts.end());
  }

  mlir::SymbolTable symbol_table(module);
  mlir::Block* module_body = module.getBody();
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(module_body);
  // Build a new main function with no initial attributes/return type.
  mlir::func::FuncOp main_func = mlir::func::FuncOp::create(
      old_main_func.getLoc(), "main", builder.getFunctionType({}, {}));
  mlir::Block* entry_block = main_func.addEntryBlock();
  builder.setInsertionPointToEnd(entry_block);

  // Copy the arguments from the translated function to the new main function.
  std::vector<mlir::Value> inputs;
  for (auto [arg_index, arg] :
       llvm::enumerate(translated_func.getArguments())) {
    main_func.insertArgument(arg_index, arg.getType(),
                             translated_func.getArgAttrDict(arg_index),
                             old_main_func.getLoc());
    inputs.emplace_back(main_func.getArgument(arg_index));
  }

  // Get the type of the translated function.
  mlir::FunctionType func_type = translated_func.getFunctionType();
  // Then build a call op targeting it (reflecting its result types).
  auto expanded_call_op = builder.create<mlir::TF::StatefulPartitionedCallOp>(
      call_ops[0].getLoc(), func_type.getResults(), inputs,
      translated_func.getSymName(),
      /*config=*/builder.getStringAttr(""),
      /*config_proto=*/builder.getStringAttr(""),
      /*executor_type=*/builder.getStringAttr(""));

  // Set the output layout attribute on the new call op.
  llvm::ArrayRef<mlir::Attribute> output_layouts_ref(output_layouts);
  mlir::ArrayAttr output_layouts_attr =
      builder.getArrayAttr(output_layouts_ref);
  expanded_call_op->setAttr(kLayoutAttr, output_layouts_attr);

  // Return all the values from the new call op.
  mlir::Operation::result_range outputs = expanded_call_op.getResults();
  if (return_op) {
    builder.create<mlir::func::ReturnOp>(return_op.getLoc(), outputs);
  } else if (!outputs.empty()) {
    call_ops[0]->emitOpError("Call had results, but they were not used.");
    return mlir::failure();
  }

  // Update the function's type based on the arguments and return values.
  main_func.setFunctionType(GetFunctionType(builder, main_func, outputs));
  UpdateEntryFuncAttr(builder, main_func);

  // Erase the original main func.
  symbol_table.remove(old_main_func);
  old_main_func.erase();
  // Add the new main function to the module's symbol table, ensuring that it's
  // located before all the other functions with the module.
  symbol_table.insert(main_func, module_body->begin());

  return mlir::success();
}

struct DTensorMultiDeviceExpansion
    : public impl::DTensorMultiDeviceExpansionBase<
          DTensorMultiDeviceExpansion> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTable symbol_table(module);
    mlir::func::FuncOp main_func =
        module.lookupSymbol<mlir::func::FuncOp>(kMainFuncName);
    if (!main_func) {
      return;
    }

    mlir::OpBuilder builder = mlir::OpBuilder::atBlockEnd(module.getBody());
    const char* translated_func_name = "multi_device_main";
    mlir::func::FuncOp translated_func =
        mlir::func::FuncOp::create(main_func.getLoc(), translated_func_name,
                                   builder.getFunctionType({}, {}));

    // build the entry block and return op of the translated function
    builder.setInsertionPointToEnd(translated_func.addEntryBlock());
    auto translated_terminator_op =
        builder.create<mlir::func::ReturnOp>(main_func.getLoc());

    // so the function has a "terminator" and we can insert it into the module
    translated_func.setVisibility(mlir::SymbolTable::Visibility::Private);
    symbol_table.insert(translated_func);

    ExpandedArgumentMap expanded_arguments_map;
    for (unsigned i = 1; i < main_func.getNumArguments(); ++i) {
      // Expand all the arguments (in case they're unused).
      StatusOr<absl::Span<mlir::Value>> expanded_arguments =
          GetExpandedArguments(builder, translated_func, expanded_arguments_map,
                               main_func.getArgument(i));
      if (!expanded_arguments.ok()) {
        main_func->emitOpError(
            tsl::NullTerminatedMessage(expanded_arguments.status()));
        return;
      }
    }

    // Note, we cannot simultaneously walk through the call ops and expand
    // them since we'd be creating and removing ops as we walk through them.
    llvm::SmallVector<mlir::TF::StatefulPartitionedCallOp, 8> stateful_call_ops;
    main_func.walk([&](mlir::Operation* op) {
      if (const auto stateful_call_op =
              llvm::dyn_cast_or_null<mlir::TF::StatefulPartitionedCallOp>(op)) {
        if (stateful_call_op->hasAttr(kLayoutAttr) &&
            stateful_call_op->hasAttr(kMeshAttr)) {
          stateful_call_ops.emplace_back(stateful_call_op);
        }
      }
    });

    // Ensure that all the call ops return results via the same op.
    mlir::func::ReturnOp return_op = GetReturnOpFromUsers(
        absl::Span<mlir::TF::StatefulPartitionedCallOp>(stateful_call_ops));
    if (!return_op && !stateful_call_ops.empty()) {
      stateful_call_ops[0]->emitOpError(
          "Calls must be used by exactly one return op.");
      return;
    }

    ExpandedResultsMap expanded_results;
    for (const mlir::TF::StatefulPartitionedCallOp& stateful_call_op :
         stateful_call_ops) {
      mlir::LogicalResult status =
          ExpandOperation(translated_func, return_op, expanded_arguments_map,
                          expanded_results, stateful_call_op);
      if (mlir::failed(status)) {
        return;
      }
    }

    std::vector<mlir::Value> results;
    for (unsigned i = 0; i < return_op->getNumOperands(); ++i) {
      ExpandedResultsMap::iterator search = expanded_results.find(i);
      if (search == expanded_results.end()) {
        results.emplace_back(return_op->getOperand(i));
      } else {
        std::vector<mlir::Value>& values = search->second;
        results.insert(results.end(), values.begin(), values.end());
      }
    }

    // update the operands of the translated return op
    translated_terminator_op->setOperands(results);
    // and, update the function's type accordingly
    translated_func.setFunctionType(GetFunctionType(
        builder, translated_func, absl::Span<mlir::Value>(results)));
    UpdateEntryFuncAttr(builder, translated_func);

    mlir::LogicalResult status = BuildOuterMainFunc(
        module, main_func, translated_func, return_op,
        absl::Span<mlir::TF::StatefulPartitionedCallOp>(stateful_call_ops));
    if (mlir::failed(status)) {
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorMultiDeviceExpansionPass() {
  return std::make_unique<DTensorMultiDeviceExpansion>();
}

}  // namespace dtensor
}  // namespace tensorflow
