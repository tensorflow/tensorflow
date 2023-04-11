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

mlir::BlockArgument MakeArgumentForDevice(mlir::Builder& builder,
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

StatusOr<absl::Span<mlir::Value>> GetExpandedArguments(
    mlir::func::FuncOp func, ExpandedArgumentMap& expanded_arguments,
    unsigned int argument_number, const Mesh* target_mesh = nullptr);

template <typename Operation>
Status ExpandOperation(ExpandedArgumentMap& expanded_arguments_map,
                       absl::Span<const std::string> devices, Operation op,
                       const Layout& layout) {
  auto func = op->template getParentOfType<mlir::func::FuncOp>();
  if (!func) {
    // This line should be unreachable within the current framework.
    // This function is only called on operations discovered while walking
    // through the main function.
    return errors::InvalidArgument("Operator not within function.");
  }

  mlir::OpBuilder builder(op);
  const Mesh& mesh = layout.mesh();
  const std::size_t num_devices = devices.size();

  llvm::SmallVector<Operation> replications;
  for (std::size_t i = 0; i < num_devices; ++i) {
    llvm::SmallVector<mlir::Value, 8> operands;
    for (const mlir::Value& operand : op->getOperands()) {
      if (const auto arg = operand.dyn_cast_or_null<mlir::BlockArgument>()) {
        TF_ASSIGN_OR_RETURN(const absl::Span<mlir::Value> expanded_arguments,
                            GetExpandedArguments(func, expanded_arguments_map,
                                                 arg.getArgNumber(), &mesh));
        if (expanded_arguments.empty()) {
          operands.push_back(operand);
        } else {
          operands.push_back(expanded_arguments[i]);
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

  mlir::func::ReturnOp return_op;
  for (const mlir::OpOperand& user : op->getUses()) {
    const mlir::Operation* owner = user.getOwner();
    if (!(return_op = llvm::dyn_cast_or_null<mlir::func::ReturnOp>(owner))) {
      // TODO(twelve) : Determine whether this restriction should be lifted.
      return errors::InvalidArgument("Call result must be used by return op.");
    }
  }

  if (return_op) {
    llvm::SmallVector<mlir::Value, 8> operands;
    for (const mlir::Value operand : return_op->getOperands()) {
      if (op == operand.getDefiningOp()) {
        const mlir::Operation::result_range results = op->getResults();
        const mlir::Operation::result_range::iterator search =
            llvm::find(results, operand);
        const std::size_t result_number = search - results.begin();
        for (const Operation& replication : replications) {
          operands.push_back(replication->getResult(result_number));
        }
      } else {
        operands.push_back(operand);
      }
    }

    llvm::SmallVector<mlir::Type, 8> results;
    for (const mlir::Value& operand : operands) {
      results.push_back(operand.getType());
    }

    const mlir::FunctionType func_type = func.getFunctionType();
    func.removeResAttrsAttr();
    func.setFunctionType(
        builder.getFunctionType(func_type.getInputs(), results));

    builder.create<mlir::func::ReturnOp>(return_op->getLoc(), operands);

    return_op->erase();
  }

  return OkStatus();
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

// Extracts the operation's layouts, then expands it across them.
template <typename Operation>
mlir::LogicalResult ExpandOperations(ExpandedArgumentMap& expanded_arguments,
                                     Operation op) {
  const StatusOr<std::optional<Mesh>> mesh = ExtractDeviceMeshFromOp(op);
  const StatusOr<std::vector<std::optional<Layout>>> layouts =
      ExtractLayoutFromOp(op);
  if (!((mesh.ok() && *mesh) && (layouts.ok() && !layouts->empty()))) {
    op->emitOpError("Failed to retrieve op mesh or layout.");
    return mlir::failure();
  }

  bool expanded = false;
  for (const std::optional<Layout>& layout : *layouts) {
    if (layout) {
      const Mesh& layout_mesh = layout->mesh();
      if (**mesh != layout_mesh) {
        op->emitOpError("Unimplemented, outputs not on op mesh.");
        return mlir::failure();
      } else if (layout_mesh.IsSingleDevice()) {
        op->emitOpError("Unimplemented, single-device expansion support.");
        return mlir::failure();
      } else {
        const absl::Span<const std::string> devices = GetDevices(layout_mesh);
        const Status status =
            ExpandOperation(expanded_arguments, devices, op, *layout);
        if (status.ok()) {
          expanded = true;
        } else {
          op->emitOpError(status.error_message());
          return mlir::failure();
        }
      }
    }
  }

  if (expanded) {
    op->erase();
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
    mlir::func::FuncOp func, ExpandedArgumentMap& expanded_arguments,
    unsigned int argument_number, const Mesh* target_mesh) {
  if (func.getName() != kMainFuncName) {
    return absl::Span<mlir::Value>();  // only expand main function arguments
  }
  const mlir::BlockArgument arg = func.getArgument(argument_number);
  std::optional<Mesh> mesh;
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
      mlir::Block& func_block = func.getBody().front();
      mlir::OpBuilder builder(&(func_block.front()));
      if (argument_number == kDeviceIDArgumentNumber) {
        mlir::Location loc = func_block.front().getLoc();
        for (int i = 0; i < num_devices; ++i) {
          const auto value_attr = mlir::DenseIntElementsAttr::get<int>(
              mlir::RankedTensorType::get({0}, builder.getI32Type()), {i});
          replications.emplace_back(
              builder.create<mlir::TF::ConstOp>(loc, value_attr));
        }
      } else {
        mlir::TensorType tensor_type =
            arg.getType().dyn_cast_or_null<mlir::TensorType>();
        if (!tensor_type) {
          return errors::InvalidArgument("Could not determine tensor type.");
        }
        for (int i = 0; i < num_devices; ++i) {
          replications.emplace_back(
              MakeArgumentForDevice(builder, func, tensor_type, devices[i]));
        }
      }
    }
    return absl::Span<mlir::Value>(replications);
  } else {
    return absl::Span<mlir::Value>();  // no per-device arguments necessary
  }
}

struct DTensorMultiDeviceExpansion
    : public impl::DTensorMultiDeviceExpansionBase<
          DTensorMultiDeviceExpansion> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::func::FuncOp main_func =
        module.lookupSymbol<mlir::func::FuncOp>(kMainFuncName);
    if (!main_func) {
      return;
    }

    ExpandedArgumentMap expanded_arguments_map;
    for (unsigned i = 1; i < main_func.getNumArguments(); ++i) {
      // Expand all the arguments (in case they're unused).
      StatusOr<absl::Span<mlir::Value>> expanded_arguments =
          GetExpandedArguments(main_func, expanded_arguments_map, i);
      if (!expanded_arguments.ok()) {
        main_func->emitOpError(expanded_arguments.status().error_message());
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

    for (const mlir::TF::StatefulPartitionedCallOp& stateful_call_op :
         stateful_call_ops) {
      mlir::LogicalResult status =
          ExpandOperations(expanded_arguments_map, stateful_call_op);
      if (status.failed()) {
        return;
      }
    }

    if (main_func && !expanded_arguments_map.empty()) {
      mlir::OpBuilder builder(main_func);
      const mlir::FunctionType func_type = main_func.getFunctionType();
      const llvm::ArrayRef<mlir::Type> inputs = func_type.getInputs();
      llvm::SmallVector<mlir::Type, 8> next_inputs;
      unsigned num_erased = 0;
      for (unsigned i = 0; i < inputs.size(); ++i) {
        const ExpandedArgumentMap::iterator search =
            expanded_arguments_map.find(i);
        // Always erase the device id, even when it's unexpanded.
        if ((search == expanded_arguments_map.end()) &&
            (i != kDeviceIDArgumentNumber)) {
          next_inputs.push_back(inputs[i]);
        } else {
          main_func.eraseArgument(i - num_erased);
          num_erased += 1;
        }
      }
      main_func.setFunctionType(
          builder.getFunctionType(next_inputs, func_type.getResults()));
      UpdateEntryFuncAttr(builder, main_func);
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
