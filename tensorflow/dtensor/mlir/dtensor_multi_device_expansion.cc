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
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORMULTIDEVICEEXPANSION
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";
constexpr char kEntryFuncAttr[] = "tf.entry_function";
constexpr char kMainFuncName[] = "main";
constexpr char kIsStatelessAttr[] = "is_stateless";
constexpr int kDeviceIDArgumentNumber = 0;

// This is a map from argument numbers and meshes to per-device values.
// Most arguments will only be expanded on one mesh (the one given by its
// "tf._layout" attribute); however, the device id may be expanded across
// multiple meshes. For example, when main functions have both cpu and tpu
// mesh partitioned calls.
using ExpandedArgumentMap =
    absl::flat_hash_map<int,
                        absl::flat_hash_map<Mesh, std::vector<mlir::Value>>>;

struct ExpandedResults {
  std::optional<Layout> layout;
  std::vector<mlir::Value> results;

  template <typename Value>
  void insert(Value&& value) {
    using T = std::decay_t<Value>;
    if constexpr (std::is_same_v<T, mlir::Value>) {
      results.emplace_back(std::forward<Value>(value));
    } else {
      results.insert(results.end(), value.begin(), value.end());
    }
  }
};

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
template <typename UserType, typename Operations>
mlir::LogicalResult GetUniqueUserFromOps(const Operations& ops,
                                         UserType* result) {
  for (mlir::Operation* op : ops) {
    for (mlir::Operation* user : op->getUsers()) {
      // TODO(twelve): Determine whether we should follow identity ops.
      UserType typed;
      if constexpr (std::is_same_v<UserType, mlir::Operation*>) {
        typed = user;
      } else {
        typed = llvm::dyn_cast_or_null<UserType>(user);
      }
      if (typed) {
        if (*result == nullptr) {
          *result = typed;
        } else if (*result != typed) {
          return mlir::failure();
        }
      } else {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
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
    const Mesh* target_mesh = nullptr, bool has_device_id = true);

StatusOr<std::optional<std::vector<Layout>>> GetResourceLayouts(
    mlir::Operation* op) {
  if (op->hasAttr(kNewResourceArgLayouts)) {
    auto attrs = op->getAttrOfType<mlir::ArrayAttr>(kNewResourceArgLayouts);
    std::vector<Layout> layouts;
    layouts.reserve(attrs.size());
    for (mlir::Attribute attr : attrs) {
      auto string_attr = mlir::cast<mlir::StringAttr>(attr);
      auto layout = Layout::FromString(string_attr.str());
      if (layout.ok()) {
        layouts.emplace_back(std::move(layout.value()));
      } else {
        return layout.status();
      }
    }
    return layouts;
  } else {
    return std::nullopt;
  }
}

bool IsResource(mlir::Value value) {
  return mlir::isa<mlir::TF::ResourceType>(
      getElementTypeOrSelf(value.getType()));
}

StatusOr<std::optional<Layout>> FindResourceLayout(mlir::BlockArgument arg) {
  uint32_t arg_num = arg.getArgNumber();
  for (mlir::Operation* user : arg.getUsers()) {
    auto resource_layouts = GetResourceLayouts(user);
    if (resource_layouts.ok()) {
      const auto& opt = resource_layouts.value();
      if (!opt || opt->empty()) {
        continue;
      }
    } else {
      return resource_layouts.status();
    }

    auto resource_indices = user->getAttrOfType<mlir::DenseIntElementsAttr>(
        kNewResourceLayoutIndices);
    if (!resource_indices) {
      return absl::InvalidArgumentError(
          absl::StrCat("missing ", kNewResourceLayoutIndices));
    }

    for (auto [i, index] : llvm::enumerate(resource_indices)) {
      int64_t index_value = index.getSExtValue();
      if (index_value == arg_num) {
        return (resource_layouts.value())->at(i);
      }
    }
  }

  return std::nullopt;
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

mlir::LogicalResult MakeParallelExecute(
    mlir::OpBuilder& builder, mlir::tf_device::LaunchOp launch_op,
    absl::Span<const std::string> devices,
    std::vector<mlir::IRMapping>& mappings) {
  const unsigned num_devices = devices.size();
  const unsigned num_results = launch_op->getNumResults();

  std::vector<mlir::Type> result_types;
  result_types.reserve(num_results * num_devices);

  // Expand the result types across all the devices.
  for (int i = 0; i < num_devices; ++i) {
    for (mlir::Type result_type : launch_op.getResultTypes()) {
      result_types.emplace_back(result_type);
    }
  }

  // Build the ParallelExecuteOp.
  mlir::Location loc = launch_op.getLoc();
  mlir::tf_device::ParallelExecuteOp parallel_execute_op =
      builder.create<mlir::tf_device::ParallelExecuteOp>(loc, num_devices,
                                                         result_types);

  // Clone the LaunchOp for each of the devices.
  for (unsigned dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
    mlir::IRMapping& mapping = mappings[dev_idx];
    mlir::Block& block = parallel_execute_op.GetRegionBlockWithIndex(dev_idx);
    builder.setInsertionPointToStart(&block);
    mlir::Operation* clone =
        builder.clone(*((mlir::Operation*)launch_op), mapping);
    clone->setAttr(kDeviceAttr, builder.getStringAttr(devices[dev_idx]));
    builder.create<mlir::tf_device::ReturnOp>(loc, clone->getResults());
  }

  // Map the results of the LaunchOp to the ParallelExecuteOp.
  for (unsigned dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
    mlir::IRMapping& mapping = mappings[dev_idx];
    for (unsigned res_idx = 0; res_idx < num_results; ++res_idx) {
      unsigned expanded_idx = res_idx + dev_idx * num_results;
      mapping.map(launch_op->getResult(res_idx),
                  parallel_execute_op->getResult(expanded_idx));
    }
  }

  return mlir::success();
}

// Rewrites a launch to a ParallelExecuteOp with per-device launches.
mlir::LogicalResult RewriteTPUFunction(mlir::func::FuncOp func,
                                       mlir::tf_device::LaunchOp launch_op,
                                       const Mesh& target_mesh,
                                       mlir::FunctionType& func_type) {
  mlir::OpBuilder builder = mlir::OpBuilder::atBlockBegin(func->getBlock());
  builder.setInsertionPointAfter(launch_op);

  const absl::Span<const std::string> devices = GetDevices(target_mesh);
  const size_t num_devices = devices.size();

  // Maps values from original to per-device representations.
  std::vector<mlir::IRMapping> mappings(num_devices);

  // Expand each of the function's inputs across all the devices.
  // TODO(twelve): This logic assumes all the function's inputs should be
  //               expanded; however, when might that not be true?
  ExpandedArgumentMap expanded_arguments_map;
  const unsigned num_arguments = func.getNumArguments();
  for (unsigned arg_idx = 0; arg_idx < num_arguments; ++arg_idx) {
    mlir::BlockArgument argument = func.getArgument(arg_idx);
    StatusOr<absl::Span<mlir::Value>> expanded_arguments = GetExpandedArguments(
        builder, func, expanded_arguments_map, argument, &target_mesh, false);
    if (expanded_arguments.ok()) {
      // Map the original arguments to their per-device counterparts.
      for (unsigned dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
        mappings[dev_idx].map(argument, expanded_arguments->at(dev_idx));
      }
    } else {
      func->emitOpError(absl::StatusMessageAsCStr(expanded_arguments.status()));
      return mlir::failure();
    }
  }

  // Collect all the original ops.
  llvm::iterator_range<mlir::Region::OpIterator> op_range =
      func.getFunctionBody().getOps();
  std::vector<mlir::Operation*> ops;
  ops.reserve(std::distance(op_range.begin(), op_range.end()));
  for (mlir::Operation& op : op_range) {
    ops.emplace_back(&op);
  }

  // Clone all the original ops.
  std::vector<mlir::Value> results;
  for (mlir::Operation* op : ops) {
    builder.setInsertionPointAfter(op);

    if (llvm::isa<mlir::tf_device::LaunchOp>(op)) {
      if (op == (mlir::Operation*)launch_op) {
        // Build the parallel execute operation.
        mlir::LogicalResult result =
            MakeParallelExecute(builder, launch_op, devices, mappings);
        if (mlir::failed(result)) {
          return result;
        }
      }
    } else if (llvm::isa<mlir::func::ReturnOp>(op)) {
      // Collect the per-device results.
      results.reserve(op->getNumOperands() * num_devices);

      for (mlir::Value operand : op->getOperands()) {
        for (unsigned dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
          results.emplace_back(mappings[dev_idx].lookup(operand));
        }
      }

      builder.create<mlir::func::ReturnOp>(op->getLoc(), results);
    } else {
      // Clone the operation across all the devices.
      for (unsigned dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
        mlir::Operation* clone = builder.clone(*op, mappings[dev_idx]);
        clone->setAttr(kDeviceAttr, builder.getStringAttr(devices[dev_idx]));
      }
    }
  }

  // Erase the original non-launch ops (go backwards to remove users first).
  for (int op_idx = (int)ops.size() - 1; op_idx >= 0; --op_idx) {
    mlir::Operation* op = ops[op_idx];
    if (!llvm::isa<mlir::tf_device::LaunchOp>(op) ||
        (op == (mlir::Operation*)launch_op)) {
      op->erase();
    }
  }

  // Erase the function's original arguments.
  for (unsigned arg_idx = 0; arg_idx < num_arguments; ++arg_idx) {
    func.eraseArgument(0);
  }

  // Update the function's type.
  func_type = GetFunctionType(builder, func, results);
  func.setFunctionType(func_type);

  return mlir::success();
}

// Rewrites a call-like op into an equivalent op for each device;
// de/multiplexes the per-device inputs/outputs for each "expanded" op.
// Usable when there are not any device launch ops (typically on CPU/GPU).
template <typename OperationType>
mlir::LogicalResult ExpandOperation(
    mlir::func::FuncOp target_func, mlir::func::ReturnOp return_op,
    ExpandedArgumentMap& expanded_arguments,
    std::vector<ExpandedResults>& expanded_results, const Mesh& target_mesh,
    OperationType op);

// Rewrites a call-like op targeting a single device launch op into
// a parallel execute.
template <typename OperationType>
mlir::LogicalResult ExpandTPUOperation(
    mlir::func::FuncOp target_func, mlir::func::ReturnOp return_op,
    ExpandedArgumentMap& expanded_arguments,
    std::vector<ExpandedResults>& expanded_results, const Mesh& target_mesh,
    OperationType op) {
  mlir::FunctionType func_type;
  mlir::OpBuilder builder(target_func.getBody());
  mlir::ModuleOp module = op->template getParentOfType<mlir::ModuleOp>();
  mlir::func::FuncOp func = module.lookupSymbol<mlir::func::FuncOp>(op.getF());

  mlir::tf_device::LaunchOp launch_op;
  func.walk([&](mlir::tf_device::LaunchOp op) {
    (op.getBody()).walk([&](mlir::Operation* child) {
      if (llvm::isa<mlir::TF::TPUExecuteOp>(child) ||
          llvm::isa<mlir::TF::TPUExecuteAndUpdateVariablesOp>(child)) {
        launch_op = op;
        return mlir::WalkResult::interrupt();
      } else {
        return mlir::WalkResult::advance();
      }
    });
  });
  if (launch_op) {
    // Found a device launch with TPUExecute, try expanding it.
    if (mlir::failed(
            RewriteTPUFunction(func, launch_op, target_mesh, func_type))) {
      return mlir::failure();
    }
  } else {
    // There were not any TPUExecute ops, fallback to the conventional path.
    return ExpandOperation(target_func, return_op, expanded_arguments,
                           expanded_results, target_mesh, op);
  }

  llvm::SmallVector<mlir::Value, 8> operands;
  for (const mlir::Value& operand : op->getOperands()) {
    if (const auto arg = mlir::dyn_cast_or_null<mlir::BlockArgument>(operand)) {
      const StatusOr<absl::Span<mlir::Value>> new_args = GetExpandedArguments(
          builder, target_func, expanded_arguments, arg, &target_mesh);
      if (!new_args.ok()) {
        op->emitOpError(absl::StatusMessageAsCStr(new_args.status()));
        return mlir::failure();
      } else if (new_args->empty()) {
        operands.push_back(operand);
      } else {
        operands.insert(operands.end(), new_args->begin(), new_args->end());
      }
    } else {
      operands.push_back(operand);
    }
  }

  auto call_op = builder.create<OperationType>(
      op->getLoc(), func_type.getResults(), operands, op.getFAttr(),
      /*config=*/builder.getStringAttr(""),
      /*config_proto=*/builder.getStringAttr(""),
      /*executor_type=*/builder.getStringAttr(""));
  const absl::Span<const std::string> devices = GetDevices(target_mesh);
  const size_t num_devices = devices.size();

  if (return_op) {
    mlir::Operation::operand_range operands = return_op->getOperands();
    // Expand each operand of the return operation.
    for (const auto [result_index, operand] : llvm::enumerate(operands)) {
      // (All the operands should originate from the original call op.)
      if (op == operand.getDefiningOp()) {
        const mlir::Operation::result_range results = op->getResults();
        const mlir::Operation::result_range::iterator search =
            llvm::find(results, operand);
        const size_t i = search - results.begin();
        // For each device--
        for (int j = 0; j < num_devices; j++) {
          // Find the corresponding expanded output within the new op.
          mlir::Value result = call_op->getResult(i * num_devices + j);
          auto identity_op = builder.create<mlir::TF::IdentityOp>(
              result.getLoc(), result.getType(), result);
          expanded_results[result_index].insert(
              (mlir::Value)identity_op.getResult());
        }
      }
    }
  }

  return mlir::success();
}

template <typename Operation>
mlir::LogicalResult ExpandOperation(
    mlir::func::FuncOp target_func, mlir::func::ReturnOp return_op,
    ExpandedArgumentMap& expanded_arguments,
    std::vector<ExpandedResults>& expanded_results, const Mesh& target_mesh,
    Operation op) {
  mlir::OpBuilder builder(target_func.getBody());
  const absl::Span<const std::string> devices = GetDevices(target_mesh);
  const size_t num_devices = devices.size();

  llvm::SmallVector<Operation> replications;
  for (size_t i = 0; i < num_devices; ++i) {
    llvm::SmallVector<mlir::Value, 8> operands;
    for (const mlir::Value& operand : op->getOperands()) {
      if (const auto arg =
              mlir::dyn_cast_or_null<mlir::BlockArgument>(operand)) {
        const StatusOr<absl::Span<mlir::Value>> new_args = GetExpandedArguments(
            builder, target_func, expanded_arguments, arg, &target_mesh);
        if (!new_args.ok()) {
          op->emitOpError(absl::StatusMessageAsCStr(new_args.status()));
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

    // Set the "is_stateless" attribute to ensure that side-effect analysis
    // does not set the per-device call ops to depend on one another (see
    // `CanHaveSideEffects`), which would cause collectives to hang.
    new_op->setAttr(kIsStatelessAttr, builder.getBoolAttr(true));
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
        const size_t result_number = search - results.begin();
        for (const Operation& replication : replications) {
          expanded_results[i].insert(
              (mlir::Value)replication->getResult(result_number));
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
    const Mesh* target_mesh, bool has_device_id) {
  std::optional<Mesh> mesh;
  unsigned int argument_number = arg.getArgNumber();
  if (!has_device_id || (argument_number == kDeviceIDArgumentNumber)) {
    if (target_mesh) {
      mesh = *target_mesh;
    }
  } else {
    TF_ASSIGN_OR_RETURN(std::optional<Layout> layout,
                        ExtractLayoutFromOperand(arg));
    if (layout) {
      mesh = layout->mesh();

      if (mesh->IsEmpty()) {
        if (target_mesh) {
          mesh = *target_mesh;
        } else if (IsResource(arg)) {
          TF_ASSIGN_OR_RETURN(layout, FindResourceLayout(arg));
          if (layout) {
            mesh = layout->mesh();
          } else {
            return absl::InvalidArgumentError(
                absl::StrCat("Could not find resource layout for %arg",
                             arg.getArgNumber(), "!"));
          }
        }
      }
    }
  }
  if (mesh.has_value()) {
    std::vector<mlir::Value>& replications =
        expanded_arguments[argument_number][*mesh];
    if (replications.empty()) {
      const absl::Span<const std::string> devices = GetDevices(*mesh);
      const size_t num_devices = devices.size();
      replications.reserve(num_devices);
      if (has_device_id && argument_number == kDeviceIDArgumentNumber) {
        for (int i = 0; i < num_devices; ++i) {
          const auto value_attr = mlir::DenseIntElementsAttr::get<int>(
              mlir::RankedTensorType::get({}, builder.getI32Type()), {i});
          replications.emplace_back(
              builder.create<mlir::TF::ConstOp>(arg.getLoc(), value_attr));
        }
      } else {
        mlir::TensorType tensor_type =
            mlir::dyn_cast_or_null<mlir::TensorType>(arg.getType());
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

struct InferredResourceAttributes {
  mlir::Attribute layouts;
  mlir::Attribute indices;

  InferredResourceAttributes(mlir::Attribute layouts_, mlir::Attribute indices_)
      : layouts(layouts_), indices(indices_) {}
};

template <typename Operations>
mlir::LogicalResult GetInferredResourceAttributes(
    mlir::OpBuilder& builder, const Operations& call_ops,
    std::optional<InferredResourceAttributes>* resource_attrs) {
  llvm::SmallVector<mlir::Attribute, 8> resource_layouts;
  llvm::SmallVector<int32_t, 8> resource_indices;
  for (mlir::Operation* call_op : call_ops) {
    const auto resource_layouts_attr =
        call_op->getAttrOfType<mlir::ArrayAttr>(kNewResourceArgLayouts);
    const auto resource_indices_attr =
        call_op->getAttrOfType<mlir::DenseIntElementsAttr>(
            kNewResourceLayoutIndices);
    if (resource_indices_attr && resource_layouts_attr) {
      for (auto [index, layout] :
           llvm::zip(resource_indices_attr, resource_layouts_attr)) {
        // Build up the lists of resource indices and layouts.
        resource_indices.emplace_back(index.getSExtValue());
        resource_layouts.emplace_back(layout);
      }
    }
  }
  if (!resource_layouts.empty()) {
    resource_attrs->emplace(builder.getArrayAttr(resource_layouts),
                            builder.getI32VectorAttr(resource_indices));
  }
  return mlir::success();
}

// Build a new main function that calls the multi-device/translated function.
template <typename Operations>
mlir::LogicalResult BuildOuterMainFunc(
    mlir::ModuleOp module, mlir::func::FuncOp old_main_func,
    mlir::func::FuncOp translated_func, mlir::func::ReturnOp return_op,
    const std::vector<ExpandedResults>& expanded_results,
    mlir::ArrayAttr num_local_outputs_attr, Operations&& call_ops) {
  using CallOp = typename std::decay_t<Operations>::value_type;

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
  // Then build a call op targeting it (reflecting its result types)
  auto expanded_call_op =
      builder.create<CallOp>(call_ops[0].getLoc(), func_type.getResults(),
                             inputs, translated_func.getSymName(),
                             /*config=*/builder.getStringAttr(""),
                             /*config_proto=*/builder.getStringAttr(""),
                             /*executor_type=*/builder.getStringAttr(""));

  // Set the output layout attribute on the new call op.
  std::vector<std::optional<Layout>> output_layouts;
  std::transform(expanded_results.begin(), expanded_results.end(),
                 std::back_inserter(output_layouts),
                 [](const ExpandedResults& result) { return result.layout; });
  SetLayoutOnOp(expanded_call_op, builder, output_layouts);

  expanded_call_op->setAttr(kNumLocalOutputsAttr, num_local_outputs_attr);

  std::optional<InferredResourceAttributes> resource_attrs;
  if (failed(
          GetInferredResourceAttributes(builder, call_ops, &resource_attrs))) {
    return mlir::failure();
  }

  if (resource_attrs) {
    expanded_call_op->setAttr(kNewResourceArgLayouts, resource_attrs->layouts);
    expanded_call_op->setAttr(kNewResourceLayoutIndices,
                              resource_attrs->indices);
  }

  // Return all the values from the new call op.
  mlir::Operation::result_range outputs = expanded_call_op.getResults();
  if (return_op || outputs.empty()) {
    mlir::Location loc = return_op ? return_op.getLoc() : main_func.getLoc();
    builder.create<mlir::func::ReturnOp>(loc, outputs);
  } else {
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

absl::Status ExtractResultLayouts(
    mlir::Operation* op, mlir::func::ReturnOp return_op,
    std::vector<ExpandedResults>& expanded_results) {
  if (!return_op || (return_op.getNumOperands() == 0)) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(std::vector<std::optional<Layout>> layouts,
                      ExtractLayoutFromOp(op));
  mlir::Operation::operand_range operands = return_op.getOperands();
  for (auto [layout_index, result] : llvm::enumerate(op->getResults())) {
    auto search = std::find(operands.begin(), operands.end(), result);
    if (search == operands.end()) {
      continue;
    }
    size_t result_index = std::distance(operands.begin(), search);
    expanded_results[result_index].layout = layouts[layout_index];
  }
  return absl::OkStatus();
}

struct DTensorMultiDeviceExpansion
    : public impl::DTensorMultiDeviceExpansionBase<
          DTensorMultiDeviceExpansion> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    auto multi_device_mode =
        module->getAttrOfType<mlir::BoolAttr>(dtensor::kEnableMultiDeviceMode);
    if (!multi_device_mode || !multi_device_mode.getValue()) {
      return;  // Skip modules for whom multi-device mode is disabled.
    }

    mlir::SymbolTable symbol_table(module);
    mlir::func::FuncOp main_func =
        module.lookupSymbol<mlir::func::FuncOp>(kMainFuncName);
    if (!main_func) {
      return;
    }

    std::string translated_func_name =
        llvm::formatv("_multi_device_func_{0}_{1}", OpHash(module),
                      OpHash(main_func))
            .str();
    mlir::OpBuilder builder = mlir::OpBuilder::atBlockEnd(module.getBody());
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
            absl::StatusMessageAsCStr(expanded_arguments.status()));
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
    mlir::func::ReturnOp return_op;
    if (GetUniqueUserFromOps(stateful_call_ops, &return_op).failed()) {
      stateful_call_ops[0]->emitOpError(
          "Calls must be used by exactly one return op.");
      return;
    }

    std::vector<ExpandedResults> expanded_results(
        return_op ? return_op->getNumOperands() : 0);
    for (const mlir::TF::StatefulPartitionedCallOp& stateful_call_op :
         stateful_call_ops) {
      const absl::Status status =
          ExtractResultLayouts(stateful_call_op, return_op, expanded_results);
      const StatusOr<std::optional<Mesh>> mesh =
          status.ok() ? ExtractDeviceMeshFromOp(stateful_call_op) : status;
      if (!(mesh.ok() && *mesh)) {
        stateful_call_op->emitOpError("Failed to retrieve op mesh or layout.");
        return;
      }

      const Mesh& target_mesh = **mesh;
      if (target_mesh.IsSingleDevice()) {
        stateful_call_op->emitOpError(
            "Unimplemented, single-device expansion support.");
        return;
      } else if (target_mesh.is_tpu_mesh()) {
        if (mlir::failed(ExpandTPUOperation(
                translated_func, return_op, expanded_arguments_map,
                expanded_results, target_mesh, stateful_call_op))) {
          return;
        }
      } else {
        if (mlir::failed(ExpandOperation(
                translated_func, return_op, expanded_arguments_map,
                expanded_results, target_mesh, stateful_call_op))) {
          return;
        }
      }
    }

    std::vector<mlir::Value> results;
    llvm::SmallVector<mlir::Attribute, 8> num_local_outputs;
    if (return_op) {
      for (unsigned i = 0; i < return_op->getNumOperands(); ++i) {
        std::vector<mlir::Value>& values = expanded_results[i].results;
        int num_outputs;
        if (values.empty()) {
          results.emplace_back(return_op->getOperand(i));
          num_outputs = 1;
        } else {
          results.insert(results.end(), values.begin(), values.end());
          num_outputs = values.size();
        }
        num_local_outputs.emplace_back(builder.getI64IntegerAttr(num_outputs));
      }
    }

    mlir::ArrayAttr num_local_outputs_attr =
        builder.getArrayAttr(num_local_outputs);

    // update the operands of the translated return op
    translated_terminator_op->setOperands(results);
    // and, update the function's type accordingly
    translated_func.setFunctionType(GetFunctionType(
        builder, translated_func, absl::Span<mlir::Value>(results)));
    UpdateEntryFuncAttr(builder, translated_func);

    mlir::LogicalResult status = BuildOuterMainFunc(
        module, main_func, translated_func, return_op, expanded_results,
        num_local_outputs_attr, stateful_call_ops);
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
