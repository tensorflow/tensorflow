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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
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
template <typename Operations>
mlir::LogicalResult GetReturnOpFromUsers(Operations&& ops,
                                         mlir::func::ReturnOp* return_op) {
  for (mlir::Operation* op : ops) {
    for (mlir::Operation* user : op->getUsers()) {
      // TODO(twelve): Determine whether we should follow identity ops.
      if (mlir::func::ReturnOp op =
              llvm::dyn_cast_or_null<mlir::func::ReturnOp>(user)) {
        if (*return_op) {
          if (*return_op != op) {
            return mlir::failure();
          }
        } else {
          *return_op = op;
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
    const Mesh* target_mesh = nullptr);

StatusOr<std::optional<std::vector<Layout>>> GetResourceLayouts(
    mlir::Operation* op) {
  if (op->hasAttr(kNewResourceArgLayouts)) {
    auto attrs = op->getAttrOfType<mlir::ArrayAttr>(kNewResourceArgLayouts);
    std::vector<Layout> layouts;
    layouts.reserve(attrs.size());
    for (mlir::Attribute attr : attrs) {
      auto string_attr = attr.cast<mlir::StringAttr>();
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
  return getElementTypeOrSelf(value.getType()).isa<mlir::TF::ResourceType>();
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

mlir::tf_device::ClusterFuncOp ExtractDeviceClusterFromFunctionCall(
    mlir::TF::StatefulPartitionedCallOp op) {
  mlir::tf_device::ClusterFuncOp result;
  mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
  mlir::func::FuncOp func = module.lookupSymbol<mlir::func::FuncOp>(op.getF());
  func->walk(
      [&](mlir::tf_device::ClusterFuncOp cluster_op) { result = cluster_op; });
  return result;
}

// Adds metadata used in TPU Compilation to `cluster` as attributes.
void AddMetadataToTPUCluster(const Mesh& mesh_config, int64_t num_devices,
                             mlir::tf_device::ClusterFuncOp cluster,
                             mlir::OpBuilder* builder) {
  cluster->setAttr("_tpu_replicate",
                   builder->getStringAttr(mesh_config.ToString()));
  cluster->setAttr("step_marker_location", builder->getStringAttr(""));
  cluster->setAttr("padding_map", builder->getArrayAttr({}));
  cluster->setAttr("use_spmd_for_xla_partitioning",
                   builder->getBoolAttr(false));
  cluster->setAttr("topology", builder->getStringAttr(""));
  cluster->setAttr("device_assignment", builder->getArrayAttr({}));
  cluster->setAttr("num_replicas", builder->getI64IntegerAttr(num_devices));
  cluster->setAttr("num_cores_per_replica", builder->getI64IntegerAttr(1LL));
}

// Rewrites a call-like op targeting a TF device cluster func
// into a cluster func that has partitioned inputs and outputs ops;
// it will be rewritten by TPURewritePass into per-device TPUExecute ops.
template <typename Operation>
mlir::LogicalResult ExpandTPUOperation(
    mlir::func::FuncOp target_func, mlir::func::ReturnOp return_op,
    ExpandedArgumentMap& expanded_arguments,
    std::vector<ExpandedResults>& expanded_results, const Mesh& target_mesh,
    Operation op) {
  const absl::Span<const std::string> devices = GetDevices(target_mesh);
  const std::size_t num_devices = devices.size();

  mlir::OpBuilder builder(target_func.getBody());
  mlir::ArrayAttr partition_dims =
      builder.getArrayAttr(llvm::ArrayRef<mlir::Attribute>());

  llvm::SmallVector<mlir::Value, 8> operands;
  for (const mlir::Value& operand : op->getOperands()) {
    if (const auto arg = operand.dyn_cast_or_null<mlir::BlockArgument>()) {
      const StatusOr<absl::Span<mlir::Value>> new_args = GetExpandedArguments(
          builder, target_func, expanded_arguments, arg, &target_mesh);
      if (!new_args.ok()) {
        op->emitOpError(tsl::NullTerminatedMessage(new_args.status()));
        return mlir::failure();
      } else if (new_args->empty()) {
        operands.push_back(operand);
      } else {
        llvm::ArrayRef<mlir::Value> values(new_args.value().begin(),
                                           new_args.value().end());
        auto input_op = builder.create<mlir::TF::TPUPartitionedInputV2Op>(
            op->getLoc(), operand.getType(), values,
            /*partition_dims=*/partition_dims,
            /*is_packed=*/builder.getBoolAttr(false),
            /*_XlaSharding=*/builder.getStringAttr(""));
        operands.push_back(input_op);
      }
    } else {
      operands.push_back(operand);
    }
  }

  auto cluster_op = ExtractDeviceClusterFromFunctionCall(op);
  if (!cluster_op) {
    op->emitOpError("Could not find device cluster func!");
    return mlir::failure();
  }

  auto new_cluster_op = builder.create<mlir::tf_device::ClusterFuncOp>(
      op->getLoc(), cluster_op->getResultTypes(), cluster_op.getFuncAttr(),
      operands);
  AddMetadataToTPUCluster(target_mesh, num_devices, new_cluster_op, &builder);

  if (return_op) {
    llvm::SmallDenseMap<size_t, mlir::Operation::result_range> replications;
    for (const auto [result_number, result] :
         llvm::enumerate(new_cluster_op->getResults())) {
      llvm::SmallVector<mlir::Type, 8> result_types(num_devices,
                                                    result.getType());
      auto output_op = builder.create<mlir::TF::TPUPartitionedOutputV2Op>(
          op->getLoc(), result_types, result,
          /*partition_dims=*/partition_dims,
          /*_XlaSharding=*/builder.getStringAttr(""));
      replications.try_emplace(result_number, output_op->getResults());
    }

    mlir::Operation::operand_range operands = return_op->getOperands();
    for (const auto [i, operand] : llvm::enumerate(operands)) {
      if (op == operand.getDefiningOp()) {
        const mlir::Operation::result_range results = op->getResults();
        const mlir::Operation::result_range::iterator search =
            llvm::find(results, operand);
        const std::size_t result_number = search - results.begin();
        const mlir::Operation::result_range replicated_results =
            replications.at(result_number);
        expanded_results[i].insert(replicated_results);
      }
    }
  }

  return mlir::success();
}

// Rewrites a call-like op into an equivalent op for each device;
// de/multiplexes the per-device inputs/outputs for each "expanded" op.
// Only usable on CPU/GPU devices, which do not require additional rewriting.
template <typename Operation>
mlir::LogicalResult ExpandOperation(
    mlir::func::FuncOp target_func, mlir::func::ReturnOp return_op,
    ExpandedArgumentMap& expanded_arguments,
    std::vector<ExpandedResults>& expanded_results, const Mesh& target_mesh,
    Operation op) {
  mlir::OpBuilder builder(target_func.getBody());
  const absl::Span<const std::string> devices = GetDevices(target_mesh);
  const std::size_t num_devices = devices.size();

  llvm::SmallVector<Operation> replications;
  for (std::size_t i = 0; i < num_devices; ++i) {
    llvm::SmallVector<mlir::Value, 8> operands;
    for (const mlir::Value& operand : op->getOperands()) {
      if (const auto arg = operand.dyn_cast_or_null<mlir::BlockArgument>()) {
        const StatusOr<absl::Span<mlir::Value>> new_args = GetExpandedArguments(
            builder, target_func, expanded_arguments, arg, &target_mesh);
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
        const std::size_t result_number = search - results.begin();
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
    const Mesh* target_mesh) {
  std::optional<Mesh> mesh;
  unsigned int argument_number = arg.getArgNumber();
  if (argument_number == kDeviceIDArgumentNumber) {
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
      const std::size_t num_devices = devices.size();
      replications.reserve(num_devices);
      if (argument_number == kDeviceIDArgumentNumber) {
        for (int i = 0; i < num_devices; ++i) {
          const auto value_attr = mlir::DenseIntElementsAttr::get<int>(
              mlir::RankedTensorType::get({}, builder.getI32Type()), {i});
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

Status ExtractResultLayouts(mlir::Operation* op, mlir::func::ReturnOp return_op,
                            std::vector<ExpandedResults>& expanded_results) {
  if (!return_op || (return_op.getNumOperands() == 0)) {
    return OkStatus();
  }
  TF_ASSIGN_OR_RETURN(std::vector<std::optional<Layout>> layouts,
                      ExtractLayoutFromOp(op));
  mlir::Operation::operand_range operands = return_op.getOperands();
  for (auto [layout_index, result] : llvm::enumerate(op->getResults())) {
    auto search = std::find(operands.begin(), operands.end(), result);
    if (search == operands.end()) {
      continue;
    }
    std::size_t result_index = std::distance(operands.begin(), search);
    expanded_results[result_index].layout = layouts[layout_index];
  }
  return OkStatus();
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
    mlir::func::ReturnOp return_op;
    if (GetReturnOpFromUsers(stateful_call_ops, &return_op).failed()) {
      stateful_call_ops[0]->emitOpError(
          "Calls must be used by exactly one return op.");
      return;
    }

    std::vector<ExpandedResults> expanded_results(
        return_op ? return_op->getNumOperands() : 0);
    for (const mlir::TF::StatefulPartitionedCallOp& stateful_call_op :
         stateful_call_ops) {
      const Status status =
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
