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

#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_constants.h"
#include "xla/service/computation_placer.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/platform/protobuf.h"  // IWYU pragma: keep
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_config.pb.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_SINKVARIABLEASNAMEDARRAYPASS
#define GEN_PASS_DECL_SINKVARIABLEASNAMEDARRAYPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

class SinkVariableAsNamedArrayPass
    : public impl::SinkVariableAsNamedArrayPassBase<
          SinkVariableAsNamedArrayPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(&getContext());

    absl::flat_hash_map<std::string, VariableConfig> variable_config_by_name;
    llvm::SmallDenseMap<mlir::TF::IfrtCallOp, IfrtArgConfigList>
        ifrt_call_argument_configs;

    // First, we backtrack from IFRT call to collect variable tensors that needs
    // to converted to loaded ifrt arrays and their associated information such
    // as their name and defining ops.
    std::vector<mlir::TF::IfrtCallOp> ifrt_call_ops;
    module.walk([&ifrt_call_ops](mlir::TF::IfrtCallOp call) {
      ifrt_call_ops.push_back(call);
    });
    for (const auto& call : ifrt_call_ops) {
      if (mlir::failed(CollectVariablesUsedByDevice(
              call, variable_config_by_name, ifrt_call_argument_configs))) {
        return signalPassFailure();
      }
    }

    // TODO(b/332906178): collapse the below with the
    // CollectVariablesUsedByDevice above or just remove the
    // CollectVariablesUsedByDevice.
    //
    // Rewrite ReadVariableOp with IfrtLoadVariableOp
    llvm::SmallDenseMap<mlir::TF::ReadVariableOp, mlir::TF::IfrtLoadVariableOp>
        read_to_load;

    mlir::WalkResult walk_result =
        module.walk([&](mlir::TF::ReadVariableOp read_variable_op) {
          mlir::FailureOr<std::string> variable_runtime_name =
              GetVariableTensorName(read_variable_op);
          if (mlir::failed(variable_runtime_name)) {
            read_variable_op->emitError() << "Failed to get variable name.";
            return mlir::WalkResult::interrupt();
          }

          builder.setInsertionPointAfter(read_variable_op);
          // TODO(b/319045348): consider use resource alias analysis for
          // this.
          auto var_handle = GetDefiningOp<mlir::TF::VarHandleOp>(
              read_variable_op.getResource());

          if (!var_handle) {
            read_variable_op->emitError(
                "ReadVariableOp has no defining VarHandleOp.");
            return mlir::WalkResult::interrupt();
          }

          auto iter = variable_config_by_name.find(*variable_runtime_name);
          mlir::StringAttr device_sharding_config_attr;
          if (iter == variable_config_by_name.end()) {
            device_sharding_config_attr = builder.getStringAttr("");
          } else {
            device_sharding_config_attr =
                builder.getStringAttr(iter->second.device_sharding_config);
          }

          std::vector<mlir::Type> result_types;
          result_types.push_back(mlir::RankedTensorType::get(
              {}, builder.getType<mlir::TF::StringType>()));
          result_types.push_back(read_variable_op.getResult().getType());

          auto load_variable_op = builder.create<mlir::TF::IfrtLoadVariableOp>(
              read_variable_op->getLoc(), result_types, var_handle.getResult(),
              device_sharding_config_attr,
              builder.getStringAttr(*variable_runtime_name));
          read_to_load[read_variable_op] = load_variable_op;

          return mlir::WalkResult::advance();
        });

    if (walk_result.wasInterrupted()) {
      return signalPassFailure();
    }

    // Rewrite ifrt call: variable tensors are sunk as attribute.
    // The runtime guarantees the binding of corresponding loaded ifrt array
    // based on attributes.
    for (auto& call : ifrt_call_ops) {
      if (!call.getVariableArgIndicesAttr().empty()) {
        call->emitError() << "Expect empty "
                          << call.getVariableArgIndicesAttrName().str()
                          << " attributes, but got "
                          << call.getVariableArgIndicesAttr().size()
                          << " elements";
        return signalPassFailure();
      }
      if (call->getOpOperands().size() !=
          ifrt_call_argument_configs[call].size()) {
        call->emitError() << "IfrtCallOp got " << call->getOpOperands().size()
                          << " operands, but expects "
                          << ifrt_call_argument_configs[call].size();
        return signalPassFailure();
      }
      llvm::SmallVector<int> variable_arg_indices;
      llvm::SmallVector<mlir::Attribute> variable_arg_names;
      llvm::SmallVector<mlir::Value> updated_args;

      for (const auto& [arg_idx, arg] :
           llvm::enumerate(ifrt_call_argument_configs[call])) {
        if (arg.is_variable) {
          variable_arg_names.push_back(
              builder.getStringAttr(arg.variable_name));
          variable_arg_indices.push_back(arg_idx);
          // Variable use the key from IfrtLoadVariable.
          updated_args.push_back(
              read_to_load[arg.read_variable_op].getArrayKey());
        } else {
          // non variable
          updated_args.push_back(call->getOperand(arg_idx));
        }
      }

      builder.setInsertionPointAfter(call);
      auto updated_ifrt_call = builder.create<mlir::TF::IfrtCallOp>(
          call->getLoc(), call.getResultTypes(), updated_args);

      updated_ifrt_call->setAttrs(call->getAttrs());
      // Update variable_arg_indices attribute.
      updated_ifrt_call.setVariableArgIndicesAttr(
          builder.getI32ArrayAttr(variable_arg_indices));

      call.replaceAllUsesWith(updated_ifrt_call);
      call.erase();
    }

    // Remove all ReadVariableOp after replacing the CPU usage of
    // ReadVariableOp.
    module.walk([&](mlir::TF::ReadVariableOp read_variable_op) {
      if (!read_variable_op->use_empty()) {
        // This variable tensor is used by CPU host.
        read_to_load[read_variable_op].setUsedByHost(true);

        // Replace CPU use of ReadVariableOp
        read_variable_op.replaceAllUsesWith(
            read_to_load[read_variable_op].getTensorFuture());
      }
      read_variable_op.erase();
    });
  }

 private:
  struct VariableConfig {
    // VariableDeviceShardingConfig text proto.
    std::string device_sharding_config;
    // All ReadVariableOps that returns this named variable.
    std::vector<mlir::TF::ReadVariableOp> read_variable_op;
  };
  struct IfrtArgConfig {
    bool is_variable;
    std::string variable_name;
    mlir::TF::ReadVariableOp read_variable_op;
  };
  using IfrtArgConfigList = llvm::SmallVector<IfrtArgConfig>;

  // Find defining ReadVariableOps and also build argument configuration map of
  // a IfrtCallOp.
  mlir::LogicalResult CollectVariablesUsedByDevice(
      mlir::TF::IfrtCallOp call,
      absl::flat_hash_map<std::string, VariableConfig>& variable_config_by_name,
      llvm::SmallDenseMap<mlir::TF::IfrtCallOp, IfrtArgConfigList>&
          ifrt_call_argument_configs) {
    IfrtArgConfigList& args = ifrt_call_argument_configs[call];

    tensorflow::tpu::TPUCompileMetadataProto metadata;

    // TODO(b/319045348):  remove the usage kMetadataAttrName.
    auto metadata_attr =
        call->getAttrOfType<mlir::StringAttr>(kMetadataTextAttrName);
    if (metadata_attr && !metadata_attr.empty()) {
      if (!tensorflow::protobuf::TextFormat::ParseFromString(
              metadata_attr.getValue().str(), &metadata)) {
        return call.emitError()
               << "Failed to parse TPUCompileMetadataProto from attr :"
               << metadata_attr.getValue().str();
      }
    } else {
      return call.emitError()
             << "Failed to Get TPUCompileMetadataProto from attr";
    }

    for (const auto& [arg_idx, input] : llvm::enumerate(call->getOperands())) {
      // Assuming the nested function calls are inlined.
      if (auto read_variable_op =
              GetDefiningOp<mlir::TF::ReadVariableOp>(input)) {
        mlir::FailureOr<std::string> variable_tensor_name =
            GetVariableTensorName(read_variable_op);

        if (mlir::failed(variable_tensor_name)) {
          return mlir::failure();
        }

        absl::StatusOr<std::string> device_sharding_config =
            GetVariableShardingConfig(metadata, arg_idx);
        if (!device_sharding_config.ok()) {
          return call->emitError()
                 << "Fail to get device sharding config for argument index "
                 << arg_idx;
        }
        VariableConfig& variable_config =
            variable_config_by_name[*variable_tensor_name];
        if (!variable_config.read_variable_op.empty()) {
          if (variable_config.device_sharding_config !=
              *device_sharding_config) {
            return call->emitError()
                   << "A variable tensor has different sharding config: "
                   << variable_config.device_sharding_config << " vs "
                   << *device_sharding_config;
          }
        } else {
          variable_config.device_sharding_config = *device_sharding_config;
        }

        variable_config.read_variable_op.push_back(read_variable_op);
        args.push_back({.is_variable = true,
                        .variable_name = *variable_tensor_name,
                        .read_variable_op = read_variable_op});
      } else {
        args.push_back({.is_variable = false});
      }
    }

    return mlir::success();
  }

  // The returned variable tensor name is used both as an internal hash key,
  // and as the binding name between the tensor and the array in the
  // runtime.
  std::string GetVariableTensorName(mlir::TF::VarHandleOp var_handle) {
    return absl::StrCat(absl::string_view(var_handle.getContainer()), "__",
                        absl::string_view(var_handle.getSharedName()));
  }

  mlir::FailureOr<std::string> GetVariableTensorName(
      mlir::TF::ReadVariableOp read_variable_op) {
    mlir::Value variable_definition = read_variable_op.getResource();
    auto var_handle = GetDefiningOp<mlir::TF::VarHandleOp>(variable_definition);

    if (!var_handle) {
      return read_variable_op->emitError("ReadVariableOp has no defining op.");
    }

    return GetVariableTensorName(var_handle);
  }

  absl::StatusOr<std::string> GetVariableShardingConfig(
      const tensorflow::tpu::TPUCompileMetadataProto& metadata, int arg_idx) {
    tensorflow::ifrt_serving::VariableDeviceShardingConfigProto
        device_sharding_config;
    std::vector<int> device_ids;

    if (metadata.has_device_assignment()) {
      absl::StatusOr<std::unique_ptr<xla::DeviceAssignment>> da =
          xla::DeviceAssignment::Deserialize(metadata.device_assignment());

      if (!da.ok()) {
        return da.status();
      }
      if (metadata.num_replicas() != (*da)->replica_count() ||
          metadata.num_cores_per_replica() != (*da)->computation_count()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Device assignment has different replica count: ",
            metadata.num_replicas(), " vs ", (*da)->replica_count(),
            " or computation count: ", metadata.num_cores_per_replica(), " vs ",
            (*da)->computation_count(), "."));
      }

      device_ids.reserve(metadata.num_replicas() *
                         metadata.num_cores_per_replica());
      for (int i = 0; i < (*da)->replica_count(); ++i) {
        for (int j = 0; j < (*da)->computation_count(); ++j) {
          device_ids.push_back((**da)(i, j));
        }
      }
    } else {
      // Default use first N devices.
      device_ids.resize(metadata.num_replicas() *
                        metadata.num_cores_per_replica());
      std::iota(device_ids.begin(), device_ids.end(), 0);
    }

    device_sharding_config.mutable_device_ids()->Assign(device_ids.begin(),
                                                        device_ids.end());

    if (metadata.args_size() > 0) {
      *device_sharding_config.mutable_sharding() =
          metadata.args(arg_idx).sharding();
    }

    std::string proto_text;
    tsl::protobuf::TextFormat::Printer printer;
    printer.SetSingleLineMode(true);
    printer.PrintToString(device_sharding_config, &proto_text);

    return proto_text;
  }

  template <typename OpT>
  OpT GetDefiningOp(const mlir::Value& value) {
    mlir::Operation* op = value.getDefiningOp();

    while (op && !llvm::isa<OpT>(op)) {
      if (llvm::isa<mlir::TF::IdentityOp>(op)) {
        op = op->getOperand(0).getDefiningOp();
      } else {
        return nullptr;
      }
    }

    if (op != nullptr) {
      return llvm::dyn_cast<OpT>(op);
    } else {
      return nullptr;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateSinkVariableAsNamedArrayPass() {
  return std::make_unique<SinkVariableAsNamedArrayPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
