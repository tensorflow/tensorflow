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

#include <string>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/cc/xla_spmd/layout_to_xla_sharding.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORUPDATETPUMETADATA
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

constexpr char kDeviceAttr[] = "device";
constexpr char kFuncDeviceAttr[] = "tf.device";

// By default, all TPUs are connected, construct a single replica group.
void ComputeReplicaGroupSplitInfo(int requested_num_replicas, int* num_replicas,
                                  int* core_id_local_offset) {
  *num_replicas = requested_num_replicas;
  *core_id_local_offset = 0;
}

// Removes explicit device assignment on TPUExecute and _TPUCompileMlir ops.
// As TPU execution replication logic is delegated to DTensorDevice,
// DTensorDevice should handle replication and Placer would assign devices.
void UpdateTPUDeviceAssignment(mlir::func::FuncOp function,
                               mlir::OpBuilder* builder) {
  function.walk([&](mlir::Operation* op) {
    if (!llvm::isa<
            mlir::TF::TPUExecuteOp, mlir::TF::TPUExecuteAndUpdateVariablesOp,
            mlir::TF::_TPUCompileMlirOp, mlir::TF::TPUCompileSucceededAssertOp>(
            op))
      return;

    assert(!op->getAttrOfType<mlir::StringAttr>(kDeviceAttr));

    auto enclosing_launch = op->getParentOfType<mlir::tf_device::LaunchOp>();
    if (!enclosing_launch) return;

    enclosing_launch.setDeviceAttr(builder->getStringAttr(""));

    // Remove placeholder device attributes of resource arguments to TPU
    // computation.
    for (int i = 0; i < function.getNumArguments(); ++i)
      function.removeArgAttr(i, builder->getStringAttr(kFuncDeviceAttr));
  });
}
absl::Status UpdateMetadataProtoXlaSpmd(const Mesh& mesh_config,
                                        mlir::TF::_TPUCompileMlirOp compile,
                                        tpu::TPUCompileMetadataProto& proto) {
  const int64_t num_devices = mesh_config.num_devices();
  int core_id_local_offset = 0;
  int num_replicas = mesh_config.num_devices();

  ComputeReplicaGroupSplitInfo(num_replicas, &num_replicas,
                               &core_id_local_offset);

  // DTensor will interact with Xla Spmd by setting 1 replica and
  // `num_devices` number of cores per that replica to ensure
  // Xla Spmd inserts the correct communicatives between operations
  // when doing Spmd.
  //
  // If we do num_replicas = `num_devices` instead, then Xla Spmd
  // will expect users to insert their own communicatives for cross replica
  // communicative.
  proto.set_num_replicas(1);
  proto.set_num_cores_per_replica(num_devices);
  proto.set_use_spmd_for_xla_partitioning(true);

  // Set metadata's input and output shardings. First extract the
  // TPUCompile op's attached module function.
  auto nested_module = mlir::parseSourceString<mlir::ModuleOp>(
      compile.getMlirModule().str(), compile.getContext());

  mlir::func::FuncOp main_tpu_func =
      nested_module->lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_tpu_func) {
    return errors::Internal(
        "Could not find function definition for "
        "tpu_func attached to TPUCompileOp.");
  }

  for (int arg_index = 0; arg_index < main_tpu_func.getNumArguments();
       ++arg_index) {
    mlir::StringAttr arg_sharding_attr =
        main_tpu_func.getArgAttrOfType<mlir::StringAttr>(arg_index,
                                                         kXlaShardingAttr);
    if (!arg_sharding_attr) {
      return errors::Internal("Expected sharding arg attr for input index: ",
                              std::to_string(arg_index));
    }
    proto.mutable_args(arg_index)->mutable_sharding()->ParseFromString(
        arg_sharding_attr.getValue().str());
  }

  for (int retval_index = 0; retval_index < main_tpu_func.getNumResults();
       ++retval_index) {
    mlir::StringAttr retval_sharding_attr =
        main_tpu_func.getResultAttrOfType<mlir::StringAttr>(retval_index,
                                                            kXlaShardingAttr);
    if (!retval_sharding_attr) {
      return errors::Internal("Expected sharding arg attr for output index: ",
                              std::to_string(retval_index));
    }

    proto.mutable_retvals(retval_index)
        ->mutable_sharding()
        ->ParseFromString(retval_sharding_attr.getValue().str());
  }

  // Finally, set the device assignment proto for the metadata proto.
  // This will just be increasing from 0 to N-1 where N is the num_devices in
  // the mesh.
  if (proto.has_device_assignment()) {
    // TODO(samuelslee) Support User specified device assignment.
    return errors::Unimplemented(
        "Xla Spmd for user specified device "
        "assignment is not supported yet.");
  }
  if (!Mesh::tpu_core_ids().empty()) {
    std::string mesh_name = mesh_config.name();
    if (Mesh::tpu_core_ids().count(mesh_name) == 0) {
      // This can happen only for manually created meshes (2 above) with
      // non-empty names. This mesh should use the default mapping.
      VLOG(1) << "mesh_name " << mesh_name << " not found, using empty name";
      mesh_name = "";
    }
    const std::vector<int>& tpu_core_ids = Mesh::tpu_core_ids()[mesh_name];
    VLOG(1) << "tpu_core_ids: " << absl::StrJoin(tpu_core_ids, ", ");

    xla::DeviceAssignmentProto device_assignment;
    device_assignment.set_replica_count(1);
    device_assignment.set_computation_count(num_devices);
    const int64_t start_device_id = mesh_config.min_global_device_id();
    for (int i = 0; i < num_devices; ++i) {
      auto* computation_device = device_assignment.add_computation_devices();
      int tpu_core_id_index = i + start_device_id + core_id_local_offset;
      computation_device->add_replica_device_ids(
          tpu_core_ids[tpu_core_id_index]);
    }
    *proto.mutable_device_assignment() = device_assignment;
  }
  return absl::OkStatus();
}

absl::Status UpdateMetadataProtoDtensorSpmd(
    const Mesh& mesh_config, tpu::TPUCompileMetadataProto& proto) {
  int core_id_local_offset = 0;
  int num_replicas = mesh_config.num_devices();

  ComputeReplicaGroupSplitInfo(num_replicas, &num_replicas,
                               &core_id_local_offset);

  proto.set_num_replicas(num_replicas);

  // We keep DTensor mesh global device IDs equal to XLA replica IDs, both
  // sequentially increasing over mesh dimensions. Collective lowering has
  // generated `replica_groups` using these IDs.
  //
  // We need to set appropriate XLA replica ID-to-core ID mappings here to get
  // correct results, by being consistent with what the user Python program
  // gets and assumes. There are three kinds of mesh:
  //
  // 1. The first mesh getting here is a one-of-a-kind mesh for merging core
  //    IDs across hosts during TPU initialization. This mesh doesn't need any
  //    mapping to be set. Mesh::tpu_core_ids() is empty when this happens.
  // 2. Users can manually create meshes, with empty or non-empty names. These
  //    meshes have global device IDs equal to TF task-device ordinals, and
  //    they do not place any entry in Mesh::tpu_core_ids(). The default entry
  //    in Mesh::tpu_core_ids(), stored under an empty name key by the mesh
  //    computation in 1, works on these meshes.
  // 3. Users can create ring reduction-optimized meshes using provided
  //    helpers. These meshes must have non-empty names and store an entry in
  //    Mesh::tpu_core_ids() when they are created, using their name as key.
  //
  // For any user-defined mesh, if users have manually specified device
  // assignment, always respect that.
  if (!Mesh::tpu_core_ids().empty() && !proto.has_device_assignment()) {
    std::string mesh_name = mesh_config.name();
    if (Mesh::tpu_core_ids().count(mesh_name) == 0) {
      // This can happen only for manually created meshes (2 above) with
      // non-empty names. This mesh should use the default mapping.
      VLOG(1) << "mesh_name " << mesh_name << " not found, using empty name";
      mesh_name = "";
    }
    const std::vector<int>& tpu_core_ids = Mesh::tpu_core_ids()[mesh_name];
    VLOG(1) << "tpu_core_ids: " << absl::StrJoin(tpu_core_ids, ", ");

    xla::DeviceAssignmentProto device_assignment;
    device_assignment.set_replica_count(num_replicas);
    device_assignment.set_computation_count(1);
    auto* computation_device = device_assignment.add_computation_devices();
    // TODO(b/188076080): Clean up device id.
    const int64_t start_device_id = mesh_config.min_global_device_id();
    for (int i = 0; i < num_replicas; ++i) {
      int tpu_core_id_index = i + start_device_id + core_id_local_offset;
      computation_device->add_replica_device_ids(
          tpu_core_ids[tpu_core_id_index]);
    }
    *proto.mutable_device_assignment() = device_assignment;
  }
  return absl::OkStatus();
}

mlir::LogicalResult UpdateTPUCompileMetadata(const Mesh& mesh_config,
                                             mlir::func::FuncOp function,
                                             mlir::OpBuilder* builder) {
  auto result = function.walk([&](mlir::TF::_TPUCompileMlirOp compile) {
    if (mesh_config.use_xla_spmd()) {
      // Create a new compile op with the appropriate new number of operands.
      builder->setInsertionPointAfter(compile);
      auto new_compile_op = builder->create<mlir::TF::_TPUCompileMlirOp>(
          compile.getLoc(), compile.getCompilationStatus().getType(),
          /*program=*/
          llvm::SmallVector<mlir::Type, 8>(
              mesh_config.num_devices(),
              mlir::RankedTensorType::get(
                  {3}, builder->getType<mlir::TF::StringType>())),
          compile.getDynamicShapes(), compile.getMlirModule(),
          compile.getMetadata());
      // Since num computations is equal to the `num_devices` in the mesh,
      // the new compile op produces `num_devices + 1` outputs.
      // That is, (compilation_status, num_devices programs). However,
      // in DTensor world, we only have one TPUExecute, so we only replace
      // the compilation status and the first program output.
      //
      // This is a hacky way of getting around XLA SPMD to work, ideally
      // the TF2XLA bridge integration with DTensor should resolve this as well.
      compile.getResult(0).replaceAllUsesWith(new_compile_op.getResult(0));
      compile.getResult(1).replaceAllUsesWith(new_compile_op.getResult(1));
      compile.erase();
      compile = new_compile_op;
    }

    tpu::TPUCompileMetadataProto metadata_proto;
    if (!metadata_proto.ParseFromString(compile.getMetadata().str())) {
      compile.emitOpError("unable to parse TPUCompileMetadata");
      return mlir::WalkResult::interrupt();
    }

    absl::Status status =
        mesh_config.use_xla_spmd()
            ? UpdateMetadataProtoXlaSpmd(mesh_config, compile, metadata_proto)
            : UpdateMetadataProtoDtensorSpmd(mesh_config, metadata_proto);

    if (!status.ok()) {
      compile.emitOpError(status.ToString());
      return mlir::WalkResult::interrupt();
    }

    compile.setMetadataAttr(
        builder->getStringAttr(metadata_proto.SerializeAsString()));
    return mlir::WalkResult::advance();
  });
  return mlir::failure(result.wasInterrupted());
}

// Pass that updates TPU specific metadata including `num_replicas` and device
// assignment of TPUCompileMlirOp and TPUExecute ops.
struct DTensorUpdateTPUMetadata
    : public impl::DTensorUpdateTPUMetadataBase<DTensorUpdateTPUMetadata> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);
    auto module = getOperation();
    mlir::func::FuncOp main_func =
        module.lookupSymbol<mlir::func::FuncOp>("main");
    if (!main_func) return;

    auto result = main_func.walk([&](mlir::TF::StatefulPartitionedCallOp op) {
      auto call_config = op.getConfig();
      auto mesh_or_status = Mesh::FromString(call_config.str());
      if (!mesh_or_status.ok()) return mlir::WalkResult::advance();

      const auto mesh = mesh_or_status.value();
      if (!mesh.is_tpu_mesh()) return mlir::WalkResult::advance();

      auto function = MaybeFindFunction(op);
      if (!function) {
        op.emitOpError(
            "Could not find function definition for "
            "StatefulPartitionedCall op running on TPU.");
        return mlir::WalkResult::interrupt();
      }

      if (mlir::failed(UpdateTPUCompileMetadata(mesh, *function, &builder)))
        return mlir::WalkResult::interrupt();

      UpdateTPUDeviceAssignment(*function, &builder);
      return mlir::WalkResult::advance();
    });

    if (result.wasInterrupted()) return signalPassFailure();
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorUpdateTPUMetadata() {
  return std::make_unique<DTensorUpdateTPUMetadata>();
}

}  // namespace dtensor
}  // namespace tensorflow
