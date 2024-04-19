/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/tpu_metadata_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_constants.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tsl/platform/protobuf.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DECL_REWRITECLUSTERTOIFRTCALLPASS
#define GEN_PASS_DEF_REWRITECLUSTERTOIFRTCALLPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

// A pass that inserts tf.ifrt_call and create its callee as a Ifrt
// Program.
class RewriteClusterToIfrtCallPass
    : public impl::RewriteClusterToIfrtCallPassBase<
          RewriteClusterToIfrtCallPass> {
 public:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTable symbol_table(module);

    mlir::TF::RuntimeDevices devices;
    if (failed(tensorflow::GetDevicesFromOp(module.getOperation(), &devices)))
      return signalPassFailure();

    // key: original callee function in tf_device.cluster_func. value: ifrt
    // program.
    llvm::DenseMap<mlir::func::FuncOp, mlir::func::FuncOp>
        cluster_to_ifrt_program;

    std::vector<mlir::tf_device::ClusterFuncOp> cluster_func_ops;
    module.walk([&](mlir::tf_device::ClusterFuncOp cluster_func) {
      cluster_func_ops.push_back(cluster_func);
    });
    for (auto cluster_func : cluster_func_ops) {
      Rewrite(symbol_table, cluster_to_ifrt_program, cluster_func, devices);
    }

    // TODO(b/304839793): Move this to a separate pass. The old remove
    // compilation result pass rely on TPUPartitionedCall
    llvm::SmallVector<mlir::TF::TPUCompilationResultOp> compilation_result_ops;
    module.walk([&](mlir::TF::TPUCompilationResultOp op) {
      compilation_result_ops.push_back(op);
    });
    for (auto op : compilation_result_ops) {
      if (!op.use_empty()) {
        module->emitError("TPUCompilationResultOp is under use");
        return signalPassFailure();
      }
      op.erase();
    }
  }

 private:
  // Returns a new unique program id.
  static int64_t NewProgramId() {
    const uint64_t id = static_cast<int64_t>(tensorflow::random::New64());
    // We use a signed int for program ids since TensorFlow doesn't
    // support uint64_t attributes.
    return absl::bit_cast<int64_t>(id);
  }

  mlir::LogicalResult GetTpuCompileMetadata(
      mlir::tf_device::ClusterFuncOp cluster_func,
      mlir::TF::RuntimeDevices &devices,
      tensorflow::tpu::TPUCompileMetadataProto *metadata) {
    // Collect `num_replicas` and `num_cores_per_replica` attributes.
    int num_replicas = 1;
    mlir::tf_device::ReplicateOp replicate =
        cluster_func->getParentOfType<mlir::tf_device::ReplicateOp>();
    if (replicate) num_replicas = replicate.getN();

    auto num_cores_per_replica_attr =
        cluster_func->getAttrOfType<mlir::IntegerAttr>(
            tensorflow::kNumCoresPerReplicaAttr);
    if (!num_cores_per_replica_attr)
      return cluster_func.emitOpError()
             << "Attribute" << tensorflow::kNumCoresPerReplicaAttr
             << " is missing";
    int num_cores_per_replica = num_cores_per_replica_attr.getInt();

    std::optional<xla::DeviceAssignmentProto> xla_device_assignment;
    auto topology_attr = cluster_func->getAttrOfType<mlir::StringAttr>(
        tensorflow::kTopologyAttr);
    // Get device assignment.
    auto device_assignment_attr = cluster_func->getAttrOfType<mlir::ArrayAttr>(
        tensorflow::kDeviceAssignmentAttr);
    if (topology_attr && device_assignment_attr && !topology_attr.empty() &&
        !device_assignment_attr.empty()) {
      auto device_coordinates =
          tensorflow::GetDeviceCoordinates(device_assignment_attr);
      if (!device_coordinates.ok())
        return cluster_func.emitError()
               << "error in parsing tpu device coordinates: "
               << device_coordinates.status().message();

      auto device_assignment = tensorflow::GetTPUCompilationAndExecutionDevices(
          devices.device_names(), num_replicas, num_cores_per_replica,
          topology_attr.getValue(), *device_coordinates);
      if (!device_assignment.ok())
        return cluster_func.emitError()
               << "error in parsing TPU compilation/execution devices: "
               << device_assignment.status().message();
      if (!device_assignment->xla_device_assignment) {
        return cluster_func.emitError()
               << "Unexpected empty xla_device_assignment";
      }
      xla_device_assignment = device_assignment->xla_device_assignment;
    }

    return mlir::TFTPU::SetMetadataProtoFromClusterFuncOp(
        cluster_func, num_replicas, num_cores_per_replica,
        std::move(xla_device_assignment), metadata);
  }

  void Rewrite(mlir::SymbolTable &symbol_table,
               llvm::DenseMap<mlir::func::FuncOp, mlir::func::FuncOp>
                   &cluster_to_ifrt_program,
               mlir::tf_device::ClusterFuncOp cluster_func,
               mlir::TF::RuntimeDevices &devices) {
    mlir::OpBuilder builder(cluster_func);
    mlir::FlatSymbolRefAttr callee_symbol = cluster_func.getFuncAttr();
    mlir::func::FuncOp callee_func =
        symbol_table.lookup<mlir::func::FuncOp>(callee_symbol.getValue());

    auto ifrt_program_name =
        absl::StrCat("_ifrt_program_", callee_func.getSymName().str());
    if (mlir::func::FuncOp ifrt_program =
            cluster_to_ifrt_program[callee_func]) {
      // ifrt program already exists
      builder.setInsertionPoint(cluster_func);

      mlir::TF::IfrtCallOp ifrt_call_op = builder.create<mlir::TF::IfrtCallOp>(
          cluster_func->getLoc(), cluster_func.getResultTypes(),
          cluster_func->getOperands());

      int64_t program_id;
      if (auto attr = ifrt_program->getAttrOfType<mlir::IntegerAttr>(
              "tfrt_ifrt_serving.program_id")) {
        program_id = attr.getInt();
      } else {
        return signalPassFailure();
      }

      auto metadata_attr =
          ifrt_program->getAttrOfType<mlir::StringAttr>(kMetadataTextAttrName);
      if (!metadata_attr) {
        return signalPassFailure();
      }
      ifrt_call_op->setAttr(kMetadataTextAttrName, metadata_attr);

      // TODO(b/304839793): populate variable names after adding a variable
      // hoisting pass.
      ifrt_call_op.setVariableArgIndicesAttr(builder.getI32ArrayAttr({}));
      ifrt_call_op.setProgramId(program_id);

      cluster_func->replaceAllUsesWith(ifrt_call_op.getResults());
      cluster_func->erase();

      return;
    }
    mlir::OpBuilder::InsertionGuard insertion_guard(builder);
    builder.setInsertionPoint(callee_func);

    mlir::func::FuncOp cloned_ifrt_program = builder.create<mlir::func::FuncOp>(
        callee_func->getLoc(), ifrt_program_name,
        callee_func.getFunctionType());
    mlir::IRMapping mapper;
    callee_func.cloneInto(cloned_ifrt_program, mapper);

    tensorflow::tpu::TPUCompileMetadataProto metadata;
    if (mlir::failed(GetTpuCompileMetadata(cluster_func, devices, &metadata))) {
      return signalPassFailure();
    }
    std::string serialized_metadata;
    tsl::protobuf::TextFormat::Printer printer;
    printer.SetSingleLineMode(true);
    printer.PrintToString(metadata, &serialized_metadata);

    cloned_ifrt_program->setAttr(kMetadataTextAttrName,
                                 builder.getStringAttr(serialized_metadata));

    cloned_ifrt_program.setName(ifrt_program_name);

    int64_t program_id = NewProgramId();
    cloned_ifrt_program->setAttr("tfrt_ifrt_serving.program_id",
                                 builder.getI64IntegerAttr(program_id));

    // Make clonet ifrt program public so that it does not get dropped by
    // inliner.
    cloned_ifrt_program.setPublic();

    builder.setInsertionPoint(cluster_func);

    mlir::TF::IfrtCallOp ifrt_call_op = builder.create<mlir::TF::IfrtCallOp>(
        cluster_func->getLoc(), cluster_func.getResultTypes(),
        cluster_func->getOperands());

    // TODO(b/304839793): populate variable names after adding a variable
    // hoisting pass.
    ifrt_call_op.setVariableArgIndicesAttr(builder.getI32ArrayAttr({}));
    ifrt_call_op.setProgramId(program_id);
    // Additionally attach tpu_compile_metadata to IfrtCallOp. Some subsequent
    // pass such as SinkVariableAsNamedArrayPass relies on this attribute.
    ifrt_call_op->setAttr(kMetadataTextAttrName,
                          builder.getStringAttr(serialized_metadata));

    cluster_func->replaceAllUsesWith(ifrt_call_op.getResults());
    cluster_func->erase();

    symbol_table.insert(cloned_ifrt_program);
    cluster_to_ifrt_program[callee_func] = cloned_ifrt_program;
  }
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateRewriteClusterToIfrtCallPass() {
  return std::make_unique<RewriteClusterToIfrtCallPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
