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

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
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
#include "mlir/Support/WalkResult.h"  // from @llvm-project
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
  explicit RewriteClusterToIfrtCallPass(bool arg_enable_async_ifrt = false) {
    this->enable_async_ifrt = arg_enable_async_ifrt;
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::SymbolTable symbol_table(module);
    mlir::SymbolTableCollection symbol_table_collection;

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
      Rewrite(symbol_table, symbol_table_collection, cluster_to_ifrt_program,
              cluster_func, devices);
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

    return mlir::TFTPU::SetMetadataProtoFromClusterFuncOp(
        cluster_func, num_replicas, num_cores_per_replica,
        /*xla_device_assignment=*/std::nullopt, metadata);
  }

  // Returns true if the function or any of its callees recursively contains
  // tf.PwStreamResultsOp.
  //
  // PwStreamResultsOp immediately streams results back to the serving
  // controller. We must disable async execution (falling back to synchronous
  // IfrtCallOp) if this op is present. Otherwise, the async call might finish
  // without properly synchronizing or awaiting the streaming results, causing
  // race conditions or broken serving behavior (see b/500390771).
  //
  // Complexity Note: This method must recursively traverse control flow ops
  // (e.g., tf.If, tf.While) and partitioned calls. Since these ops frequently
  // reference callee functions via an ArrayAttr of symbols rather than a simple
  // SymbolRefAttr, we explicitly iterate through ArrayAttr elements to avoid
  // missing streaming ops hidden inside branches.
  bool ContainsStreamingOp(mlir::func::FuncOp func,
                           mlir::SymbolTableCollection& symbol_table_collection,
                           llvm::DenseSet<mlir::func::FuncOp>& visited) {
    // Track visited functions to safely handle recursive call graphs and
    // prevent infinite loops.
    if (!func || !visited.insert(func).second) return false;

    VLOG(1) << "Checking function for streaming ops: " << func.getName().str();

    // Walk all operations within the current function's body.
    auto walk_result = func.walk([&](mlir::Operation* op) {
      // Base case - If we find the streaming op directly, interrupt the
      // walk and signal discovery.
      if (mlir::isa<mlir::TF::PwStreamResultsOp>(op)) {
        VLOG(1) << "Found tf.PwStreamResults in " << func.getName().str();
        return mlir::WalkResult::interrupt();
      }

      // Recursive case - Many ops (like tf.If, tf.While, tf.PartitionedCall)
      // don't hold operations directly but instead reference callee functions
      // via attributes. We need to inspect these attributes to traverse the
      // call graph.
      for (const mlir::NamedAttribute& attr : op->getAttrs()) {
        if (auto sym_ref =
                mlir::dyn_cast<mlir::SymbolRefAttr>(attr.getValue())) {
          // Handle single function references (e.g., tf.PartitionedCall).
          // Resolve the symbol to the actual function and recursively check it.
          VLOG(1) << "Looking up symbol: " << sym_ref.getLeafReference().str()
                  << " from op: " << op->getName().getStringRef().str();
          if (auto callee = symbol_table_collection
                                .lookupNearestSymbolFrom<mlir::func::FuncOp>(
                                    op, sym_ref)) {
            if (ContainsStreamingOp(callee, symbol_table_collection, visited)) {
              return mlir::WalkResult::interrupt();
            }
          }
        } else if (auto arr_attr =
                       mlir::dyn_cast<mlir::ArrayAttr>(attr.getValue())) {
          // Handle arrays of function references (e.g., branches in tf.If).
          // We must check every referenced function in the array.
          for (auto elt : arr_attr.getValue()) {
            if (auto sym_ref = mlir::dyn_cast<mlir::SymbolRefAttr>(elt)) {
              VLOG(1) << "Looking up symbol from array: "
                      << sym_ref.getLeafReference().str()
                      << " from op: " << op->getName().getStringRef().str();
              if (auto callee =
                      symbol_table_collection
                          .lookupNearestSymbolFrom<mlir::func::FuncOp>(
                              op, sym_ref)) {
                if (ContainsStreamingOp(callee, symbol_table_collection,
                                        visited)) {
                  return mlir::WalkResult::interrupt();
                }
              }
            }
          }
        }
      }
      return mlir::WalkResult::advance();
    });

    // If the walk was interrupted, it means we found the streaming op
    // somewhere in this function or its callees.
    return walk_result.wasInterrupted();
  }

  void Rewrite(mlir::SymbolTable& symbol_table,
               mlir::SymbolTableCollection& symbol_table_collection,
               llvm::DenseMap<mlir::func::FuncOp, mlir::func::FuncOp>&
                   cluster_to_ifrt_program,
               mlir::tf_device::ClusterFuncOp cluster_func,
               mlir::TF::RuntimeDevices& devices) {
    mlir::OpBuilder builder(cluster_func);
    mlir::FlatSymbolRefAttr callee_symbol = cluster_func.getFuncAttr();
    mlir::func::FuncOp callee_func =
        symbol_table.lookup<mlir::func::FuncOp>(callee_symbol.getValue());

    // TODO - b/500390771: Ideally we need to make the await op the preceding
    // op of the PwStreamResultsOp, which can be achieved in the tf-to-mlrt
    // lowering pass.
    bool use_async = enable_async_ifrt;
    if (use_async) {
      llvm::DenseSet<mlir::func::FuncOp> visited;
      if (ContainsStreamingOp(callee_func, symbol_table_collection, visited)) {
        cluster_func->emitRemark(
            "Disabling AsyncIfrtCallOp because tf.PwStreamResults is present "
            "in the cluster.");
        use_async = false;
      }
    }

    auto ifrt_program_name =
        absl::StrCat("_ifrt_program_", callee_func.getSymName().str());
    if (mlir::func::FuncOp ifrt_program =
            cluster_to_ifrt_program[callee_func]) {
      // ifrt program already exists
      builder.setInsertionPoint(cluster_func);

      mlir::Operation* ifrt_call_op;
      if (use_async) {
        ifrt_call_op = mlir::TF::AsyncIfrtCallOp::create(
            builder, cluster_func->getLoc(), cluster_func.getResultTypes(),
            cluster_func->getOperands());
      } else {
        ifrt_call_op = mlir::TF::IfrtCallOp::create(
            builder, cluster_func->getLoc(), cluster_func.getResultTypes(),
            cluster_func->getOperands());
      }

      ifrt_call_op->setAttr(
          "operandSegmentSizes",
          builder.getDenseI32ArrayAttr(
              {static_cast<int32_t>(cluster_func.getNumOperands()), 0}));

      int64_t program_id;
      if (auto attr = ifrt_program->getAttrOfType<mlir::IntegerAttr>(
              "tfrt_ifrt_serving.program_id")) {
        program_id = attr.getInt();
      } else {
        return signalPassFailure();
      }

      auto metadata_attr =
          ifrt_program->getAttrOfType<mlir::StringAttr>(kMetadataTextAttrName);
      auto device_assignment_attr =
          ifrt_program->getAttrOfType<mlir::ArrayAttr>(kDeviceAssignmentAttr);
      if (!metadata_attr || !device_assignment_attr) {
        return signalPassFailure();
      }

      // For better debuggability, attach attributes such as
      // tpu_compile_metadata and device_assignment to IfrtCallOp.
      ifrt_call_op->setAttr(kMetadataTextAttrName, metadata_attr);
      ifrt_call_op->setAttr(kDeviceAssignmentAttr, device_assignment_attr);

      // TODO(b/304839793): populate variable names after adding a variable
      // hoisting pass.
      ifrt_call_op->setAttr("variable_arg_indices",
                            builder.getI32ArrayAttr({}));
      ifrt_call_op->setAttr("program_id",
                            builder.getI64IntegerAttr(program_id));

      cluster_func->replaceAllUsesWith(ifrt_call_op->getResults());
      cluster_func->erase();

      return;
    }
    mlir::OpBuilder::InsertionGuard insertion_guard(builder);
    builder.setInsertionPoint(callee_func);

    mlir::func::FuncOp cloned_ifrt_program = mlir::func::FuncOp::create(
        builder, callee_func->getLoc(), ifrt_program_name,
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

    auto device_assignment_attr =
        cluster_func->getAttrOfType<mlir::ArrayAttr>(kDeviceAssignmentAttr);
    if (!device_assignment_attr) {
      device_assignment_attr = builder.getArrayAttr({});
    }
    cloned_ifrt_program->setAttr(kDeviceAssignmentAttr, device_assignment_attr);

    cloned_ifrt_program.setName(ifrt_program_name);

    int64_t program_id = NewProgramId();
    cloned_ifrt_program->setAttr("tfrt_ifrt_serving.program_id",
                                 builder.getI64IntegerAttr(program_id));

    // Make cloned ifrt program public so that it does not get dropped by
    // inliner.
    cloned_ifrt_program.setPublic();

    builder.setInsertionPoint(cluster_func);

    mlir::Operation* ifrt_call_op;
    if (use_async) {
      ifrt_call_op = mlir::TF::AsyncIfrtCallOp::create(
          builder, cluster_func->getLoc(), cluster_func.getResultTypes(),
          cluster_func->getOperands());
    } else {
      ifrt_call_op = mlir::TF::IfrtCallOp::create(
          builder, cluster_func->getLoc(), cluster_func.getResultTypes(),
          cluster_func->getOperands());
    }

    ifrt_call_op->setAttr(
        "operandSegmentSizes",
        builder.getDenseI32ArrayAttr(
            {static_cast<int32_t>(cluster_func.getNumOperands()), 0}));

    // TODO(b/304839793): populate variable names after adding a variable
    // hoisting pass.
    ifrt_call_op->setAttr("variable_arg_indices", builder.getI32ArrayAttr({}));
    ifrt_call_op->setAttr("program_id", builder.getI64IntegerAttr(program_id));
    // For better debuggability, attach attributes such as tpu_compile_metadata
    // and device_assignment to IfrtCallOp.
    ifrt_call_op->setAttr(kMetadataTextAttrName,
                          builder.getStringAttr(serialized_metadata));
    ifrt_call_op->setAttr(kDeviceAssignmentAttr, device_assignment_attr);

    cluster_func->replaceAllUsesWith(ifrt_call_op->getResults());
    cluster_func->erase();

    symbol_table.insert(cloned_ifrt_program);
    cluster_to_ifrt_program[callee_func] = cloned_ifrt_program;
  }
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateRewriteClusterToIfrtCallPass(bool enable_async_ifrt) {
  return std::make_unique<RewriteClusterToIfrtCallPass>(enable_async_ifrt);
}

}  // namespace ifrt_serving
}  // namespace tensorflow
