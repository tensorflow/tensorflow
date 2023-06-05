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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_send_recv.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORLOWERSENDRECV
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"

// Find all DTesorSend/Recv ops and lower into TF/XLA Send/Recv operations with
// execution kernels.
mlir::LogicalResult LowerDTensorSendRecvsOps(mlir::ModuleOp module) {
  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::TF::DTensorSend send_op) {
    if (mlir::failed(result)) return;

    auto recv_op = GetCorrespondingDTensorSendRecvOp<mlir::TF::DTensorSend>(
        module, send_op);
    if (!recv_op.ok()) {
      result = send_op.emitOpError(recv_op.status().message());
      return;
    }

    auto status = LowerDTensorSendAndRecv(send_op, *recv_op);
    if (!status.ok()) {
      result = send_op->emitOpError(status.status().message());
      return;
    }
  });
  return result;
}

// Adds Identity Op that uses device_id argument as inputs for clusters that
// does not have device id usages. When send/recv operations exists in
// tf_device.Clusters to transfer data across mesh clusters, device_id argument
// is required. However, mlir::func::FuncOp's created by transforming
// tf_device.Cluster to tf_device.ClusterFunc during ClusterOutlining pass will
// **not** include device_id as input argument if there are no usages within the
// cluster op body. As so, add Identity op that uses device_id argument from
// main function in all tf_device.Clusters so that device_id argument can be
// retained when converting tf_device.Cluster to functions.
void PropagateDeviceIdToClusters(mlir::ModuleOp module) {
  mlir::WalkResult result = module.walk([&](mlir::Operation* op) {
    if (llvm::isa<mlir::TF::_XlaSendFromHostOp, mlir::TF::_XlaRecvAtHostV2Op,
                  mlir::TF::XlaSendToHostOp, mlir::TF::XlaRecvFromHostOp,
                  mlir::TF::_HostSendOp, mlir::TF::_HostRecvOp,
                  mlir::TF::SendOp, mlir::TF::RecvOp>(op))
      return mlir::WalkResult::interrupt();
    return mlir::WalkResult::advance();
  });

  const bool has_cross_mesh_send_recv = result.wasInterrupted();
  if (!has_cross_mesh_send_recv) return;

  mlir::func::FuncOp main_func =
      module.lookupSymbol<mlir::func::FuncOp>("main");
  auto device_id = DeviceId(main_func);

  module.walk([&](mlir::tf_device::ClusterOp op) {
    mlir::OpBuilder builder(&op.GetBody().front());
    builder.create<mlir::TF::IdentityOp>(main_func.getLoc(),
                                         device_id->getType(), *device_id);
  });
}

// Pass that merges multiple tf_device.Cluster ops for multi-mesh computation
// into a single cluster. After this pass, exactly one tf_device.Cluster op
// exists for each device mesh.
struct DTensorLowerSendRecv
    : public impl::DTensorLowerSendRecvBase<DTensorLowerSendRecv> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder op_builder(&context);
    auto module = getOperation();

    // Merging clusters and decomposing control flow may have created new
    // DTensorSend/DTensorRecv ops. Lower DTensorSend/DTensorRecv ops added by
    // above transformations.
    if (mlir::failed(LowerDTensorSendRecvsOps(module)))
      return signalPassFailure();

    // Ensure that all mesh clusters have at least one usages of device_id
    // argument from main function to guarantee that device_id argument is
    // retained after ClusterOutlinging.
    PropagateDeviceIdToClusters(module);
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorLowerSendRecv() {
  return std::make_unique<DTensorLowerSendRecv>();
}

}  // namespace dtensor
}  // namespace tensorflow
