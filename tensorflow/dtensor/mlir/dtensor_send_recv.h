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

#ifndef TENSORFLOW_DTENSOR_MLIR_DTENSOR_SEND_RECV_H_
#define TENSORFLOW_DTENSOR_MLIR_DTENSOR_SEND_RECV_H_

#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"

namespace tensorflow {
namespace dtensor {

// Given DTensorSend or DTensorRecv op, returns the corresponding DTensorRecv
// or DTensorSend op with the same key.
template <typename DTensorOp>
StatusOr<mlir::Operation*> GetCorrespondingDTensorSendRecvOp(
    mlir::ModuleOp module, DTensorOp dtensor_op) {
  mlir::Operation* corresponding_op = nullptr;
  if (std::is_same<DTensorOp, mlir::TF::DTensorSend>::value) {
    module.walk([&](mlir::Operation* op) {
      if (auto xla_recv_tpu = llvm::dyn_cast<mlir::TF::XlaRecvFromHostOp>(op)) {
        if (dtensor_op.key() == xla_recv_tpu.key()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto xla_recv_cpu =
                     llvm::dyn_cast<mlir::TF::_XlaRecvAtHostV2Op>(op)) {
        if (dtensor_op.key() == xla_recv_cpu.key()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto dtensor_recv =
                     llvm::dyn_cast<mlir::TF::DTensorRecv>(op)) {
        if (dtensor_op.key() == dtensor_recv.key()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto host_recv = llvm::dyn_cast<mlir::TF::_HostRecvOp>(op)) {
        if (dtensor_op.key() == host_recv.tensor_name()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      }
      return mlir::WalkResult::advance();
    });
  } else {
    const bool is_recv = std::is_same<DTensorOp, mlir::TF::DTensorRecv>::value;
    if (!is_recv) {
      return errors::Internal(
          "Error checking if is same for DTensorOp and DTensorRecv.");
    }
    module.walk([&](mlir::Operation* op) {
      if (auto xla_send_tpu = llvm::dyn_cast<mlir::TF::XlaSendToHostOp>(op)) {
        if (dtensor_op.key() == xla_send_tpu.key()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto xla_send_cpu =
                     llvm::dyn_cast<mlir::TF::_XlaSendFromHostV2Op>(op)) {
        if (dtensor_op.key() == xla_send_cpu.key()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto dtensor_send =
                     llvm::dyn_cast<mlir::TF::DTensorSend>(op)) {
        if (dtensor_op.key() == dtensor_send.key()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      } else if (auto host_send = llvm::dyn_cast<mlir::TF::_HostSendOp>(op)) {
        if (dtensor_op.key() == host_send.tensor_name()) {
          corresponding_op = op;
          return mlir::WalkResult::interrupt();
        }
      }
      return mlir::WalkResult::advance();
    });
  }

  if (!corresponding_op)
    return errors::InvalidArgument(
        "DTensorSend/DTensorRecv op must have corresponding "
        "DTensorRecv/DTensorSend op.");

  return corresponding_op;
}

// Lowers DTensorRecv op to either one of XlaRecvAtHost or XlaRecvFromHost,
// depending on src mesh cluster configuration.
StatusOr<mlir::Operation*> LowerDTensorRecvToXlaOp(
    mlir::TF::DTensorRecv dtensor_recv);

// Lowers DTensorRecv op to either one of XlaRecvAtHost or XlaRecvFromHost,
// depending on src mesh cluster configuration. `output_type` can be set to the
// specific local tensor type needed, if different from the Recv op output type.
StatusOr<mlir::Operation*> LowerDTensorRecvToXlaOp(
    mlir::TF::DTensorRecv dtensor_recv, mlir::Type output_type);

// Lowers DTensorSend Op to either one of XlaSendFromHost op or XlaSendToHost,
// depending on the src mesh cluster. `send_from_device_zero` should be set if
// control flow needs to be inserted to gather data onto and only sent from the
// zero'th device.
StatusOr<mlir::Operation*> LowerDTensorSendToXlaOp(
    const Layout& send_input_layout, mlir::Value send_input,
    mlir::TF::DTensorSend dtensor_send, bool send_from_device_zero);

// Lowers DTensorSend Op to a TF HostSend op.
StatusOr<mlir::Operation*> LowerDTensorSendFromCPUToTFOp(
    const Layout& send_input_layout, mlir::Value send_input,
    mlir::TF::DTensorSend dtensor_send);

// Lowers DTensorSend Op to a TF HostRecv op.
StatusOr<mlir::Operation*> LowerDTensorRecvFromCPUToTFOp(
    const Mesh& send_mesh, mlir::TF::DTensorRecv dtensor_recv);

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_DTENSOR_SEND_RECV_H_
