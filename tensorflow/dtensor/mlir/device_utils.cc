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

#include "tensorflow/dtensor/mlir/device_utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace dtensor {

// Returns an MLIR value representing the current device ID.
StatusOr<mlir::Value> DeviceId(mlir::Operation* op) {
  mlir::func::FuncOp function = llvm::dyn_cast<mlir::func::FuncOp>(op);
  if (!function) {
    // Device ID is the 0th argument of the enclosing function.
    function = op->getParentOfType<mlir::func::FuncOp>();
    if (!function)
      return errors::InvalidArgument(
          "operation must be enclosed inside a function.");
  }

  if (function.getNumArguments() == 0)
    return errors::InvalidArgument(
        "enclosing function must contain device id as argument");

  auto device_id = function.getArgument(0);
  auto device_id_type = device_id.getType().dyn_cast<mlir::RankedTensorType>();
  if (!device_id_type ||
      !device_id_type.getElementType().isa<mlir::IntegerType>())
    return errors::InvalidArgument(
        "0-th argument of the enclosing function should be integer device id.");

  return device_id;
}

StatusOr<mlir::Value> DeviceId(mlir::Value val) {
  if (auto block_arg = val.dyn_cast<mlir::BlockArgument>()) {
    auto device_id = block_arg.getOwner()->getArgument(0);
    auto device_id_type =
        device_id.getType().dyn_cast<mlir::RankedTensorType>();
    if (!device_id_type ||
        !device_id_type.getElementType().isa<mlir::IntegerType>())
      return errors::InvalidArgument(
          "0-th argument of the enclosing block should be integer device id.");
    return device_id;
  }
  return DeviceId(val.getDefiningOp());
}

}  // namespace dtensor
}  // namespace tensorflow
