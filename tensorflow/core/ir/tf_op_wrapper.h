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

#ifndef TENSORFLOW_CORE_IR_TF_OP_WRAPPER_H_
#define TENSORFLOW_CORE_IR_TF_OP_WRAPPER_H_

#include <cstddef>

#include "llvm/ADT/iterator_range.h"
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"

namespace mlir {
namespace tfg {

// Wrapper class exposing convenience methods to manipulate TensorFlow graph
// nodes uniformly.
class TFOp {
 public:
  // Wrap an operation. The operation can be null. The constructor must be
  // marked as implicit to support `llvm::dyn_cast`.
  TFOp(Operation *op = nullptr);  // NOLINT

  explicit TFOp(Operation &op) : TFOp(&op) {}

  // Support LLVM-style RTTI.
  static bool classof(Operation *op) {
    return isa<TFGraphDialect>(op->getDialect());
  }

  // Get the wrapped operation.
  Operation *getOperation() { return op_; }

  // Returns a pointer to the TensorFlow Graph Dialect. It nevers returns
  // nullptr.
  TFGraphDialect *getDialect() {
    return cast<TFGraphDialect>(op_->getDialect());
  }

  // Split the operands into data and control operands.
  std::tuple<OperandRange, OperandRange> splitOperands() {
    ControlType ctl_type = getDialect()->getControlType();
    OperandRange operands = op_->getOperands();
    unsigned num_ctl = 0;
    for (Value operand : llvm::reverse(operands)) {
      if (operand.getType() == ctl_type)
        ++num_ctl;
      else
        break;
    }
    unsigned split_idx = operands.size() - num_ctl;
    return std::make_tuple(operands.slice(0, split_idx),
                           operands.slice(split_idx, num_ctl));
  }

  // Returns the regular operands, the control operands will be excluded.
  OperandRange getNonControlOperands() { return std::get<0>(splitOperands()); }

  // The control operands are always after the regular inputs.
  OperandRange getControlOperands() { return std::get<1>(splitOperands()); }

  // Returns the control token produced by this operation.
  Value controlRet() { return op_->getResult(op_->getNumResults() - 1); }

  // Returns the non-control results produced by this operation.
  ResultRange getNonControlResults() {
    return op_->getResults().slice(0, op_->getNumResults() - 1);
  }

  // Returns the node name for this operation.
  StringAttr nameAttr();
  StringRef name();
  // Set a new node name for this operation.
  void setName(const Twine &name);
  void setName(StringAttr name);

  // Returns the requested device, which is also the "device" field in a
  // GraphDef.
  StringAttr requestedDeviceAttr();
  StringRef requestedDevice();
  // Set a new requested device for this operation.
  void setRequestedDevice(const Twine &requested_device);
  void setRequestedDevice(StringAttr requested_device);

  // Returns the assigned device, this field is set by placer in general.
  StringAttr assignedDeviceAttr();
  StringRef assignedDevice();
  // Set a new assigned device for this operation.
  void setAssignedDevice(const Twine &assigned_device);
  void setAssignedDevice(StringAttr assigned_device);

  // Returns the assigned TPU cluster name.
  StringAttr tpuReplicate();
  // Set the assigned TPU cluster name.
  void setTpuReplicate(StringAttr tpu_replicate);

  // Returns the device, preferring the assigned device if set, and the
  // requested device otherwise.
  StringAttr deviceAttr() {
    StringAttr device = assignedDeviceAttr();
    if (device) return device;
    return requestedDeviceAttr();
  }
  StringRef device() {
    StringAttr device_attr = deviceAttr();
    if (device_attr) return device_attr.getValue();
    return "";
  }

  // Forward `->` to the underlying operation, exposing the `Operation` methods.
  Operation *operator->() { return op_; }
  Operation &operator*() { return *op_; }

  // Converts to true if there is a wrapped operation.
  explicit operator bool() const { return op_; }

 private:
  // The wrapped operation.
  Operation *op_;
};

}  // namespace tfg
}  // namespace mlir

#endif  // TENSORFLOW_CORE_IR_TF_OP_WRAPPER_H_
