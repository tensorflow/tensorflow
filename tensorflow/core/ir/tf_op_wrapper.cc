/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/tf_op_wrapper.h"

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"

namespace mlir {
namespace tfg {

TFOp::TFOp(Operation &op) : op_(op) {
  assert(isa<TFGraphDialect>(op.getDialect()));
}

StringAttr TFOp::nameAttr() {
  return op_.getAttrOfType<StringAttr>(getDialect()->getNameAttrIdentifier());
}

StringRef TFOp::name() { return nameAttr().getValue(); }

void TFOp::setName(const Twine &name) {
  setName(StringAttr::get(op_.getContext(), name.str()));
}

void TFOp::setName(StringAttr name) {
  op_.setAttr(getDialect()->getNameAttrIdentifier(), name);
}

StringAttr TFOp::requestedDeviceAttr() {
  return op_.getAttrOfType<StringAttr>(getDialect()->getDeviceAttrIdentifier());
}

StringRef TFOp::requestedDevice() { return requestedDeviceAttr().getValue(); }

void TFOp::setRequestedDevice(const Twine &device) {
  setRequestedDevice(StringAttr::get(op_.getContext(), device.str()));
}

void TFOp::setRequestedDevice(StringAttr device) {
  op_.setAttr(getDialect()->getDeviceAttrIdentifier(), device);
}

StringAttr TFOp::assignedDeviceAttr() {
  return op_.getAttrOfType<StringAttr>(
      getDialect()->getAssignedDeviceAttrIdentifier());
}

StringRef TFOp::assignedDevice() { return assignedDeviceAttr().getValue(); }

void TFOp::setAssignedDevice(const Twine &device) {
  setAssignedDevice(StringAttr::get(op_.getContext(), device.str()));
}

void TFOp::setAssignedDevice(StringAttr device) {
  op_.setAttr(getDialect()->getAssignedDeviceAttrIdentifier(), device);
}

}  // namespace tfg
}  // namespace mlir
