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
StringAttr TFOp::requestedDeviceAttr() {
  return op_.getAttrOfType<StringAttr>(getDialect()->getDeviceAttrIdentifier());
}
StringRef TFOp::requestedDevice() { return requestedDeviceAttr().getValue(); }
StringAttr TFOp::assignedDeviceAttr() {
  return op_.getAttrOfType<StringAttr>(
      getDialect()->getAssignedDeviceAttrIdentifier());
}
StringRef TFOp::assignedDevice() { return assignedDeviceAttr().getValue(); }

GraphFuncOp getCalledFunction(Operation *op, SymbolTable &symbol_table) {
  // Check if a node does indirect function call via PartitionedCallOp.
  if (op->getName().getStringRef() == "tfg.PartitionCall" ||
      op->getName().getStringRef() == "tfg.StatefulPartitionedCall") {
    auto func_attr = op->getAttrOfType<FuncAttr>("f");
    if (!func_attr) return {};
    GraphFuncOp callee = symbol_table.lookup<GraphFuncOp>(
        func_attr.getName().getLeafReference());
    if (callee) return callee;
  }
  return symbol_table.lookup<GraphFuncOp>(op->getName().stripDialect());
}

}  // namespace tfg
}  // namespace mlir
