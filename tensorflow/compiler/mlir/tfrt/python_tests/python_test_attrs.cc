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

#include "tensorflow/compiler/mlir/tfrt/python_tests/python_test_attrs.h"

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
// Include the auto-generated dialect defs.
#include "tensorflow/compiler/mlir/tfrt/python_tests/python_test_attrs.cc.inc"

namespace mlir {
namespace tfrt {

void PythonTestAttrsDialect::initialize() {}

::mlir::LogicalResult PythonTestAttrsDialect::verifyRegionArgAttribute(
    ::mlir::Operation* op, unsigned regionIndex, unsigned argIndex,
    ::mlir::NamedAttribute attribute) {
  const auto& arg = op->getRegion(regionIndex).getArguments()[argIndex];

  // Only verify at the tensor level. We are interested in the correct attribute
  // values when processing the Tensorflow dialect IR.
  auto arg_type = arg.getType().dyn_cast<RankedTensorType>();
  if (!arg_type) return success();

  if (attribute.getName() == GetStaticTypeAttrName()) {
    auto type_attr = attribute.getValue().dyn_cast<TypeAttr>();
    if (!type_attr) {
      return op->emitError()
             << GetStaticTypeAttrName()
             << " argument attribute of other type than TypeAttr";
    }

    auto attr_type = type_attr.getValue().dyn_cast<RankedTensorType>();
    if (!attr_type) {
      return op->emitError()
             << GetStaticTypeAttrName()
             << " argument type attribute is not a ranked tensor type";
    }
    if (attr_type.getNumDynamicDims() > 0) {
      return op->emitError() << GetStaticTypeAttrName()
                             << " argument type attribute is a ranked tensor "
                                "type with dynamic dimensions";
    }
    if (attr_type.getRank() != arg_type.getRank()) {
      return op->emitError()
             << GetStaticTypeAttrName()
             << " argument type attribute is a ranked tensor type with a "
                "different rank than the rank of the argument tensor";
    }
    if (attr_type.getElementType() != arg_type.getElementType()) {
      return op->emitError()
             << GetStaticTypeAttrName()
             << " argument type attribute is a ranked tensor type with a "
                "different element type than the element type of the argument "
                "tensor";
    }
    const auto& attr_shape = attr_type.getShape();
    const auto& arg_shape = arg_type.getShape();
    for (int64_t i = 0; i < attr_shape.size(); ++i) {
      if (!arg_type.isDynamicDim(i) && arg_shape[i] != attr_shape[i]) {
        return op->emitError()
               << GetStaticTypeAttrName()
               << " argument type attribute is a ranked tensor type with a "
                  "shape that doesn't match the static dimensions of the "
                  "argument tensor";
      }
    }
  } else if (attribute.getName() == GetShapeValueAttrName()) {
    auto dense_attr = attribute.getValue().dyn_cast<DenseIntElementsAttr>();
    if (!dense_attr) {
      return op->emitError()
             << GetShapeValueAttrName()
             << " argument attribute is not a dense int elements attribute";
    }

    if (dense_attr.getType() != arg_type) {
      return op->emitError() << GetShapeValueAttrName()
                             << " argument elements attribute has a different "
                                "type than the argument type";
    }

    // We expect a valid shape value, therefore check that the dimension values
    // are not negative.
    for (auto&& dim : dense_attr) {
      if (dim.isNegative()) {
        return op->emitError()
               << GetShapeValueAttrName()
               << " argument elements attribute has a negative dimension value";
      }
    }
  }
  return success();
}

}  // namespace tfrt
}  // namespace mlir
