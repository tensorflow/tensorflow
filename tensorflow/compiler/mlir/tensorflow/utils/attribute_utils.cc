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

#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project

namespace mlir {
namespace TF {

LogicalResult HasValidCompilationAndReplicationAttributes(Operation& op) {
  auto replicate_attr = op.getAttrOfType<StringAttr>(kReplicationInfoAttr);
  auto compile_attr = op.getAttrOfType<StringAttr>(kCompileDeviceTypeAttr);
  // TODO(jiancai): we need to generalize the checks here once we allow more
  // general cases, e.g. only compilation but not replication marker.
  if (!replicate_attr && !compile_attr) return success();
  if (!replicate_attr || !compile_attr)
    return op.emitOpError() << "is expected to have either both or none of '"
                            << kReplicationInfoAttr << " and "
                            << kCompileDeviceTypeAttr << " attributes.";
  if (replicate_attr.getValue().empty())
    return op.emitOpError()
           << "has an empty " << kReplicationInfoAttr << " attribute.";
  if (compile_attr.getValue().empty())
    return op.emitOpError()
           << "has an empty " << kCompileDeviceTypeAttr << " attribute.";
  return success();
}

}  // namespace TF
}  // namespace mlir
